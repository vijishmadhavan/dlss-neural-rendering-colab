"""Bounded motion guides and full-resolution GPU five-tap history sampling.

Guide reduction changes optical-flow estimates, not neural rendering resolution.
Patches are scoped to one serial run and restored after the prefetch worker joins.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import time
import numpy as np
import torch


def guide_size(width, height, max_side):
    ratio = min(1.0, max_side / max(width, height))
    return max(1, round(width * ratio)), max(1, round(height * ratio))


@torch.inference_mode()
def gpu_history(history, u, v, device):
    image = torch.as_tensor(np.ascontiguousarray(history), device=device)
    u = torch.as_tensor(np.ascontiguousarray(u), device=device)
    v = torch.as_tensor(np.ascontiguousarray(v), device=device)
    return tensor_history(image, u, v).cpu().numpy()


def rounded_divide(numerator, denominator):
    """Round the quotient to FP32 without a FP32 reciprocal approximation.

    History values are subsequently quantized to half; a one-ULP division
    difference can cross that boundary and become a visible blend difference.
    The temporary FP64 quotient is rounded back immediately, not propagated.
    """
    return (numerator.double() / denominator.double()).float()


def tensor_history(image, u, v):
    """Same five taps, accepting/returning tensors without a host round trip."""
    height, width = image.shape[:2]
    def coordinates(normalized, dimension):
        pixel = normalized * dimension - 0.5
        base_index = torch.floor(pixel)
        t = (pixel - base_index).clamp(0, 1)
        square, cube = t * t, t * t * t
        w0 = -0.5 * t + square - 0.5 * cube
        w1 = 1 - 2.5 * square + 1.5 * cube
        w2 = 0.5 * t + 2 * square - 1.5 * cube
        w3 = -0.5 * square + 0.5 * cube
        g = w1 + w2
        base = base_index + 0.5
        return ((base - 1).clamp(0.5, dimension - 0.5),
                (base + rounded_divide(w2, g)).clamp(0.5, dimension - 0.5),
                (base + 2).clamp(0.5, dimension - 0.5), w0, w3, g)
    def sample(x, y):
        px, py = x - 0.5, y - 0.5
        x0 = px.floor().clamp(0, width - 1).long()
        y0 = py.floor().clamp(0, height - 1).long()
        x1, y1 = (x0 + 1).clamp(max=width-1), (y0 + 1).clamp(max=height-1)
        tx, ty = (px - x0).clamp(0, 1)[..., None], (py - y0).clamp(0, 1)[..., None]
        top = image[y0, x0] * (1-tx) + image[y0, x1] * tx
        bottom = image[y1, x0] * (1-tx) + image[y1, x1] * tx
        return top * (1-ty) + bottom * ty
    x0, xm, x3, xw0, xw3, xg = coordinates(u, width)
    y0, ym, y3, yw0, yw3, yg = coordinates(v, height)
    total, weight_sum = 0, 0
    for x, y, weight in ((x0, ym, xw0*yg), (xm, y0, xg*yw0),
                          (xm, ym, xg*yg), (xm, y3, xg*yw3), (x3, ym, xw3*yg)):
        total = total + weight[..., None] * sample(x, y)
        weight_sum = weight_sum + weight
    return rounded_divide(total, weight_sum[..., None])


def validate_history(device):
    from mlxdlss.temporal import sample_history
    rng = np.random.default_rng(73)
    worst = 0.0
    for height, width in ((1, 1), (31, 47), (73, 96)):
        history = rng.random((height, width, 3), dtype=np.float32)
        yy, xx = np.indices((height, width), dtype=np.float32)
        for offset in (0.0, 0.032, -0.1):
            u, v = (xx+0.5)/width + offset, (yy+0.5)/height - offset
            expected = sample_history(history, u, v)
            actual = gpu_history(history, u, v, device)
            error = float(np.max(np.abs(expected - actual)))
            worst = max(worst, error)
            if not np.isfinite(actual).all() or error > 2e-5:
                raise RuntimeError(f"GPU history sampler parity failed: max error {error}")
    return {"max_absolute_error": worst, "passed": True, "tolerance": 2e-5}


@contextmanager
def temporal_runtime(*, max_side, device, use_gpu_history, stats, resident=False):
    from mlxdlss import temporal as t
    import cv2
    if max_side < 64:
        raise ValueError("Motion-guide longest side must be at least 64")
    original_class = t.FlowMotionEstimator
    class BoundedFlow(original_class):
        def estimate(self, current, previous, *, scene_cut_threshold=0.3):
            h, w = current.shape[:2]
            gw, gh = guide_size(w, h, max_side)
            if (gw, gh) == (w, h):
                return super().estimate(current, previous, scene_cut_threshold=scene_cut_threshold)
            small_current = cv2.resize(current, (gw, gh), interpolation=cv2.INTER_AREA)
            small_previous = cv2.resize(previous, (gw, gh), interpolation=cv2.INTER_AREA)
            estimate = super().estimate(small_current, small_previous,
                                        scene_cut_threshold=scene_cut_threshold)
            # Motion already uses normalized UV units: do NOT multiply by scale.
            estimate.motion_uv = cv2.resize(estimate.motion_uv, (w, h), interpolation=cv2.INTER_LINEAR)
            estimate.confidence = cv2.resize(estimate.confidence, (w, h),
                                              interpolation=cv2.INTER_NEAREST)[..., None]
            return estimate

    def timed(name, function):
        def call(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                stats[name] = stats.get(name, 0.0) + time.perf_counter() - start
        return call
    with ExitStack() as stack:
        stack.enter_context(patch.object(t, "FlowMotionEstimator", BoundedFlow))
        if use_gpu_history:
            stack.enter_context(patch.object(t, "sample_history",
                lambda history, u, v: gpu_history(history, u, v, device)))
        # Inclusive stage times: history sampling is part of temporal_features.
        for name in ("sample_history", "make_features", "make_temporal_features", "extend_features",
                     "compose_head", "compose_temporal", "compose_detail", "resolve_motion",
                     "prepare_temporal_frame"):
            stack.enter_context(patch.object(t, name, timed(name, getattr(t, name))))
        # Video streaming imports these two names directly, including its
        # CPU-only motion prefetch path; cover both it and session.process().
        from mlxdlss import video_pipeline
        for name in ("resolve_motion", "prepare_temporal_frame"):
            stack.enter_context(patch.object(video_pipeline, name, getattr(t, name)))
        if resident:
            from resident_temporal import resident_session_class
            session_class = resident_session_class(t.TemporalSession, stats)
            stack.enter_context(patch.object(t, "TemporalSession", session_class))
            stack.enter_context(patch.object(video_pipeline, "TemporalSession", session_class))
        yield
