"""GPU feature construction, history and composition for the pinned temporal path.

CPU motion/Lanczos input resize is retained in the existing prefetch worker.
No host-sized sixteen-channel arrays are constructed in this session.
"""
from __future__ import annotations
import time
import numpy as np
import torch
from fast_temporal import tensor_history


def half(x):
    return x.to(torch.float16).to(torch.float32)


def scaled_color(color):
    return half(half(half(color) - 0.5) * 0.125)


def shift_mix(value):
    value = value & 0xFFFFFFFF
    mixed = value ^ (value >> ((value >> 28) + 4))
    return (mixed * 0x108EF2D9) & 0xFFFFFFFF


def uniform24(value):
    mixed = shift_mix(value)
    bits = (mixed >> 30) ^ (mixed >> 8)
    return (bits + 1).float() * 5.960464477539063e-8


def noise_from_seed(spatial_seed, frame_index):
    seed = spatial_seed ^ ((int(frame_index) * 0x9E3779B9) & 0xFFFFFFFF)
    multiplied = shift_mix(seed)
    mixed = multiplied ^ (multiplied >> 22)
    ua = uniform24(mixed * 0xCAA5B80D + 0x21DD796B)
    ub = uniform24(mixed * 0x83232C31 + 0x3463E0AC)
    uc = uniform24(mixed * 0x2C9277B5 + 0xAC564B05)
    ud = uniform24(mixed * 0xFA6DC5F9 + 0x4712A88E)
    ra, rb = (-2 * ua.log()).sqrt(), (-2 * uc.log()).sqrt()
    aa, ab = 6.2831854820251465 * ud, 6.2831854820251465 * ub
    return torch.stack((rb * aa.cos(), rb * aa.sin(), ra * ab.cos()), -1).half()


class FeatureBuilder:
    """One fixed shape; caches coordinates, seed and reusable half input storage."""
    def __init__(self, geometry, device):
        self.geometry = geometry
        h, w = geometry.output_height, geometry.output_width
        nh, nw = geometry.network_height, geometry.network_width
        y = torch.arange(nh, device=device, dtype=torch.int64)[:, None]
        x = torch.arange(nw, device=device, dtype=torch.int64)[None, :]
        self.seed = ((y * 0xD8163841) ^ (x * 0x8DA6B343) ^ 0x243F6A88) & 0xFFFFFFFF
        self.rows = torch.where(y < h, y, (2*h-2-y).clamp(min=0))
        self.cols = torch.where(x < w, x, (2*w-2-x).clamp(min=0))
        # Build these small cached axes with the exact reference arithmetic.
        # CUDA scalar division may instead multiply by an approximate reciprocal;
        # a tiny UV error can cross a later FP16 history-rounding boundary.
        self.u = torch.as_tensor((np.arange(w, dtype=np.float32)[None, :] + np.float32(0.5)) / np.float32(w), device=device)
        self.v = torch.as_tensor((np.arange(h, dtype=np.float32)[:, None] + np.float32(0.5)) / np.float32(h), device=device)
        self.features = torch.empty((1, nh, nw, 16), dtype=torch.float16, device=device)

    def build(self, color, frame_index, controls, *, history=None, motion=None, confidence=None):
        current = scaled_color(color)
        history_features = current
        if history is not None:
            reprojected = scaled_color(tensor_history(history, self.u + motion[..., 0],
                                                       self.v + motion[..., 1]))
            if confidence is None:
                history_features = reprojected
            else:
                mixed = current + confidence * (reprojected - current)
                history_features = torch.where(confidence == 0, current,
                                                torch.where(confidence == 1, reprojected, mixed))
        f = self.features[0]
        f[..., :3] = noise_from_seed(self.seed, frame_index)
        f[..., 3] = 1
        f[..., 4:7] = current[self.rows, self.cols]
        f[..., 7:10] = history_features[self.rows, self.cols]
        f[..., 10] = controls['normalized_style']
        f[..., 11] = controls['local_tone_strength']
        f[..., 12] = controls['local_structure_strength']
        f[..., 13:15] = -1
        f[..., 15] = 0
        # Keep FLOAT32 mixed history for composition: half input rounding must
        # not leak into the original blend formula.
        return self.features, history_features


def compose(head, color, history_features, confidence, *, temporal, intensity, blend_scale):
    head = head.float()
    predicted = (color + half(head[..., :3]) * 0.25).clamp(0, 1)
    if temporal:
        logit = half(head[..., 3:4])
        # Explicit operations preserve the reference's sigmoid arithmetic.
        rounded_scale = float(np.float16(blend_scale))
        alpha = ((1 / (1 + torch.exp(-logit))) * rounded_scale).clamp(0, 1)
        if confidence is not None:
            alpha = alpha * confidence
        reconstructed = history_features * 8 + 0.5
        output = predicted + alpha * (reconstructed - predicted)
        if intensity == 1:
            return output
    else:
        output = predicted
    strength = float(np.clip(np.float32(intensity), 0, 1))
    return (color + strength * (output - color)).clamp(0, 1)


def integer_downsample(output, height, width):
    h, w = output.shape[:2]
    if (h, w) == (height, width):
        return output
    if h % height or w % width or h // height != w // width:
        return None  # retain upstream Lanczos for non-integer processing scales
    factor = h // height
    return output.reshape(height, factor, width, factor, 3).mean(dim=(1, 3))


def resident_session_class(base, stats):
    class ResidentSession(base):
        @torch.inference_mode()
        def _process_prepared(self, prepared, control_mask=None):
            from mlxdlss.features import NetworkGeometry
            from mlxdlss.composition import resample, compose_detail
            if control_mask is not None:
                raise ValueError('Resident path does not support control masks; disable GPU_RESIDENT_TEMPORAL')
            if prepared.reset:
                self.reset()
                self.scene_cuts += 1
            if self.history is not None and tuple(self.history.shape) != prepared.color.shape:
                self.reset()
            started = time.perf_counter()
            events = []
            def mark(name, fn):
                a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                a.record()
                result = fn()
                b.record()
                events.append((name, a, b))
                return result
            with torch.cuda.device(self.pipeline.device):
                def upload():
                    def tensor(value):
                        return torch.as_tensor(np.ascontiguousarray(value), device=self.pipeline.device)
                    return (tensor(prepared.color),
                            None if prepared.confidence is None else tensor(prepared.confidence),
                            None if self.history is None else tensor(prepared.motion))
                color, confidence, motion = mark('resident_upload_gpu_seconds', upload)
                h, w = color.shape[:2]
                geometry = NetworkGeometry.vendor_aligned(w, h)
                if not hasattr(self, '_builder') or self._builder.geometry != geometry:
                    self._builder = FeatureBuilder(geometry, self.pipeline.device)
                had_history = self.history is not None
                network, history_features = mark('resident_features_gpu_seconds', lambda:
                    self._builder.build(color, self.frame_index, self._controls(), history=self.history,
                                        motion=motion, confidence=confidence))
                head = self.pipeline.run_device_features(network)[0, :h, :w]
                output = mark('resident_compose_gpu_seconds', lambda: compose(
                    head, color, history_features, confidence, temporal=had_history,
                    intensity=self.options.intensity, blend_scale=self.options.blend_scale))
                # Output has its own storage, not the replay graph's reused head.
                self.history = output
                self.previous = prepared.source  # frames are immutable in upstream prefetch
                self.frame_index += 1
                sh, sw = prepared.source.shape[:2]
                small = mark('resident_resize_gpu_seconds', lambda: integer_downsample(output, sh, sw))
                download_start = time.perf_counter()
                if small is None:
                    result = resample(output.cpu().numpy(), sw, sh)
                else:
                    result = small.cpu().numpy()
                stats['resident_download_and_fallback_seconds'] = stats.get(
                    'resident_download_and_fallback_seconds', 0.0) + time.perf_counter() - download_start
                # Host copy synchronized all queued work. Event times below are
                # GPU intervals, while session wall includes launches and waits.
                for name, a, b in events:
                    stats[name] = stats.get(name, 0.0) + a.elapsed_time(b) / 1000
                result = compose_detail(prepared.source, result, detail_strength=self.options.detail_strength,
                                        colour_strength=self.options.colour_strength, radius=self.options.detail_radius)
                if not np.isfinite(result).all():
                    raise RuntimeError('Nonfinite resident output; refusing export')
                stats['resident_session_wall_seconds'] = stats.get('resident_session_wall_seconds', 0.0) + time.perf_counter() - started
                return result
    return ResidentSession


@torch.inference_mode()
def validate_resident(device):
    """GPU numeric self-checks; complete real-video parity remains mandatory."""
    from mlxdlss import features as f, temporal as t
    from mlxdlss.composition import compose_head, resample
    rng = np.random.default_rng(12)
    report = {'passed': False, 'noise_max': 0.0, 'noise_mae': 0.0,
              'feature_nonnoise_max': 0.0, 'composition_max': 0.0, 'resize_max': 0.0,
              'history_features_max': 0.0, 'composition_same_history_max': 0.0,
              'worst_composition_case': None}
    for h, w in ((1, 1), (63, 97), (320, 320)):
        color = rng.random((h, w, 3), dtype=np.float32)
        history = rng.random((h, w, 3), dtype=np.float32)
        motion = rng.uniform(-0.03, 0.03, (h, w, 2)).astype(np.float32)
        confidence = rng.random((h, w, 1), dtype=np.float32)
        confidence.reshape(-1)[::3] = 0
        confidence.reshape(-1)[1::3] = 1
        geometry = f.NetworkGeometry.vendor_aligned(w, h)
        builder = FeatureBuilder(geometry, device)
        controls = f.PROFILES['natural']
        for index in (0, 1, 17):
            temporal = index != 0
            kwargs = {} if not temporal else dict(history=torch.as_tensor(history, device=device),
                motion=torch.as_tensor(motion, device=device), confidence=torch.as_tensor(confidence, device=device))
            actual, history_f = builder.build(torch.as_tensor(color, device=device), index, controls, **kwargs)
            if temporal:
                logical = t.make_temporal_features(color, history, motion, frame_index=index,
                                                   history_confidence=confidence, **controls)
                expected = t.extend_features(logical, geometry, index)
            else:
                expected = f.make_features(color, frame_index=index, geometry=geometry, **controls)
                logical = expected[:h, :w]
            actual_np = actual[0].float().cpu().numpy()
            delta = np.abs(actual_np - expected.astype(np.float16).astype(np.float32))
            if not np.isfinite(actual_np).all():
                raise RuntimeError('Nonfinite GPU feature construction')
            report['noise_max'] = max(report['noise_max'], float(delta[..., :3].max()))
            report['noise_mae'] = max(report['noise_mae'], float(delta[..., :3].mean()))
            report['feature_nonnoise_max'] = max(report['feature_nonnoise_max'], float(delta[..., 3:].max()))
            report['history_features_max'] = max(report['history_features_max'],
                float(np.abs(history_f.cpu().numpy() - logical[..., 7:10]).max()))
            head = rng.normal(0, 0.2, (h, w, 4)).astype(np.float16).astype(np.float32)
            for intensity in (0.0, 0.8, 1.0):
                actual_rgb = compose(torch.as_tensor(head, device=device), torch.as_tensor(color, device=device),
                    history_f, torch.as_tensor(confidence, device=device), temporal=temporal,
                    intensity=intensity, blend_scale=t.BLEND_SCALE).cpu().numpy()
                expected_rgb = (t.compose_temporal(head, color, logical, history_confidence=confidence,
                                                   intensity=intensity) if temporal else compose_head(head, color, intensity=intensity))
                error = float(np.abs(actual_rgb-expected_rgb).max())
                if error > report['composition_max']:
                    report['composition_max'] = error
                    report['worst_composition_case'] = {'height': h, 'width': w, 'frame_index': index, 'intensity': intensity}
                same_history_rgb = compose(torch.as_tensor(head, device=device), torch.as_tensor(color, device=device),
                    torch.as_tensor(np.ascontiguousarray(logical[..., 7:10]), device=device),
                    torch.as_tensor(confidence, device=device), temporal=temporal,
                    intensity=intensity, blend_scale=t.BLEND_SCALE).cpu().numpy()
                report['composition_same_history_max'] = max(report['composition_same_history_max'],
                    float(np.abs(same_history_rgb - expected_rgb).max()))
    for factor in (1, 2, 3, 4):
        image = rng.random((16*factor, 24*factor, 3), dtype=np.float32)
        actual = integer_downsample(torch.as_tensor(image, device=device), 16, 24).cpu().numpy()
        report['resize_max'] = max(report['resize_max'], float(np.abs(actual-resample(image,24,16)).max()))
    # Float32 transcendental libraries can differ at half-quantization edges.
    # Tolerances are narrow; a real-frame comparison additionally gates export.
    if (report['noise_max'] > 0.008 or report['noise_mae'] > 2e-5 or
        report['feature_nonnoise_max'] > 0.00025 or report['composition_max'] > 2e-5 or report['resize_max'] > 2e-6):
        raise RuntimeError(f'Resident feature/composition parity failed: {report}')
    report['passed'] = True
    return report
