"""Benchmark-gated portable NR runner; no Wine, Vulkan, or NGX execution."""
from __future__ import annotations

import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mlxdlss.pipeline import NeuralRenderingPipeline, load_weights
from mlxdlss.temporal import TemporalOptions, TemporalSession
from mlxdlss.video import ConvertOptions, convert
from fast_kernels import build_fast_model, validate_kernels
from graph_replay import ReplayModel, profile_network
from fast_temporal import temporal_runtime, validate_history
from resident_temporal import validate_resident

IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def parity_metrics(reference, candidate):
    diff = np.abs(reference - candidate)
    return {"mae": float(diff.mean()), "p99": float(np.quantile(diff, 0.99)),
            "max": float(diff.max())}


def video_info(path):
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-show_entries",
        "stream=codec_type,width,height,nb_read_frames,avg_frame_rate",
        "-of", "json", str(path)], text=True))
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    return {"width": int(video["width"]), "height": int(video["height"]),
            "frames": int(video["nb_read_frames"]), "fps": video["avg_frame_rate"],
            "audio": any(s["codec_type"] == "audio" for s in data["streams"])}


def sample_frames(path, count):
    path = Path(path)
    if path.suffix.lower() in IMAGES:
        with Image.open(path) as image:
            return [np.asarray(image.convert("RGB"), np.float32) / 255], None
    info = video_info(path)
    count = min(count, info["frames"])
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-an",
        "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    expected = count * info["width"] * info["height"] * 3
    if len(raw) != expected:
        raise RuntimeError(f"Sample decode: got {len(raw)} bytes, expected {expected}")
    frames = np.frombuffer(raw, np.uint8).reshape(count, info["height"], info["width"], 3)
    return [frame.astype(np.float32) / 255 for frame in frames], info


class TimedPipeline(NeuralRenderingPipeline):
    def __init__(self, weights, device):
        super().__init__(weights, device=device, precision="fast")
        self.reuse_buffers = False
        self._host_input = self._device_input = self._host_output = None
        self.reset_stats()

    def reset_stats(self):
        self.stats = {"calls": 0, "network_seconds": 0.0,
                      "transfer_seconds": 0.0, "input_staging_seconds": 0.0,
                      "network_plus_io_wall_seconds": 0.0}

    @torch.inference_mode()
    def run_device_features(self, features):
        """Resident session: sixteen-channel input and model head stay on CUDA."""
        if (features.ndim != 4 or features.shape[-1] != 16 or features.dtype != torch.float16
                or features.device.type != 'cuda' or any(v % 64 for v in features.shape[1:3])):
            raise ValueError('Expected CUDA float16 NHWC features, aligned to 64')
        self._device_input = features  # retain latest real input for optional operator profiling
        started = time.perf_counter()
        a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        a.record()
        head = self.model(features)
        b.record()
        b.synchronize()
        self.stats['calls'] += 1
        self.stats['network_seconds'] += a.elapsed_time(b) / 1000
        self.stats['network_plus_io_wall_seconds'] += time.perf_counter() - started
        return head

    @torch.inference_mode()
    def run_features_batch(self, features):
        started = time.perf_counter()
        features = np.ascontiguousarray(features, dtype=np.float32)
        if features.ndim != 4 or features.shape[-1] != 16 or any(v % 64 for v in features.shape[1:3]):
            raise ValueError("Expected NHWC features with 16 channels and dimensions divisible by 64")
        with torch.cuda.device(self.device):
            cpu_tensor = torch.from_numpy(features)
            if self.reuse_buffers:
                if self._host_input is None or self._host_input.shape != cpu_tensor.shape:
                    self._host_input = torch.empty(cpu_tensor.shape, dtype=torch.float32, pin_memory=True)
                    self._device_input = torch.empty(cpu_tensor.shape, dtype=torch.float16, device=self.device)
                self._host_input.copy_(cpu_tensor)
            staged = time.perf_counter()
            a, b, c, d = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            a.record()
            if self.reuse_buffers:
                self._device_input.copy_(self._host_input, non_blocking=True)
                tensor = self._device_input
            else:
                tensor = cpu_tensor.to(self.device, torch.float16)
            b.record()
            head = self.model(tensor)
            c.record()
            if self.reuse_buffers:
                if self._host_output is None or self._host_output.shape != head.shape:
                    self._host_output = torch.empty(head.shape, dtype=torch.float32, pin_memory=True)
                self._host_output.copy_(head.float(), non_blocking=True)
                d.record()
                d.synchronize()
                result = self._host_output.numpy().copy()
            else:
                result = head.float().cpu().numpy()
                d.record()
                d.synchronize()
            self.stats["calls"] += 1
            self.stats["network_seconds"] += b.elapsed_time(c) / 1000
            self.stats["transfer_seconds"] += (a.elapsed_time(b) + c.elapsed_time(d)) / 1000
            self.stats["input_staging_seconds"] += staged - started
            self.stats["network_plus_io_wall_seconds"] += time.perf_counter() - started
            if not np.isfinite(result).all():
                raise RuntimeError("Nonfinite model output; refusing to encode corrupted frames")
            return result


class FastEngine:
    def __init__(self, weights, *, device="cuda", enhance=None, temporal=True, motion_max_side=640,
                 resident_temporal=True):
        self.started = time.perf_counter()
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The fast runner requires working CUDA PyTorch and Triton")
        self.enhance = dict(enhance or {})
        self.temporal = temporal
        self.resident_temporal = bool(resident_temporal and temporal)
        self.motion_max_side = int(motion_max_side)
        self.temporal_stats = {}
        self.selected_fast = False
        self.approved = False
        with torch.cuda.device(self.device):
            self.validation = validate_kernels(self.device)
            self.history_validation = validate_history(self.device)
            self.resident_validation = validate_resident(self.device) if self.resident_temporal else {'enabled': False}
            print('Resident GPU self-check:', self.resident_validation, flush=True)
            loaded = load_weights(weights)
            self.pipeline = TimedPipeline(loaded, self.device)
            self.baseline = self.pipeline.model
            self.optimized = ReplayModel(build_fast_model(loaded, self.device, self.validation))
        self.startup_seconds = time.perf_counter() - self.started
        self.environment = {"gpu": torch.cuda.get_device_name(self.device),
                            "runner_revision": "v4.2-parity-isolation",
                            "compute_capability": list(torch.cuda.get_device_capability(self.device)),
                            "torch": torch.__version__, "cuda": torch.version.cuda,
                            "weights": str(weights)}
        try:
            self.environment["drivers"] = subprocess.check_output([
                "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
        except (OSError, subprocess.SubprocessError):
            self.environment["drivers"] = "unavailable"

    def _select(self, fast):
        self.selected_fast = fast
        self.temporal_stats = {}
        if not fast:
            self.baseline.to(self.device)
        self.pipeline.model = self.optimized if fast else self.baseline
        self.pipeline.reuse_buffers = fast
        self.pipeline.reset_stats()

    def _temporal_context(self):
        return temporal_runtime(max_side=self.motion_max_side, device=self.device,
                                use_gpu_history=self.selected_fast, stats=self.temporal_stats,
                                resident=self.selected_fast and self.resident_temporal)

    def _sequence(self, frames):
        with self._temporal_context():
            return self._sequence_scoped(frames)

    def _sequence_scoped(self, frames):
        # Resolve after entering scoped patches, like the streaming video path.
        from mlxdlss.temporal import TemporalSession as ActiveSession
        session = ActiveSession(self.pipeline, options=TemporalOptions(**self.enhance)) if self.temporal else None
        outputs, seconds = [], []
        for i, frame in enumerate(frames):
            started = time.perf_counter()
            result = session.process(frame) if session else self.pipeline.enhance(frame, frame_index=i, **self.enhance).image
            torch.cuda.synchronize(self.device)
            seconds.append(time.perf_counter() - started)
            if not np.isfinite(result).all():
                raise RuntimeError("Nonfinite pixels in benchmark; refusing this implementation")
            outputs.append(result.copy())
        return outputs, seconds

    def benchmark(self, source, report_path, *, count=4, target_seconds=60):
        self.approved = False
        if count < 1 or target_seconds <= 0:
            raise ValueError("Benchmark frame count and target seconds must be positive")
        begin = time.perf_counter()
        frames, info = sample_frames(source, count)
        height, width = frames[0].shape[:2]
        scale = self.enhance.get('processing_scale', 1.0)
        ph, pw = round(height * scale), round(width * scale)
        resolution = {'input': [width, height], 'internal_processing': [pw, ph],
                      'network_padded': [((max(320, pw)+63)//64)*64, ((max(320, ph)+63)//64)*64],
                      'output': [width, height]}
        print(f'Resolution contract: input {width}x{height}; internal {pw}x{ph}; '
              f'output {width}x{height}. Processing scale is NOT output upscaling.', flush=True)
        decode_seconds = time.perf_counter() - begin
        modes, outputs = {}, {}
        for name, fast in (("baseline", False), ("fused", True)):
            self._select(fast)
            print(f"{name}: first full-resolution call (includes any kernel compilation)...", flush=True)
            _, cold = self._sequence(frames[:1])
            self.pipeline.reset_stats()
            self.temporal_stats = {}
            print(f"{name}: comparing {len(frames)} consecutive full-resolution frames...", flush=True)
            outputs[name], durations = self._sequence(frames)
            modes[name] = {"cold_first_frame_seconds": cold[0], "frame_seconds": durations,
                           "mean_seconds": statistics.mean(durations),
                           "median_seconds": statistics.median(durations),
                           "temporal_stage_seconds_inclusive": dict(self.temporal_stats),
                           **self.pipeline.stats}
        errors = []
        for a, b in zip(outputs["baseline"], outputs["fused"]):
            diff = np.abs(a - b)
            errors.append({"mae": float(diff.mean()), "p99": float(np.quantile(diff, 0.99)),
                           "max": float(diff.max())})
        # Small sample comparison against the old portable pipeline, not a
        # claim of equality with NVIDIA NGX or all future input frames.
        passed = all(e["mae"] <= 0.5 / 255 and e["p99"] <= 2 / 255 for e in errors)
        report = {"environment": self.environment, "kernel_checks": self.validation,
                  "source": str(source), "source_info": info,
                  "sample_frames": len(frames), "shape": list(frames[0].shape),
                  "resolution_contract": resolution,
                  "controls": self.enhance, "temporal": self.temporal,
                  "startup_seconds": self.startup_seconds, "sample_decode_seconds": decode_seconds,
                  "modes": modes, "parity_errors": errors, "parity_passed": passed,
                  "graph_replay": getattr(getattr(self, "optimized", None), "reports", []),
                  "motion_guide_max_side": getattr(self, "motion_max_side", 640),
                  "comparison_scope": "Both paths use the SAME bounded motion guides; this tests compute parity, not parity with full-resolution optical flow.",
                  "history_sampler_check": getattr(self, "history_validation", {}),
                  "resident_gpu_check": getattr(self, "resident_validation", {}),
                  "gpu_resident_temporal": getattr(self, "resident_temporal", False),
                  "output_resolution_policy": "Source size; processing_scale is internal supersampling, NOT output upscaling",
                  "measured_sample_speedup": modes["baseline"]["mean_seconds"] / modes["fused"]["mean_seconds"],
                  "target_seconds": target_seconds,
                  "estimated_processing_seconds_without_video_io": None if info is None else info["frames"] * modes["fused"]["mean_seconds"],
                  "benchmark_wall_seconds": time.perf_counter() - begin}
        Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
        self.calibration_seconds = self.startup_seconds + report["benchmark_wall_seconds"]
        print(json.dumps(report, indent=2), flush=True)
        if not passed:
            if getattr(self, 'resident_temporal', False):
                diagnostic_path = Path(report_path).with_suffix('.diagnostic.json')
                print('Isolating first-frame mismatch; export remains blocked.', flush=True)
                try:
                    diagnostic = self.diagnose_parity(source, diagnostic_path, frame=frames[0],
                        baseline_output=outputs['baseline'][0], resident_output=outputs['fused'][0])
                    report['failure_diagnostic'] = diagnostic
                except Exception as exc:
                    report['failure_diagnostic_error'] = repr(exc)
                report['benchmark_wall_seconds'] = time.perf_counter() - begin
                self.calibration_seconds = self.startup_seconds + report['benchmark_wall_seconds']
                Path(report_path).write_text(json.dumps(report, indent=2) + '\n')
            raise RuntimeError(f"Real-frame comparison failed. See {report_path}; full processing is blocked")
        self.approved = True
        self._select(True)
        self.baseline.to("cpu")
        torch.cuda.empty_cache()
        print("Portable-output comparison passed. Full video timing is still needed to confirm the speed target.", flush=True)
        return report

    def diagnose_parity(self, source, report_path, *, frame=None, baseline_output=None, resident_output=None):
        """2x2 isolation: model implementation vs feature/composition backend.

        First-frame only: intentionally no previous history, flow or cut decisions.
        No fallback or approval; this reports evidence for the next targeted fix.
        """
        from mlxdlss.features import NetworkGeometry, make_features, PROFILES
        from mlxdlss.composition import resample
        from resident_temporal import FeatureBuilder
        self.approved = False
        begin = time.perf_counter()
        if frame is None:
            frame = sample_frames(source, 1)[0][0]
        original_resident = self.resident_temporal
        timings = {}
        try:
            self.resident_temporal = False
            if baseline_output is None:
                self._select(False)
                result, seconds = self._sequence([frame])
                baseline_output, timings['baseline_cpu'] = result[0], seconds[0]
            self._select(True)
            result, seconds = self._sequence([frame])
            optimized_cpu = result[0]
            timings['optimized_cpu'] = seconds[0]
            self.resident_temporal = True
            if resident_output is None:
                self._select(True)
                result, seconds = self._sequence([frame])
                resident_output, timings['optimized_resident'] = result[0], seconds[0]
            # Same original model, new preparation/composition path.
            self._select(True)
            self.baseline.to(self.device)
            self.pipeline.model = self.baseline
            result, seconds = self._sequence([frame])
            baseline_resident = result[0]
            timings['baseline_resident'] = seconds[0]

            scale = self.enhance.get('processing_scale', 1.0)
            h, w = (round(d * scale) for d in frame.shape[:2])
            color = resample(frame, w, h)
            geometry = NetworkGeometry.vendor_aligned(w, h)
            controls = dict(PROFILES[self.enhance.get('profile', 'standard')])
            for key in controls:
                if self.enhance.get(key) is not None:
                    controls[key] = self.enhance[key]
            reference = make_features(color, frame_index=0, geometry=geometry, **controls).astype(np.float16)
            with torch.inference_mode():
                builder = FeatureBuilder(geometry, self.device)
                actual, _ = builder.build(torch.as_tensor(color, device=self.device), 0, controls)
                actual = actual[0].cpu().numpy()
            noise_diff = np.abs(reference[..., :3].astype(np.float32)-actual[..., :3].astype(np.float32))
            report = {
                'source': str(source), 'input_shape': list(frame.shape), 'first_frame_only': True,
                'model_change_with_cpu_preparation': parity_metrics(baseline_output, optimized_cpu),
                'preparation_change_with_optimized_model': parity_metrics(optimized_cpu, resident_output),
                'preparation_change_with_original_model': parity_metrics(baseline_output, baseline_resident),
                'combined_change': parity_metrics(baseline_output, resident_output),
                'actual_input_features': {
                    'noise_mae': float(noise_diff.mean()), 'noise_max': float(noise_diff.max()),
                    'noise_changed_values': int(np.count_nonzero(reference[..., :3] != actual[..., :3])),
                    'nonnoise_changed_values': int(np.count_nonzero(reference[..., 3:] != actual[..., 3:])),
                },
                'timings': timings, 'diagnostic_seconds': time.perf_counter()-begin,
                'export_approved': False,
            }
            Path(report_path).write_text(json.dumps(report, indent=2)+'\n')
            print('Parity isolation:', json.dumps(report, indent=2), flush=True)
            print('Diagnostic report:', report_path, flush=True)
            return report
        finally:
            self.resident_temporal = original_resident
            self._select(True)
            self.approved = False

    def profile(self, directory):
        if not self.approved or self.pipeline._device_input is None:
            raise RuntimeError("Run the benchmark first to obtain real feature inputs")
        profile_network(self.optimized.model, self.pipeline._device_input, directory)

    def process(self, source, destination, *, limit=None, target_seconds=60, crf=14):
        if not self.approved:
            raise RuntimeError("Run the real-frame benchmark and comparison before processing")
        source, destination = Path(source), Path(destination)
        if destination.exists():
            raise FileExistsError(destination)
        self._select(True)
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        if source.suffix.lower() in IMAGES:
            with Image.open(source) as image:
                frame = np.asarray(image.convert("RGB"), np.float32) / 255
            result = self.pipeline.enhance(frame, **self.enhance).image
            if not np.isfinite(result).all():
                raise RuntimeError("Nonfinite output pixels")
            Image.fromarray(np.clip(result * 255 + 0.5, 0, 255).astype(np.uint8)).save(destination)
            output_info = {"frames": 1, "width": frame.shape[1], "height": frame.shape[0], "audio": False}
        else:
            info = video_info(source)
            if info["width"] % 2 or info["height"] % 2:
                raise ValueError("H.264 yuv420p needs even width/height; this runner will not resize your video")
            expected = min(limit, info["frames"]) if limit is not None else info["frames"]
            if expected < 1:
                raise ValueError("No frames to process")
            options = ConvertOptions(
                frame_limit=limit, temporal=self.temporal, prefetch=True,
                audio="none" if limit is not None else "copy", enhance=self.enhance,
                status_interval=5,
                encode_args=["-frames:v", str(expected), "-c:v", "libx264", "-crf", str(crf),
                             "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            with self._temporal_context():
                converted = convert(source, destination, self.pipeline, options)
            processing_seconds = time.perf_counter() - started
            output_info = video_info(destination)
            if (output_info["frames"], output_info["width"], output_info["height"], output_info["fps"]) != (expected, info["width"], info["height"], info["fps"]):
                raise RuntimeError(f"Unverified output retained for diagnosis: {output_info}; expected {info}, {expected} frames")
            if limit is None and info["audio"] and not output_info["audio"]:
                raise RuntimeError("Output lost source audio")
            output_info["scene_cuts"] = converted.scene_cuts
        if source.suffix.lower() in IMAGES:
            processing_seconds = time.perf_counter() - started
        report = {"environment": self.environment, "output": str(destination),
                  **output_info, "controls": self.enhance, "temporal": self.temporal,
                  "processing_seconds_including_io": processing_seconds,
                  "total_seconds_including_verification": time.perf_counter() - started,
                  "throughput_fps": output_info["frames"] / processing_seconds,
                  "model_load_and_kernel_selftest_seconds": self.startup_seconds,
                  "calibration_seconds": self.calibration_seconds,
                  "first_use_compute_seconds_including_calibration": self.calibration_seconds + processing_seconds,
                  "target_seconds": target_seconds,
                  "target_met_for_this_run": processing_seconds <= target_seconds if limit is None else None,
                  "target_met_including_calibration": self.calibration_seconds + processing_seconds <= target_seconds if limit is None else None,
                  "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
                  "gpu_stage_totals": dict(self.pipeline.stats),
                  "motion_guide_max_side": getattr(self, "motion_max_side", 640),
                  "gpu_resident_temporal": getattr(self, "resident_temporal", False) and source.suffix.lower() not in IMAGES,
                  "output_resolution_policy": "Source size; processing_scale is internal supersampling, NOT output upscaling",
                  "temporal_stage_seconds_inclusive": dict(getattr(self, "temporal_stats", {})),
                  "graph_replay": getattr(getattr(self, "optimized", None), "reports", [])}
        timing_path = destination.with_suffix(".timings.json")
        timing_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Verified {output_info['frames']} frames: {destination}\n"
              f"Processing: {processing_seconds:.2f}s, {report['throughput_fps']:.2f} fps; timings: {timing_path}", flush=True)
        if limit is None:
            print(f"Warm job target {target_seconds:.0f}s: {'MET' if report['target_met_for_this_run'] else 'NOT MET'}; "
                  f"including calibration: {'MET' if report['target_met_including_calibration'] else 'NOT MET'}", flush=True)
        return report
