"""Fixed-shape CUDA replay with changed-input checks and measured selection."""
from __future__ import annotations

import time
import torch


class ReplayModel:
    """Single-stream, single-shape cache. Returned storage is reused next call.

    The caller consumes/composes output before the next call. Temporal state is
    in the input tensor, not in this graph, and is copied on EVERY replay.
    """
    def __init__(self, model):
        self.model = model
        self.graph = self.input = self.output = None
        self.key = None
        self.reports = []

    @staticmethod
    def signature(x):
        return (tuple(x.shape), x.dtype, x.device)

    @staticmethod
    def measured(fn, device, repeats=3):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize(device)
        return (time.perf_counter() - started) / repeats

    @torch.inference_mode()
    def prepare(self, x):
        self.key = self.signature(x)
        self.graph = self.input = self.output = None
        started = time.perf_counter()
        report = {"shape": list(x.shape), "enabled": False}
        self.reports.append(report)
        # Avoid hiding a failed capture behind endless attempts every frame.
        try:
            with torch.cuda.device(x.device):
                self.input = x.clone()
                stream = torch.cuda.Stream(device=x.device)
                stream.wait_stream(torch.cuda.current_stream(x.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self.model(self.input)
                torch.cuda.current_stream(x.device).wait_stream(stream)
                torch.cuda.synchronize(x.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    self.output = self.model(self.input)
                # Two distinct feature tensors detect accidentally frozen input.
                for probe in (x, x * 0.875):
                    expected = self.model(probe).clone()
                    self.input.copy_(probe)
                    graph.replay()
                    if not torch.isfinite(self.output).all().item() or not torch.equal(expected, self.output):
                        raise RuntimeError("Graph replay changed model output; retaining eager model")
                self.input.copy_(x)
                eager_seconds = self.measured(lambda: self.model(x), x.device)
                def replay():
                    self.input.copy_(x)
                    graph.replay()
                replay_seconds = self.measured(replay, x.device)
                report.update(eager_seconds=eager_seconds, replay_seconds=replay_seconds,
                              speedup=eager_seconds / replay_seconds, changed_input_exact=True)
                if replay_seconds < eager_seconds:
                    self.graph = graph
                    report["enabled"] = True
                else:
                    report["reason"] = "Replay was not faster on this shape"
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            report["reason"] = str(exc)
        if self.graph is None:
            self.input = self.output = None
        report["setup_seconds"] = time.perf_counter() - started
        print("CUDA graph:", report, flush=True)

    @torch.inference_mode()
    def __call__(self, x):
        if self.signature(x) != self.key:
            self.prepare(x)
        if self.graph is None:
            return self.model(x)
        self.input.copy_(x)
        self.graph.replay()
        return self.output


@torch.inference_mode()
def profile_network(model, features, directory):
    """Profile EAGER GPU kernels separately from time between launches."""
    from pathlib import Path
    import json
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as prof:
        model(features)
        torch.cuda.synchronize(features.device)
    prof.export_chrome_trace(str(directory / "network_trace.json"))
    events = prof.key_averages(group_by_input_shape=True)
    (directory / "network_operators.txt").write_text(
        events.table(sort_by="self_device_time_total", row_limit=60))
    rows = [{"operator": e.key, "calls": e.count,
             "self_gpu_us": e.self_device_time_total,
             "self_cpu_us": e.self_cpu_time_total,
             "shapes": str(e.input_shapes)} for e in events]
    rows.sort(key=lambda r: r["self_gpu_us"], reverse=True)
    (directory / "network_operators.json").write_text(json.dumps(rows, indent=2))
    print("Operator profile:", directory / "network_operators.txt", flush=True)
