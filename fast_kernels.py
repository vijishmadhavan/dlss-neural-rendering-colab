"""Optional fused CUDA kernels for the pinned MLX-DLSS recovered graph.

No model weights or NVIDIA binaries are included. Each kernel must pass the
upstream operation comparison on the actual GPU before it is enabled.
"""
from __future__ import annotations

import importlib.util
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _e4(x):
    # Software E4M3 rounding also works on GPUs without FP8 Tensor Cores.
    x = x.to(tl.float32)
    mag = tl.minimum(tl.abs(x), 448.0)
    normal = tl.maximum(mag, 0.015625)
    exponent = (normal.to(tl.int32, bitcast=True) >> 23) & 255
    step = ((exponent - 3) << 23).to(tl.float32, bitcast=True)
    step = tl.where(mag < 0.015625, 0.001953125, step)
    scaled = mag / step
    floor = tl.floor(scaled)
    fraction = scaled - floor
    odd = (floor.to(tl.int32) & 1) != 0
    rounded = floor + ((fraction > 0.5) | ((fraction == 0.5) & odd)).to(tl.float32)
    out = rounded * step
    return tl.where(x < 0, -out, out)


@triton.jit
def _round_kernel(X, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    tl.store(Y + i, _e4(x), i < N)


def round_e4(value):
    x = value.contiguous()
    out = torch.empty_like(x)
    _round_kernel[(triton.cdiv(x.numel(), 1024),)](
        x, out, x.numel(), 1024, enable_fp_fusion=False)
    return out


@triton.jit
def _gate_kernel(X, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0).to(tl.float16)
    clamped = tl.minimum(tl.maximum(x.to(tl.float32), -4.0), 4.0)
    linear = (tl.abs(clamped) * -0.055908203125 + 0.447265625).to(tl.float16)
    gate = (clamped * linear.to(tl.float32) + 0.89453125).to(tl.float16)
    result = (x.to(tl.float32) * gate.to(tl.float32)).to(tl.float16)
    tl.store(Y + i, result, i < N)


def gate_activation(value):
    x = value.contiguous()
    out = torch.empty_like(x)
    _gate_kernel[(triton.cdiv(x.numel(), 1024),)](
        x, out, x.numel(), 1024, enable_fp_fusion=False)
    return out


@triton.jit
def _softmax_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    # Every logical pair participates in the original packed half-bit affine
    # transform, including the carry from the low to the high halfword.
    even = c & ~1
    a = tl.load(X + row * N + even, even < N, other=0).to(tl.float16).to(tl.float32)
    b = tl.load(X + row * N + even + 1, even + 1 < N, other=0).to(tl.float16).to(tl.float32)
    a = (a * 0.044921875 + 1.30078125).to(tl.float16)
    b = (b * 0.044921875 + 1.30078125).to(tl.float16)
    a = tl.minimum(tl.maximum(a, 1.03125), 1.5693359375).to(tl.float16)
    b = tl.minimum(tl.maximum(b, 1.03125), 1.5693359375).to(tl.float16)
    packed = a.to(tl.uint16, bitcast=True).to(tl.uint32) | (b.to(tl.uint16, bitcast=True).to(tl.uint32) << 16)
    transformed = (packed << 5) + 0x7FF88000
    bits = tl.where((c & 1) == 0, transformed & 65535, (transformed >> 16) & 65535)
    weight = bits.to(tl.uint16).to(tl.float16, bitcast=True)
    total = tl.sum(tl.where(c < N, weight.to(tl.float32), 0), 0).to(tl.float16)
    reciprocal = tl.div_rn(1.0, total.to(tl.float32)).to(tl.float16)
    probability = (weight.to(tl.float32) * reciprocal.to(tl.float32)).to(tl.float16)
    tl.store(Y + row * N + c, _e4(probability), c < N)


def softmax(value):
    x = value.contiguous()
    n = x.shape[-1]
    out = torch.empty_like(x)
    _softmax_kernel[(x.numel() // n,)](
        x, out, n, triton.next_power_of_2(n), num_warps=4,
        enable_fp_fusion=False)
    return out


@triton.jit
def _cosine_kernel(X, SCALE, Y, ROWS: tl.constexpr, HEADS: tl.constexpr,
                   TOKENS: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
                   S2: tl.constexpr, S3: tl.constexpr,
                   HAS_SCALE: tl.constexpr, BLOCK_ROWS: tl.constexpr):
    r = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    c = tl.arange(0, 8)
    h = (r // TOKENS) % HEADS
    base = (r // (HEADS * TOKENS)) * S0 + h * S1 + (r % TOKENS) * S2
    offsets = base[:, None] + c[None, :] * S3
    mask = r[:, None] < ROWS
    a = tl.load(X + offsets, mask, other=0).to(tl.float32)
    b = tl.load(X + offsets + 8 * S3, mask, other=0).to(tl.float32)
    d = tl.load(X + offsets + 16 * S3, mask, other=0).to(tl.float32)
    e = tl.load(X + offsets + 24 * S3, mask, other=0).to(tl.float32)
    first = (b * b + (a * a).to(tl.float16).to(tl.float32)).to(tl.float16)
    second = (e * e + (d * d).to(tl.float16).to(tl.float32)).to(tl.float16)
    partial = (first.to(tl.float32) + second.to(tl.float32)).to(tl.float16)
    idx4 = tl.broadcast_to((c ^ 4)[None, :], (BLOCK_ROWS, 8))
    xor2 = (partial.to(tl.float32) + tl.gather(partial, idx4, 1).to(tl.float32)).to(tl.float16)
    idx2 = tl.broadcast_to((c ^ 2)[None, :], (BLOCK_ROWS, 8))
    xor1 = (xor2.to(tl.float32) + tl.gather(xor2, idx2, 1).to(tl.float32)).to(tl.float16)
    n0 = tl.sum(tl.where(c[None, :] == 0, xor1.to(tl.float32), 0), 1)
    n1 = tl.sum(tl.where(c[None, :] == 1, xor1.to(tl.float32), 0), 1)
    norm = (n0 + n1).to(tl.float16).to(tl.float32)
    reciprocal = tl.rsqrt(tl.maximum(norm, 0.00006198883056640625)).to(tl.float16).to(tl.float32)
    na = (a * reciprocal[:, None]).to(tl.float16)
    nb = (b * reciprocal[:, None]).to(tl.float16)
    nd = (d * reciprocal[:, None]).to(tl.float16)
    ne = (e * reciprocal[:, None]).to(tl.float16)
    if HAS_SCALE:
        scale = tl.load(SCALE + h, r < ROWS, other=1).to(tl.float16).to(tl.float32)
        na = (na.to(tl.float32) * scale[:, None]).to(tl.float16)
        nb = (nb.to(tl.float32) * scale[:, None]).to(tl.float16)
        nd = (nd.to(tl.float32) * scale[:, None]).to(tl.float16)
        ne = (ne.to(tl.float32) * scale[:, None]).to(tl.float16)
    dest = r[:, None] * 32 + c[None, :]
    tl.store(Y + dest, _e4(na), mask)
    tl.store(Y + dest + 8, _e4(nb), mask)
    tl.store(Y + dest + 16, _e4(nd), mask)
    tl.store(Y + dest + 24, _e4(ne), mask)


def cosine_publish(value, scale=None):
    out = torch.empty(value.shape, device=value.device, dtype=value.dtype)
    rows = value.numel() // 32
    _cosine_kernel[(triton.cdiv(rows, 8),)](
        value, value if scale is None else scale, out,
        rows, value.shape[1], value.shape[2], *value.stride(),
        scale is not None, 8, num_warps=4, enable_fp_fusion=False)
    return out


def validate_kernels(device):
    """Enable only kernels whose test outputs exactly equal the pinned reference.

    A failed kernel is reported and left on the reference implementation.
    A separate real-frame temporal comparison is required by the notebook.
    """
    from mlxdlss import model as reference
    gen = torch.Generator(device="cpu").manual_seed(816)
    report = {}

    def check(name, cases):
        try:
            for expected, actual in cases():
                torch.cuda.synchronize(device)
                if not torch.equal(expected, actual):
                    diff = (expected.float() - actual.float()).abs()
                    raise RuntimeError(f"not exact: max={diff.max().item():g}, mean={diff.mean().item():g}")
            report[name] = {"enabled": True, "result": "exact on test inputs"}
        except Exception as exc:
            report[name] = {"enabled": False, "result": str(exc)}
        print(f"Kernel {name}: {report[name]}", flush=True)

    def round_cases():
        # Every finite float16 encoding, including all half-way boundaries.
        all_half = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
        x = all_half[torch.isfinite(all_half)].to(device)
        yield reference.e4m3_round_trip(x), round_e4(x)

    def softmax_cases():
        for n in (64, 96, 256, 512):
            x = (torch.randn((257, n), generator=gen) * 12).half().to(device)
            x[0] = 0
            yield reference.vendor_approximate_softmax(x), softmax(x)

    def gate_cases():
        all_half = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
        x = all_half[torch.isfinite(all_half)].to(device)
        yield reference.quadratic_gate_activation(x), gate_activation(x)

    def cosine_cases():
        for heads in (1, 2, 16, 32):
            for amplitude in (0.0001, 0.1, 1.0, 4.0):
                x = (torch.randn((2, 65, heads, 32), generator=gen) * amplitude).half().to(device).permute(0, 2, 1, 3)
                x[0, :, 0] = 0
                for scale in (None, torch.linspace(0.5, 8, heads, device=device).half()):
                    yield reference.vendor_cosine_publish(x, scale), cosine_publish(x, scale)

    with torch.inference_mode(), torch.cuda.device(device):
        check("round_e4", round_cases)
        check("gate_activation", gate_cases)
        check("softmax", softmax_cases)
        check("cosine_publish", cosine_cases)
    if not any(entry["enabled"] for entry in report.values()):
        raise RuntimeError(f"No fused kernel passed validation on this host: {report}")
    return report


def build_fast_model(weights, device, validation):
    """Private copy of the pinned graph; upstream module stays available for A/B."""
    from mlxdlss import model as reference
    name = "_portable_nr_fast_reference"
    spec = importlib.util.spec_from_file_location(name, reference.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    # Batch independent output-head/branch GEMMs, but preserve the original
    # input-head and branch accumulation order (including half rounding).
    # No concatenated large GEMM: that would change the reduction semantics.
    def batched_branched(value, *, expansion_weight, branch_projection_weight,
                         output_projection_weight):
        groups = value.shape[-1] // 32
        lead = value.shape[:-1]
        inputs = value.reshape(-1, groups, 32)
        expanded = 0
        for index in range(groups):
            expanded = expanded + torch.matmul(inputs[:, index, :], expansion_weight[:, :, index])
        activated = module.e4m3_round_trip(module.quadratic_gate_activation(expanded))
        branches = torch.matmul(activated, branch_projection_weight)
        summed = branches[:, 0] + branches[:, 1]
        summed = summed + branches[:, 2]
        summed = summed + branches[:, 3]
        heads = module.e4m3_round_trip(summed)
        merged = heads.permute(1, 0, 2).reshape(*lead, groups * 32)
        return merged @ output_projection_weight

    module.branched_feed_forward = batched_branched
    def batched_split(value, *, first_projection_weight, expand_weight, project_weight):
        hidden = module.e4m3_round_trip(value @ first_projection_weight)
        groups = hidden.shape[-1] // 64
        grouped = hidden.reshape(-1, groups, 64).permute(1, 0, 2)
        expanded = torch.bmm(grouped, expand_weight)
        projected = torch.bmm(module.quadratic_gate_activation(expanded), project_weight)
        merged = projected.permute(1, 0, 2).reshape(*hidden.shape)
        return module.e4m3_round_trip(merged)

    module.split_group_feed_forward = batched_split
    if validation["round_e4"]["enabled"]:
        original_round = module.e4m3_round_trip
        module.e4m3_round_trip = lambda x: round_e4(x) if x.is_cuda and x.dtype == torch.float16 else original_round(x)
    if validation["gate_activation"]["enabled"]:
        original_gate = module.quadratic_gate_activation
        module.quadratic_gate_activation = lambda x: gate_activation(x) if x.is_cuda and x.dtype == torch.float16 else original_gate(x)
    if validation["softmax"]["enabled"]:
        original_softmax = module.vendor_approximate_softmax
        module.vendor_approximate_softmax = lambda x: softmax(x) if x.is_cuda and x.dtype == torch.float16 and x.shape[-1] % 2 == 0 and x.shape[-1] <= 4096 else original_softmax(x)
    if validation["cosine_publish"]["enabled"]:
        original_cosine = module.vendor_cosine_publish
        module.vendor_cosine_publish = lambda x, scale=None: cosine_publish(x, scale) if x.is_cuda and x.dtype == torch.float16 and x.ndim == 4 and x.shape[-1] == 32 else original_cosine(x, scale)
    model = module.NeuralRenderingModel(weights).half().to(device).eval()
    # Bias layout is constant across every frame. Resolve it once, on-device.
    with torch.no_grad():
        for key in model._weight_attributes:
            if key.endswith(".attn_bias"):
                bias = model.weight(key)
                if bias.shape[0] in (1, 16):
                    bias.copy_(reference.recover_attention_bias_layout(bias))
    module.uses_fragment_swizzle = lambda index, heads: False
    return model
