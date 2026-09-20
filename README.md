# DLSS Neural Rendering for Colab — Unofficial CUDA Workflow

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vijishmadhavan/dlss-neural-rendering-colab/blob/master/portable_neural_rendering_colab.ipynb)

An experimental image and video notebook built on [MLX-DLSS](https://github.com/iamwavecut/MLX-DLSS), with CUDA/Triton optimizations for its PyTorch pipeline. This repository packages an accessible Colab workflow and optimization layer; it does not claim to have discovered or independently recovered the underlying model.

**Tested on NVIDIA A100. Other GPUs are unverified. Not real-time, not an NVIDIA product, and not a native NGX/DLSS integration.**

## Start

Open [the notebook](portable_neural_rendering_colab.ipynb) in Google Colab and select an NVIDIA GPU runtime. Run its cells in order:

1. Set the input path and controls.
2. Install the pinned upstream pipeline and load the embedded optimization modules.
3. Supply your own compatible logical weights or `nvngx_dlssnr.dll`; review the applicable model terms before accepting them.
4. Run kernel checks, consecutive-frame comparisons, and the benchmark.
5. Inspect the short preview, especially faces, motion, occlusion, and cuts.
6. Approve and export the full result.

No NVIDIA DLLs or model weights are included or downloaded by this release. A supplied DLL is parsed for model extraction, not executed. Its model resource must match the pinned implementation. Existing logical weights must be compatible with that implementation. The source-code license does not grant rights to NVIDIA models or other people's media.

## What this repository adds

- A self-contained Colab notebook with readable sibling Python modules.
- GPU-resident temporal feature construction, history sampling, and composition.
- Triton arithmetic kernels, batched independent matrix operations, and optional CUDA graph replay.
- Numerical validation and failure diagnostics; failed comparisons block export.
- Preview approval, timing reports, and output frame-count verification.

The model architecture, recovery work, extraction tools, reference arithmetic, and foundational media pipeline come from MLX-DLSS. This release pins upstream commit `0ca2deab092fe6f3e331bf4f616271dbc64521d0`; newer upstream features are not automatically included.

## Measured performance

One maintainer-run Colab video test, September 19, 2026:

| Item | Measurement |
| --- | --- |
| GPU | NVIDIA A100-SXM4-40GB |
| Driver / software | 580.82.07; PyTorch 2.11.0+cu128; CUDA 12.8 |
| Input and output | 2416 × 1800; 240 frames; 30 fps; audio retained |
| Settings | Temporal on; scale 1; natural; intensity 0.8; detail/colour 1 |
| Processing including I/O | 103.07 seconds, 2.33 frames/second |
| Processing plus verification | 109.32 seconds |
| Benchmark/calibration | 91.82 seconds, separate from processing |
| Processing plus calibration | 194.89 seconds; model load and downloads additional |

These are reported measurements, not a hardware-independent guarantee. The test footage is not distributed. A 60-second target was **not met**. There is no matched native-NVIDIA benchmark establishing equivalent performance or output. No speedup ratio is claimed against a different resolution or GPU.

## Resolution and quality

`PROCESSING_SCALE` is **internal supersampling, not output upscaling**. For example, 960 × 720 at scale 2 performs internal work at 1920 × 1440, then exports 960 × 720. This notebook does not implement DLSS Super Resolution or frame generation.

Motion/confidence estimation is limited to a 640-pixel longest-side guide by default. Rendering remains at the requested internal resolution. The smaller motion guide is an intentional approximation and may affect fast movement, occlusions, and scene-cut decisions.

Both benchmark paths use the same bounded motion guide. Their numerical comparison is not a test against native NVIDIA output or the older full-resolution-flow pipeline. A short passing test cannot guarantee temporal quality for an entire video. Floating-point calculations are not claimed bit-exact.

## Compatibility and limitations

- Requires Linux, a supported CUDA GPU/driver combination, CUDA-enabled PyTorch, and compatible Triton. No Wine or Vulkan graphics stack is used.
- A100-40GB/80GB have been used during development; only the configuration above is the published full-resolution benchmark. Other GPUs need testing. This CUDA notebook is not an AMD, Intel, or Apple GPU implementation.
- VRAM use depends on resolution and settings; no universal minimum is established.
- CPU optical flow, input resizing, and video encoding remain part of the pipeline. Non-default detail finishing can add CPU cost.
- GPU-resident optimization targets temporal video. Images and non-temporal processing use the existing path and do not inherit the video benchmark claim.
- Control masks are not supported by the resident temporal path.
- Dependency downloads, compilation, calibration, and model loading can make first use substantially slower.

## Credits and licensing

- **[MLX-DLSS / iamwavecut and contributors](https://github.com/iamwavecut/MLX-DLSS)** — foundational implementation and model recovery; Apache-2.0 source. This optimization layer adapts its arithmetic and processing concepts. See [LICENSE](LICENSE), [NOTICE](NOTICE), and the preserved [upstream notice](UPSTREAM_NOTICE).
- **NVIDIA** — original DLSS neural-rendering technology and model. Proprietary model/binary terms are separate; no affiliation or endorsement is implied.
- **[PyTorch](https://github.com/pytorch/pytorch)** and **[Triton](https://github.com/triton-lang/triton)** — tensor execution and GPU kernels.
- **[OpenCV](https://github.com/opencv/opencv)**, **[NumPy](https://github.com/numpy/numpy)**, **[FFmpeg](https://ffmpeg.org/)**, and **[safetensors](https://github.com/huggingface/safetensors)** — motion/image operations, arrays, media I/O, and tensor storage.

The repository's source is distributed under Apache-2.0. Dependencies retain their respective licenses. No permission to redistribute extracted models, NVIDIA binaries, or third-party footage is conveyed.

## Reporting results

Include GPU, driver, PyTorch/CUDA/Triton versions, input dimensions/frame count, all controls, and the generated benchmark/timing JSON. Report cold-start and warm-run times separately. Remove private paths or media before posting. Please do not attach proprietary weights, DLLs, credentials, or unlicensed footage.
