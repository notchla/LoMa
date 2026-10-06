<div align="center">
<h1>LoMa: Local Feature Matching Revisited</h1>


<a href="https://arxiv.org/abs/2604.04931"><img src="https://img.shields.io/badge/arXiv-2604.04931-b31b1b" alt="arXiv"></a>
<a href="https://www.davnords.com/loma"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://github.com/davnords/HardMatch"><img src="https://img.shields.io/badge/HardMatch-Dataset-181717?logo=github" alt="HardMatch Dataset"></a>

**Chalmers University of Technology**; **Linköping University**; **University of Amsterdam**; **Lund University**

[David Nordström*](https://scholar.google.com/citations?user=-vJPE04AAAAJ), [Johan Edstedt*](https://scholar.google.com/citations?user=Ul-vMR0AAAAJ&hl), [Georg Bökman](https://scholar.google.com/citations?user=FUE3Wd0AAAAJ), [Jonathan Astermark](https://scholar.google.com/citations?user=dsEPAvUAAAAJ), [Anders Heyden](https://scholar.google.com/citations?user=9j-6i_oAAAAJ), [Viktor Larsson](https://scholar.google.com/citations?user=vHeD0TYAAAAJ), [Mårten Wadenbäck](https://scholar.google.com/citations?user=6WRQpCQAAAAJ), [Michael Felsberg](https://scholar.google.com/citations?user=lkWfR08AAAAJ), [Fredrik Kahl](https://scholar.google.com/citations?user=P_w6UgMAAAAJ)
</div>

<p align="center">
    <img src="assets/loma.jpg" alt="example" width=45%>
    <br>
    <em>Performance on a difficult matching pair compared to LightGlue.</em>
</p>

## Overview
LoMa is a fast and accurate family of local feature matchers. It works similar to [LightGlue](https://github.com/cvg/LightGlue) but significantly improves matching robustness and accuracy across benchmarks, even outperforming [RoMa](https://github.com/Parskatt/RoMa) and [RoMa v2](https://github.com/Parskatt/RoMaV2) on the difficult [WxBS](https://arxiv.org/abs/1504.06603) benchmark. As LoMa leverages local keypoint descriptions, the models are perfect drop-in replacement in e.g. SfM and Visual Localization pipelines.

## Updates
- [June 27, 2026] An initial public release of HardMatch can be found [here](https://github.com/davnords/HardMatch).
- [June 18, 2026] LoMa has been accepted to ECCV 2026 in Malmö as an Oral paper (1.6%).
- [April 14, 2026] Rotation invariant LoMa released. The model, which we call LoMa-R, is great at aerial imagery (e.g. [SatAst](https://github.com/georg-bn/satast)). See the paper [Who Handles Orientation?](https://arxiv.org/abs/2604.11809) (CVPRW26) for more information.
- [April 13, 2026] Integration available with [HLoc](https://github.com/davnords/Hierarchical-Localization) and [vismatch](https://github.com/gmberton/vismatch/pull/63).
- [April 6, 2026] LoMa inference code released. 

## How to Use
```python
import cv2
from loma import LoMa, LoMaB

# load pretrained model
model = LoMa(LoMaB())  # also available: LoMaB128, LoMaL, LoMaG, LoMaR
# Define image paths, e.g.
img_A_path, img_B_path = "assets/0015_A.jpg", "assets/0015_B.jpg"
# Extract matching keypoints in image coordinates
kptsA, kptsB = model.match(img_A_path, img_B_path)

# Find a fundamental matrix (or anything else of interest)
F, mask = cv2.findFundamentalMat(
    kptsA, kptsB, ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
)
```
We provide additional code examples in [demo.py](demo.py), which might help in understanding. To run the demo, use the following API:
```bash
uv run demo.py matcher:loma-b
```

## Setup/Install
In your python environment (tested on Linux python 3.12), run:
```bash
uv pip install -e .
```
or 
```bash
uv sync
```

## Benchmarks
We initially provide code for evaluating on MegaDepth, ScanNet, WxBS and RUBIK. If you do not already have MegaDepth1500 and ScanNet1500, you may run the following to download them:
```bash
source scripts/eval_prep.sh
```
To run a benchmark you need to install the optional dependencies by e.g. `uv sync --extra eval`. Thereafter, you can use the following call signature:
```bash
uv run eval.py matcher:loma-b --benchmark wxbs
```
Use `uv run eval.py --help` to explore the different options. 

### Expected Results
The results are similar to those reported in the paper. For example, running the evaluation for LoMa-B on WxBS gives us `mAA_10px: 0.6876`.

## TensorRT
We provide scripts to export `detect_and_describe` and the matcher scores to ONNX and build TensorRT engines from them. Install the optional dependencies with `uv sync --extra export`. Engines only run on the TensorRT version that built them, so change the `tensorrt-cu13` pin in `pyproject.toml` to match your TensorRT installation if you plan to use its runtime or `trtexec`.
```bash
uv run export_onnx.py matcher:loma-b --precision fp16 --size 784 784
uv run export_trt.py exports/loma_B_detect_describe_784x784_fp16.onnx
uv run export_trt.py exports/loma_B_matcher_2048_fp16.onnx
uv run demo_trt.py matcher:loma-b --precision fp16
```
[demo_trt.py](demo_trt.py) shows how to match an image pair with the engines and compares the results with PyTorch. Things to keep in mind:
- The image size is static, so you need one engine per image size, and images must be resized to it. For DINOv2-based models (all but LoMa-B128), height and width must be multiples of 14.
- Both engines can take a dynamic batch (of images or image pairs): export them with `--max-batch 16` and `export_trt.py` builds an optimization profile from its `--batch` (min, opt, max), (1, 8, 16) by default.
- TensorRT 11 builds strongly typed networks, so the precision (`fp16` or `fp32`) is chosen at ONNX export time.
- TensorRT's TopK limits `num_keypoints` to at most 3840.
- All engine inputs and outputs are FP32, also for `--precision fp16`, which only sets the precision inside the graph.
- Match filtering (`filter_matches`) and the conversion to pixel coordinates (`to_pixel_coords`) run in Python, not in the engines.

| Engine | Tensor | | Shape | Format |
| --- | --- | --- | --- | --- |
| detect/describe | `image` | in | (B, 3, H, W) | RGB in [0, 1]. The ImageNet normalization is part of the engine. |
| | `keypoints` | out | (B, N, 2) | `grid_sample` coordinates in [-1, 1], not pixels |
| | `descriptions` | out | (B, N, D) | |
| matcher | `kpts0`, `kpts1` | in | (B, N, 2) | as returned by the detect/describe engine |
| | `desc0`, `desc1` | in | (B, N, D) | |
| | `scores` | out | (B, N, N) | dual-softmax match confidences in [0, 1], row *i* for keypoint *i* of image A |

N is `num_keypoints` (2048 by default) and D is `input_dim` of the model: 256, or 128 for LoMa-B128. B is 1, or up to `--max-batch` for dynamic-batch engines.

We tested this with LoMa-B, TensorRT 11.2 and torch 2.11 on an RTX 3080 Laptop GPU. On the Toronto pair at 784×784, the FP16 engines find the same 267 matches as PyTorch, in about 305 ms instead of 512 ms.

[profile_trt.py](profile_trt.py) times the static matcher engine, called once per pair, against a dynamic-batch one (`--dynamic-engine`), and checks that their scores agree. With LoMa-B at FP16 on an RTX 5090, the static engine takes 2.6 ms per pair. The dynamic engine returns bit-identical scores. It is 1.4× slower for a single pair (3.6 ms) and about 1.5× faster from a batch of 8 (1.7 ms per pair):

| B | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| static, ms per pair | 2.6 | 2.6 | 2.6 | 2.6 | 2.6 |
| dynamic, ms per pair | 3.6 | 2.4 | 2.0 | 1.7 | 1.7 |

[profile_detect_describe_trt.py](profile_detect_describe_trt.py) does the same for the detect/describe engine, on the Toronto images. Batching does not pay off there: at 784×784 the dynamic engine is slower than calling the static one (23.1 ms per image) at every batch size. Its keypoints agree with the static engine's within 1 px for 99.8% (as closely as the static engine agrees with PyTorch), with descriptor cosine similarity above 0.999:

| B | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| static, ms per image | 23.0 | 23.1 | 23.0 | 23.1 | 23.1 |
| dynamic, ms per image | 30.1 | 27.2 | 26.1 | 25.7 | 26.3 |

## Sizes
We an array of models: LoMA-{B, B128, L, G, R}. For most usecases LoMa-B, which is the same size as LightGlue, works fine. LoMa-G is significantly heavier but gives the most accurate matches, even surpassing the RoMa-family on e.g. WxBS and IMC22. LoMa-R provides a rotation invariant matcher and descriptor (through data augmentation).

## Pretrained Weights
The models should auto download as you initialize them. However, for those who prefer to directly download the weights we provide the links below.
- **LoMa-B** – [Download](https://github.com/davnords/storage/releases/download/loma/loma_B.pt)
- **LoMa-B128** – [Download](https://github.com/davnords/storage/releases/download/loma/loma_B128.pth)
- **LoMa-L** – [Download](https://github.com/davnords/storage/releases/download/loma/loma_L.pth)
- **LoMa-G** – [Download](https://github.com/davnords/storage/releases/download/loma/loma_G.pth)
- **LoMa-R** – [Download](https://github.com/davnords/storage/releases/download/loma/loma_R.pth)

## Checklist
- [x] Publish the inference code.
- [x] Release rotation invariant matcher.
- [x] Integrate with [HLoc](https://github.com/cvg/Hierarchical-Localization?tab=readme-ov-file). See this [fork](https://github.com/davnords/Hierarchical-Localization).
- [x] Integrate with [vismatch](https://github.com/gmberton/vismatch). See this [PR](https://github.com/gmberton/vismatch/pull/63).
- [x] Provide training code.
- [x] Release HardMatch.
- [ ] Merge training code into main branch.
- [ ] Release a lightweight descriptor.

## License
All our code except the matcher, which inherits its license from LightGlue, is MIT license. LightGlue has an [Apache-2.0](https://github.com/cvg/LightGlue/blob/main/LICENSE) license.

## Acknowledgement
Thanks to [Parskatt](https://github.com/Parskatt) for writing most of the code. Our codebase structure is mainly based on [RoMaV2](https://github.com/Parskatt/RoMaV2) and our architectures build on [LightGlue](https://github.com/cvg/lightglue), [DeDoDe](https://github.com/Parskatt/DeDoDe), and [DaD](https://github.com/Parskatt/dad). 

## BibTeX
If you find our models useful, please consider citing our papers!
```bibtex
@inproceedings{nordstrom2026loma,
      title={LoMa: Local Feature Matching Revisited}, 
      author={David Nordström and Johan Edstedt and Georg Bökman and Jonathan Astermark and Anders Heyden and Viktor Larsson and Mårten Wadenbäck and Michael Felsberg and Fredrik Kahl},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
      year={2026}
}

@inproceedings{nordstrom2026who,
  title={Who Handles Orientation? Investigating Invariance in Feature Matching},
  author={David Nordström and Johan Edstedt and Georg Bökman and Fredrik Kahl},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2026}
}
```
