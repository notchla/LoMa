"""Match an image pair with TensorRT engines and compare them against PyTorch.

uv sync --extra export
uv run export_onnx.py matcher:loma-b --precision fp16
uv run export_trt.py exports/loma_B_detect_describe_784x784_fp16.onnx
uv run export_trt.py exports/loma_B_matcher_2048_fp16.onnx
uv run demo_trt.py matcher:loma-b --precision fp16
"""

import time
from functools import partial
from pathlib import Path
from typing import Literal

import cv2
import tensorrt as trt
import torch
import torch.nn.functional as F
import tyro

from export_onnx import DetectAndDescribe, Matcher, load_model
from loma import LoMa, LoMaB
from loma.cfg import LoMaConfig
from loma.loma import filter_matches, to_pixel_coords

# TensorRT expects a single logger per process.
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTEngine:
    """Runs a TensorRT engine with torch CUDA tensors as input and output buffers."""

    def __init__(self, path: Path) -> None:
        self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(path.read_bytes())
        self.context = self.engine.create_execution_context()
        self.names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.stream = torch.cuda.Stream()

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        # Engines from export_trt.py have static shapes and float32 inputs and outputs.
        inputs = {name: x.float().contiguous() for name, x in inputs.items()}
        self.stream.wait_stream(torch.cuda.current_stream())
        outputs = {}
        with torch.cuda.stream(self.stream):
            for name in self.names:
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    tensor = inputs[name]
                else:
                    shape = tuple(self.engine.get_tensor_shape(name))
                    tensor = outputs[name] = torch.empty(shape, device="cuda")
                self.context.set_tensor_address(name, tensor.data_ptr())
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return outputs


def match_trt(
    detect_describe: TRTEngine,
    matcher: TRTEngine,
    image_A: torch.Tensor,
    image_B: torch.Tensor,
    filter_threshold: float,
):
    """TensorRT version of LoMa.match for (1, 3, H, W) images of the engine's size."""
    A = detect_describe(image=image_A)
    B = detect_describe(image=image_B)
    scores = matcher(
        kpts0=A["keypoints"],
        kpts1=B["keypoints"],
        desc0=A["descriptions"],
        desc1=B["descriptions"],
    )["scores"]
    m0, *_ = filter_matches(scores, filter_threshold)
    valid = m0[0] > -1
    matched_A = A["keypoints"][0][valid]
    matched_B = B["keypoints"][0][m0[0][valid]]
    h, w = image_A.shape[-2:]
    return (
        to_pixel_coords(matched_A, h, w).cpu().numpy(),
        to_pixel_coords(matched_B, h, w).cpu().numpy(),
    )


def num_inliers(kpts_A, kpts_B) -> int:
    if len(kpts_A) < 8:
        return 0
    _, mask = cv2.findFundamentalMat(
        kpts_A,
        kpts_B,
        ransacReprojThreshold=0.5,
        method=cv2.USAC_MAGSAC,
        confidence=0.999999,
        maxIters=10000,
    )
    return 0 if mask is None else int(mask.sum())


def latency_ms(fn, iters: int = 10) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def main(
    matcher: LoMaConfig = LoMaB(),
    precision: Literal["fp16", "fp32"] = "fp16",
    size: tuple[int, int] = (784, 784),
    im_A: str = "assets/toronto_A.jpg",
    im_B: str = "assets/toronto_B.jpg",
    engine_dir: Path = Path("exports"),
):
    """
    Args:
        size: (H, W) the engines were exported with.
    """
    H, W = size
    name = getattr(matcher, "name", "loma_custom")

    # Reference: the unmodified PyTorch model. It runs first because load_model below
    # overrides loma's global autocast dtype.
    reference = LoMa(matcher)
    images = [reference._descriptor.read_image(p, H=H, W=W) for p in (im_A, im_B)]
    ref_A, ref_B = reference.match(*images)
    ref_ms = latency_ms(partial(reference.match, *images))
    del reference
    torch.cuda.empty_cache()

    detect_describe = TRTEngine(
        engine_dir / f"{name}_detect_describe_{H}x{W}_{precision}.engine"
    )
    matcher_engine = TRTEngine(
        engine_dir / f"{name}_matcher_{matcher.num_keypoints}_{precision}.engine"
    )
    run_trt = partial(
        match_trt, detect_describe, matcher_engine, *images, matcher.filter_threshold
    )
    trt_A, trt_B = run_trt()
    trt_ms = latency_ms(run_trt)
    print(
        f"PyTorch:  {len(ref_A)} matches, {num_inliers(ref_A, ref_B)} inliers, {ref_ms:.0f} ms"
    )
    print(
        f"TensorRT: {len(trt_A)} matches, {num_inliers(trt_A, trt_B)} inliers, {trt_ms:.0f} ms ({precision})"
    )

    # Parity with the PyTorch graphs that were exported, at the same precision.
    if precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False  # export_trt.py builds without TF32
    model = load_model(matcher, precision)
    pt_detect_describe = DetectAndDescribe(model).eval()
    pt_matcher = Matcher(model).eval()
    features = []
    for path, image in zip((im_A, im_B), images):
        with torch.no_grad():
            kpts, desc = pt_detect_describe(image)
        features.append((kpts, desc))
        out = detect_describe(image=image)
        dist, idx = torch.cdist(
            to_pixel_coords(out["keypoints"][0], H, W),
            to_pixel_coords(kpts[0], H, W),
        ).min(dim=1)
        same = dist < 0.05
        cos = F.cosine_similarity(
            out["descriptions"][0][same], desc[0][idx[same]], dim=-1
        )
        print(
            f"{Path(path).name}: keypoints within 0.05 px {same.float().mean():.1%}, "
            f"within 1 px {(dist < 1).float().mean():.1%}; "
            f"descriptor cosine similarity min {cos.min():.5f}"
        )
    (kpts0, desc0), (kpts1, desc1) = features
    with torch.no_grad():
        scores_pt = pt_matcher(kpts0, kpts1, desc0, desc1)
    scores_trt = matcher_engine(kpts0=kpts0, kpts1=kpts1, desc0=desc0, desc1=desc1)[
        "scores"
    ]
    diff = (scores_trt - scores_pt).abs()
    print(
        f"scores on identical inputs: max abs diff {diff.max():.2e}, mean {diff.mean():.2e}"
    )


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.CascadeSubcommandArgs,))
