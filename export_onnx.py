"""Export LoMa's detect_and_describe and matcher scores to ONNX, for building TensorRT engines.

    uv sync --extra export
    uv run export_onnx.py matcher:loma-b --precision fp16 --size 784 784

Input shapes are static: build one engine per image size.
"""

import dataclasses
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import tyro
from torch import nn

import loma.loma
from loma import LoMa, LoMaB
from loma.cfg import LoMaConfig
from loma.device import device

DTYPES = {"fp16": torch.float16, "fp32": torch.float32}
# TensorRT's TopK layer does not build for larger k.
TRT_MAX_TOPK = 3840


class Normalize(nn.Module):
    """torchvision's Normalize without its data-dependent `(std == 0).any()` check, which breaks torch.export."""

    def __init__(self, mean: list[float], std: list[float]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, device=device)[:, None, None])
        self.register_buffer("std", torch.tensor(std, device=device)[:, None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class DetectAndDescribe(nn.Module):
    """Tensor branch of LoMa.detect_and_describe, calling forward directly to skip the inference_mode entry points."""

    def __init__(self, model: LoMa) -> None:
        super().__init__()
        self.detector = model._detector
        self.descriptor = model._descriptor
        self.num_keypoints = model.cfg.num_keypoints

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keypoints = self.detector(image, self.num_keypoints)["keypoints"]
        description_grid = self.descriptor(image)
        descriptions = F.grid_sample(
            description_grid.float(),
            keypoints[:, None],
            mode="bilinear",
            align_corners=False,
        )[:, :, 0].mT
        return keypoints, descriptions


class Matcher(nn.Module):
    def __init__(self, model: LoMa) -> None:
        super().__init__()
        self.model = model

    def forward(self, kpts0, kpts1, desc0, desc1) -> torch.Tensor:
        return self.model(kpts0, kpts1, desc0, desc1)["scores"].float()


def load_model(matcher: LoMa.Cfg, precision: Literal["fp16", "fp32"]) -> LoMa:
    dtype = DTYPES[precision]
    amp = precision != "fp32"
    model = LoMa(dataclasses.replace(matcher, mp=amp))
    # TensorRT 11 only builds strongly typed networks, so precision must be fixed in the graph.
    # loma picks its autocast dtype per GPU at import time; override it everywhere it is read.
    loma.loma.amp_dtype = dtype  # read by LoMa.forward
    for m in model.modules():
        if hasattr(m, "amp_dtype"):
            m.amp = amp
            m.amp_dtype = dtype
    if matcher.descriptor == "dedode_g":
        # FrozenDINOv2 runs entirely in amp_dtype and its weights were cast to the import-time dtype
        # before the checkpoint was loaded, so cast and reload them at the export dtype.
        model._descriptor.encoder.frozen_dinov2.dinov2_vitl14.to(dtype)
        if matcher.weights_url is not None:
            weights = torch.hub.load_state_dict_from_url(
                matcher.weights_url, map_location=device
            )
            model.load_state_dict(weights, strict=False)
    normalizer = model._detector.normalizer
    model._detector.normalizer = Normalize(normalizer.mean, normalizer.std)
    return model


def main(
    matcher: LoMaConfig = LoMaB(),
    precision: Literal["fp16", "fp32"] = "fp16",
    size: tuple[int, int] = (784, 784),
    out_dir: Path = Path("exports"),
):
    """
    Args:
        size: (H, W) of the input image. Must be multiples of 14 for DINOv2 descriptors (all but LoMa-B128).
    """
    num_keypoints = matcher.num_keypoints
    assert num_keypoints <= TRT_MAX_TOPK, (
        f"TensorRT TopK supports at most {TRT_MAX_TOPK} keypoints, got {num_keypoints}"
    )
    H, W = size
    name = getattr(matcher, "name", "loma_custom")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(matcher, precision)

    detect_describe_path = out_dir / f"{name}_detect_describe_{H}x{W}_{precision}.onnx"
    matcher_path = out_dir / f"{name}_matcher_{num_keypoints}_{precision}.onnx"
    image = torch.rand(1, 3, H, W, device=device)
    # Separate tensors per image: torch.export merges inputs that are the same tensor object.
    kpts0, kpts1 = (
        torch.rand(1, num_keypoints, 2, device=device) * 2 - 1 for _ in range(2)
    )
    desc0, desc1 = (
        torch.randn(1, num_keypoints, matcher.input_dim, device=device)
        for _ in range(2)
    )
    # optimize=False: onnxscript's Conv+BatchNorm fusion writes FP32 kernels for FP16 convs, which
    # TensorRT rejects. TensorRT does the same fusions itself.
    with torch.no_grad():
        torch.onnx.export(
            DetectAndDescribe(model).eval(),
            (image,),
            detect_describe_path,
            dynamo=True,
            opset_version=18,
            optimize=False,
            input_names=["image"],
            output_names=["keypoints", "descriptions"],
        )
        torch.onnx.export(
            Matcher(model).eval(),
            (kpts0, kpts1, desc0, desc1),
            matcher_path,
            dynamo=True,
            opset_version=18,
            optimize=False,
            input_names=["kpts0", "kpts1", "desc0", "desc1"],
            output_names=["scores"],
        )
    print(f"Saved {detect_describe_path} and {matcher_path}")


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.CascadeSubcommandArgs,))
