"""Profile the detect/describe engine: the static batch-1 engine called once per image vs a dynamic-batch engine.

uv run profile_detect_describe_trt.py matcher:loma-b
uv run profile_detect_describe_trt.py matcher:loma-b --dynamic-engine exports/loma_B_detect_describe_784x784_b16_fp16.engine
"""

from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from PIL import Image

from demo_trt import TRTEngine, latency_ms
from loma import LoMaB
from loma.cfg import LoMaConfig
from loma.loma import to_pixel_coords
from profile_trt import run_looped


def read_image(path: str, H: int, W: int) -> torch.Tensor:
    image = np.array(Image.open(path).convert("RGB").resize((W, H))) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).float().cuda()


def main(
    matcher: LoMaConfig = LoMaB(),
    precision: Literal["fp16", "fp32"] = "fp16",
    size: tuple[int, int] = (784, 784),
    images: tuple[str, ...] = ("assets/toronto_A.jpg", "assets/toronto_B.jpg"),
    engine_dir: Path = Path("exports"),
    dynamic_engine: Path | None = None,
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16),
    iters: int = 20,
):
    """
    Args:
        size: (H, W) the engines were exported with.
        images: Cycled to fill each batch. Real images, as top-k on noise is full of near-ties.
        dynamic_engine: Dynamic-batch detect/describe engine. Without it, only the static baseline is profiled.
    """
    H, W = size
    name = getattr(matcher, "name", "loma_custom")
    static = TRTEngine(
        engine_dir / f"{name}_detect_describe_{H}x{W}_{precision}.engine"
    )
    dynamic = None if dynamic_engine is None else TRTEngine(dynamic_engine)
    pool = [read_image(p, H, W) for p in images]

    print(f"{'B':>3} {'static ms':>10} {'/image':>7}", end="")
    if dynamic is not None:
        print(
            f" {'dynamic ms':>11} {'/image':>7} {'speedup':>8} {'kpts <0.05px':>12} {'<1px':>6} {'min cos':>8}",
            end="",
        )
    print()
    for B in batch_sizes:
        inputs = {"image": torch.stack([pool[i % len(pool)] for i in range(B)])}
        static_ms = latency_ms(partial(run_looped, static, inputs), iters)
        print(f"{B:>3} {static_ms:>10.2f} {static_ms / B:>7.2f}", end="")
        if dynamic is not None:
            dynamic_ms = latency_ms(partial(dynamic, **inputs), iters)
            # Keypoints are matched by position: top-k order can differ between engines.
            out_d, out_s = dynamic(**inputs), run_looped(static, inputs)
            dist, idx = torch.cdist(
                to_pixel_coords(out_d["keypoints"], H, W),
                to_pixel_coords(out_s["keypoints"], H, W),
            ).min(dim=-1)
            same = dist < 0.05
            desc_s = out_s["descriptions"].gather(
                1, idx[..., None].expand_as(out_s["descriptions"])
            )
            cos = F.cosine_similarity(out_d["descriptions"], desc_s, dim=-1)[same]
            print(
                f" {dynamic_ms:>11.2f} {dynamic_ms / B:>7.2f} {static_ms / dynamic_ms:>7.2f}x"
                f" {same.float().mean():>12.1%} {(dist < 1).float().mean():>6.1%} {cos.min():>8.5f}",
                end="",
            )
        print()


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.CascadeSubcommandArgs,))
