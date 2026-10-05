"""Profile the matcher engine: the static batch-1 engine called once per pair vs a dynamic-batch engine.

uv run profile_trt.py matcher:loma-b
uv run profile_trt.py matcher:loma-b --dynamic-engine exports/loma_B_matcher_2048_b16_fp16.engine
"""

from functools import partial
from pathlib import Path
from typing import Literal

import torch
import tyro

from demo_trt import TRTEngine, latency_ms
from loma import LoMaB
from loma.cfg import LoMaConfig


def run_looped(engine: TRTEngine, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    batch = next(iter(inputs.values())).shape[0]
    return torch.cat(
        [
            engine(**{name: x[i : i + 1] for name, x in inputs.items()})["scores"]
            for i in range(batch)
        ]
    )


def main(
    matcher: LoMaConfig = LoMaB(),
    precision: Literal["fp16", "fp32"] = "fp16",
    engine_dir: Path = Path("exports"),
    dynamic_engine: Path | None = None,
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16),
    iters: int = 50,
):
    """
    Args:
        dynamic_engine: Dynamic-batch matcher engine. Without it, only the static baseline is profiled.
    """
    name = getattr(matcher, "name", "loma_custom")
    N, D = matcher.num_keypoints, matcher.input_dim
    static = TRTEngine(engine_dir / f"{name}_matcher_{N}_{precision}.engine")
    dynamic = None if dynamic_engine is None else TRTEngine(dynamic_engine)

    print(f"{'B':>3} {'static ms':>10} {'/pair':>7}", end="")
    if dynamic is not None:
        print(
            f" {'dynamic ms':>11} {'/pair':>7} {'speedup':>8} {'max diff':>9} {'mean diff':>9}",
            end="",
        )
    print()
    for B in batch_sizes:
        inputs = {
            "kpts0": torch.rand(B, N, 2, device="cuda") * 2 - 1,
            "kpts1": torch.rand(B, N, 2, device="cuda") * 2 - 1,
            "desc0": torch.randn(B, N, D, device="cuda"),
            "desc1": torch.randn(B, N, D, device="cuda"),
        }
        static_ms = latency_ms(partial(run_looped, static, inputs), iters)
        print(f"{B:>3} {static_ms:>10.2f} {static_ms / B:>7.2f}", end="")
        if dynamic is not None:
            dynamic_ms = latency_ms(partial(dynamic, **inputs), iters)
            diff = (dynamic(**inputs)["scores"] - run_looped(static, inputs)).abs()
            print(
                f" {dynamic_ms:>11.2f} {dynamic_ms / B:>7.2f} {static_ms / dynamic_ms:>7.2f}x"
                f" {diff.max():>9.2e} {diff.mean():>9.2e}",
                end="",
            )
        print()


if __name__ == "__main__":
    tyro.cli(main, config=(tyro.conf.CascadeSubcommandArgs,))
