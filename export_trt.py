"""Build a TensorRT engine from an ONNX file written by export_onnx.py.

uv run export_trt.py exports/loma_B_detect_describe_784x784_fp16.onnx
"""

from pathlib import Path

import tensorrt as trt
import tyro


def main(onnx_path: tyro.conf.Positional[Path], engine_path: Path | None = None):
    """
    Args:
        engine_path: Defaults to onnx_path with an .engine suffix.
    """
    if engine_path is None:
        engine_path = onnx_path.with_suffix(".engine")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    # TensorRT 11 networks are strongly typed: layer precisions come from the ONNX graph.
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    # parse_from_file also resolves the external .onnx.data weights.
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit(f"Failed to parse {onnx_path}")
    config = builder.create_builder_config()
    # TF32 is on by default; keep FP32 layers in true FP32 (it otherwise reorders the detector's top-k).
    config.clear_flag(trt.BuilderFlag.TF32)
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise SystemExit(f"Failed to build an engine from {onnx_path}")
    engine_path.write_bytes(engine)
    print(f"Saved {engine_path}")


if __name__ == "__main__":
    tyro.cli(main)
