import os
import argparse
import tensorrt as trt


def build_trt_engine(
    onnx_path,
    engine_path,
    max_workspace_size=1 << 30,  # 1 GB
    fp16=False
):
    """
    Build a TensorRT engine from an ONNX model.
    """

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    # Explicit batch is REQUIRED for ONNX
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()

    # Workspace memory
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        max_workspace_size
    )

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] FP16 enabled")
        else:
            print("[WARNING] FP16 not supported on this platform")

    # Load ONNX model
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("[ERROR] Failed to parse ONNX model")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    print("[INFO] Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"[SUCCESS] TensorRT engine saved to: {engine_path}")
    return engine_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build TensorRT engine from ONNX model"
    )
    parser.add_argument(
        "--onnx",
        type=str,
        required=True,
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="model.trt",
        help="Output TensorRT engine path"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 precision"
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=1024,
        help="Workspace size in MB"
    )

    args = parser.parse_args()

    build_trt_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        max_workspace_size=args.workspace << 20,
        fp16=args.fp16
    )

