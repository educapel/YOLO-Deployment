import docker
from pathlib import Path
from omegaconf import OmegaConf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
conf = OmegaConf.load("export.yaml")
onnx_cfg = conf.get("onnx")
trt_cfg = conf.get("tensorrt")

# Validate configuration
if not onnx_cfg or not trt_cfg:
    raise ValueError(
        "ONNX or TensorRT configuration is missing. Check your configuration file."
    )

# GPU configuration
gpu_devices = [trt_cfg["device"]]
gpu_config = {
    "device_requests": [{"count": len(gpu_devices), "capabilities": [["gpu"]]}],
    "devices": [f"/dev/nvidia{n}" for n in gpu_devices],
}

# Volume mapping
volume_mapping = {
    f"{Path().resolve()}": {"bind": "/workspace", "mode": "rw"}
}

# Docker client
client = docker.from_env()

container = None

try:
    # Start Docker container
    logger.info("Starting Docker container...")
    container = client.containers.run(
        trt_cfg["image"],
        command='sh -c "while true; do sleep 3600; done"',
        detach=True,
        stdout=True,
        stderr=True,
        remove=True,
        volumes=volume_mapping,
        **gpu_config,
        name="onnx2tensorrt-container"
    )
    logger.info("Docker container started successfully.")

    # Build TensorRT conversion command
    try:
        _exec = "trtexec"
        _onnx_path = Path(onnx_cfg["onnx_path"]).stem

        _o2t = f" --onnx=/workspace/{_onnx_path}.onnx --saveEngine=/workspace/model.plan --{trt_cfg['dtype'].lower()}"

        _shapes = f" --minShapes={trt_cfg['minShapes']} --optShapes={trt_cfg['optShapes']} --maxShapes={trt_cfg['maxShapes']}"

        _force_fp16 = f" --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw"

        command = _exec + _o2t + _shapes + _force_fp16

        logger.info(f"Executing TensorRT conversion command: {command}")

        # Execute conversion command
        exec_result = container.exec_run(command, detach=False)

        if exec_result.exit_code != 0:
            raise RuntimeError(
                f"Error during conversion: {exec_result.output.decode('utf-8')}"
            )
        else:
            logger.info("Conversion successful.")
            logger.info(exec_result.output.decode("utf-8"))

    except Exception as e:
        logger.error(f"Error executing TensorRT conversion command: {e}")
        raise RuntimeError(f"Error executing TensorRT conversion command: {e}")

except docker.errors.DockerException as docker_error:
    logger.error(f"Docker error: {docker_error}")
    raise RuntimeError(f"Error starting Docker container: {docker_error}")

finally:
    # Clean up: stop and remove the container after use
    if container:
        logger.info("Stopping and removing Docker container.")
        container.stop()
        logger.info("Docker container removed.")