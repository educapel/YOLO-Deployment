# YOLO11 Deployment with TensorRT

A complete end-to-end pipeline for deploying YOLO11 models in production using ONNX and NVIDIA TensorRT for optimized inference performance.

## 🎯 Overview

This project provides a streamlined workflow to:
- Export YOLO11 models to ONNX format
- Compile ONNX models to TensorRT engines using Docker
- Prepare models for deployment with NVIDIA Triton Inference Server

YOLO11 is the latest model from Ultralytics, achieving ~2% higher mAP while reducing model size by up to 22% compared to YOLOv10.

## ✨ Features

- **ONNX Export**: Convert YOLO11 PyTorch models to optimized ONNX format
- **TensorRT Compilation**: Compile ONNX models to high-performance TensorRT engines
- **Dynamic Batch Sizing**: Flexible batch size configuration (1-8 images)
- **FP16 Precision**: Reduced memory footprint and faster inference
- **Configuration-Driven**: Manage all parameters through YAML config files
- **Docker Integration**: Automated TensorRT compilation using containerized workflow

## 📋 Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
- Docker with GPU access configured
- NVIDIA Container Toolkit

### Python Dependencies

```bash
pip install ultralytics omegaconf docker
```

## 🚀 Quick Start

### 1. Download Pre-trained Model

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
```

### 2. Configure Export Parameters

Create an `export.yaml` file:

```yaml
onnx:
  weights_path: yolo11m.pt
  is_half: false                # export the model in FP16 format
  is_dynamic: true              # the .onnx model has dynamic batch size
  is_simplified: true           # will optimize layers to reduce size
  with_nms: true                # add NonMaximumSuppression head
  image_size: 640               # the input layer shape (Nx3x640x640)
  device: cuda:0                # device used for export

tensorrt:
  device: 0
  minShapes: images:1x3x640x640   # minimum batch size
  optShapes: images:4x3x640x640   # optimal batch size
  maxShapes: images:8x3x640x640   # maximum batch size
  dtype: fp16
  image: nvcr.io/nvidia/tensorrt:22.08-py3
```

### 3. Export to ONNX

Run the ONNX export script:

```python
python ONNX_Exporter.py
```

This will:
- Load the YOLO11 model
- Export to ONNX format with specified configurations
- Save the `.onnx` file to disk
- Update the config file with the ONNX path

### 4. Compile to TensorRT

Run the TensorRT compilation script:

```python
python TensorRT_Compiler.py
```

This will:
- Start a TensorRT Docker container
- Compile the ONNX model to a TensorRT engine (`.plan` file)
- Generate performance profiling reports
- Save the optimized engine to disk

## 📁 Project Structure

```
YOLO-Deployment/
├── export.yaml              # Configuration file
├── ONNX_Exporter.py        # ONNX export script
├── TensorRT_Compiler.py    # TensorRT compilation script
├── yolo11m.pt              # Pre-trained YOLO11 model
├── yolo11m.onnx            # Exported ONNX model
└── model.plan              # Compiled TensorRT engine
```

## 🔧 Configuration Guide

### ONNX Parameters

- **weights_path**: Path to the YOLO11 `.pt` checkpoint
- **is_half**: Use FP16 precision (reduces model size by ~50%)
- **is_dynamic**: Enable dynamic batch sizing
- **is_simplified**: Apply ONNX graph optimizations
- **with_nms**: Include Non-Maximum Suppression in the model
- **image_size**: Input image dimensions (default: 640x640)
- **device**: CUDA device for export

### TensorRT Parameters

- **device**: GPU device index
- **minShapes/optShapes/maxShapes**: Batch size constraints
  - Format: `images:BxCxHxW` (batch, channels, height, width)
  - Example: `images:4x3x640x640` = batch of 4 images
- **dtype**: Precision mode (`fp16` or `fp32`)
- **image**: TensorRT Docker image version

## 📊 Performance Metrics

After TensorRT compilation, you'll see profiling reports including:

- **Throughput**: Queries per second (QPS)
- **Latency**: Min/max/mean/median inference times
- **H2D Latency**: Host-to-Device data transfer time
- **GPU Compute Time**: Actual GPU processing time
- **D2H Latency**: Device-to-Host data transfer time

Example output:
```
Throughput: 284.831 qps
Latency: mean = 4.77315 ms, median = 4.77563 ms
GPU Compute Time: mean = 3.50529 ms
```

## 🐳 Docker Setup

Ensure Docker has GPU access:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## 🎓 About YOLO11

YOLO11 is a single-stage object detection model that handles identification and classification in one pass. It consists of:

- **Backbone**: Feature extraction using convolutional layers
- **Neck**: Multi-scale feature map fusion
- **Head**: Prediction generation (bounding boxes, class scores, objectness)

### Supported Tasks

- **Detection**: Object detection with bounding boxes
- **Segmentation**: Pixel-accurate object delimitation
- **Pose Estimation**: Keypoint tracking (e.g., human joints)

## 📝 About ONNX & TensorRT

### ONNX (Open Neural Network Exchange)
An open format for representing ML models, enabling cross-framework compatibility (PyTorch, TensorFlow, CoreML, etc.)

### TensorRT
NVIDIA's high-performance deep learning inference optimizer and runtime, providing:
- Layer fusion and optimization
- Kernel auto-tuning for specific GPU architectures
- Reduced precision inference (FP16/INT8)
- Significantly faster inference compared to native frameworks

**Note**: TensorRT engines are GPU-specific and must be recompiled for different GPU architectures.

## 🚨 Troubleshooting

### Docker Connection Error
```
docker.errors.DockerException: Error while fetching server API version
```
**Solution**: Ensure Docker daemon is running:
```bash
sudo systemctl start docker
docker ps  # Verify Docker is accessible
```

### Google Colab Issues
Colab doesn't support Docker natively. Consider:
- Using a local machine with Docker
- Using cloud GPU instances (AWS, GCP, Azure)
- Installing TensorRT directly without Docker (more complex)

### Configuration Missing Error
```
ValueError: ONNX or TensorRT configuration is missing
```
**Solution**: Ensure `export.yaml` contains both `onnx` and `tensorrt` sections

## 📚 Next Steps

This is Part I of a 2-part series. Part II will cover:
- Deploying the TensorRT engine with NVIDIA Triton Inference Server
- Setting up model serving infrastructure
- Performance optimization and benchmarking
- Production deployment best practices

## 📖 References

- [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/)
- [ONNX Official Site](https://onnx.ai/)
- [NVIDIA TensorRT Documentation](https://developer.nvidia.com/tensorrt)
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server)

## 📄 License

This project follows the Ultralytics AGPL-3.0 License for the YOLO models. Check individual component licenses for other dependencies.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

**⚡ Happy Deploying!**
