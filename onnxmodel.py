import os
import ultralytics as ul
from omegaconf import OmegaConf, DictConfig

######
# .onnx model saved on disk in the same root directory as the current script.
######

export_config_path = "export.yaml"
export_cfg = OmegaConf.load(export_config_path)

onnx_cfg = export_cfg.get("onnx")

model = ul.YOLO(model=onnx_cfg["weights_path"])
onnx_path = model.export(
        format="onnx",
        device=onnx_cfg["device"],
        imgsz=onnx_cfg["image_size"],
        nms=onnx_cfg["with_nms"],
        half=onnx_cfg["is_half"],
        simplify=onnx_cfg["is_simplified"],
        dynamic=onnx_cfg["is_dynamic"]
)

# Update Export Config with ONNX model path
onnx_updated = OmegaConf.merge(
        onnx_cfg,
        DictConfig({"onnx_path": m_path}),
    )
merged = OmegaConf.merge(
        conf,
        DictConfig({"onnx": onnx_updated}),
    )
OmegaConf.save(merged, export_config_path)

#####
#
#####