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

onnx_updated = OmegaConf.merge(
    onnx_cfg,
    DictConfig({"onnx_path": onnx_path}),  
)

# 7. Merge back into full config
merged = OmegaConf.merge(
    export_cfg,  
    DictConfig({"onnx": onnx_updated}),
)

# 8. Save updated config back to file
# This overwrites export.yaml with the onnx_path added
OmegaConf.save(merged, export_config_path)

#####
#
#####