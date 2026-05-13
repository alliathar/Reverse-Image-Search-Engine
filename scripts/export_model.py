"""One-time export of MobileNetV3-Small as a feature extractor in ONNX format.

Output: models/mobilenetv3_small.onnx
  Input:  float32 tensor [1, 3, 224, 224]  (RGB, ImageNet-normalized)
  Output: float32 tensor [1, 576]          (pooled features, pre-classifier)
"""
import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUT_PATH = os.path.join(OUT_DIR, "mobilenetv3_small.onnx")
os.makedirs(OUT_DIR, exist_ok=True)

model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model.eval()

# Strip classifier: features -> avgpool -> flatten
feature_extractor = nn.Sequential(model.features, model.avgpool, nn.Flatten())

dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = feature_extractor(dummy)
print(f"Feature dim: {out.shape[1]}")

torch.onnx.export(
    feature_extractor,
    dummy,
    OUT_PATH,
    input_names=["input"],
    output_names=["features"],
    dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
    opset_version=13,
    dynamo=False,  # legacy exporter writes a single self-contained .onnx file
)
print(f"Wrote {OUT_PATH} ({os.path.getsize(OUT_PATH) / 1e6:.1f} MB)")
