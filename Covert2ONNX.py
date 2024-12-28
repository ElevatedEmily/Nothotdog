import torch
from torchvision import models
import torch.nn as nn
# Load the trained model
model_path = 'resnet_model_weights.pth'
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, 1),
    nn.Sigmoid()
)
model.load_state_dict(torch.load(model_path))
model.eval()

# Dummy input for exporting the model
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
onnx_path = "resnet_model.onnx"
torch.onnx.export(
    model, dummy_input, onnx_path,
    export_params=True,
    opset_version=11,
    input_names=['input'], output_names=['output']
)
print(f"Model exported to {onnx_path}")
