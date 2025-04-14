import torch
import torch.onnx
import onnx
from models.cryo_unet import CryoResUNet3D
from configs.cfg import cfg

cfg.device = 'cuda'
cfg.feature_maps = [24, 48, 72]
cfg.use_scSE = True

# Load PyTorch Model
weights = f'logs/checkpoints/CRYOET-364/epoch_{49}_fold{0}.pth'
checkpoint = torch.load(weights, map_location='cpu')  # Use CPU to avoid device issues
new_state_dict = {}
for k, v in checkpoint["model"].items():
    new_k = k.replace("_orig_mod.", "")  # Remove the prefix
    new_state_dict[new_k] = v

model = CryoResUNet3D(cfg).to(cfg.device)
model.load_state_dict(new_state_dict, strict=True)
model.eval()

# Dummy Input for ONNX Conversion
dummy_input = torch.randn(16, 1, 72, 72, 72, device=cfg.device)  # Fixed batch size of 16

# Export to ONNX with Explicit Naming and Fixed Batch Size
onnx_file = "model.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_file,
    export_params=True,       # Stores the trained weights
    opset_version=11,         # Good default for TensorRT
    do_constant_folding=True, # Fold constant ops
    input_names=["input"], 
    output_names=["output"]
    # Removed dynamic_axes to enforce fixed batch size
)
print(f"Exported PyTorch model to {onnx_file}")

# (Optional) Validate the ONNX file
onnx_model = onnx.load(onnx_file)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")
