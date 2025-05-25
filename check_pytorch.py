# check_pytorch.py
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version (PyTorch built with): {torch.version.cuda}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA Version (PyTorch built with): N/A")
    print("GPU Name: N/A")

print("--- Check complete ---")