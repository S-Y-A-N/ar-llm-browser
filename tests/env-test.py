import torch

cuda_available = torch.cuda.is_available()
print(torch.cuda.get_device_name(0) if cuda_available else "No CUDA GPUs are available")
