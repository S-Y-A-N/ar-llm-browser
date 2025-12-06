import torch

def parse_torch_dtype(dtype_str):
    name = dtype_str.split(".")[-1]
    return getattr(torch, name)