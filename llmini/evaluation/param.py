from transformers import AutoModelForCausalLM
from torch import count_nonzero
import argparse

# parse arguments: model_id, model_dtype
parser = argparse.ArgumentParser()
parser.add_argument("model_id", help="Model ID or path")
parser.add_argument("--dtype", help="Model dtype", default="auto", choices=["float32", "float16"])
args = parser.parse_args()

# load model
model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=args.dtype)

# calculate: parameters, size in gb
total_params = nonzero_params = total_bytes = nonzero_bytes = 0
for p in model.parameters():
    numel = p.numel()
    nonzero = count_nonzero(p).item()
    ebytes = p.element_size()

    total_params  += numel
    nonzero_params += nonzero
    total_bytes   += numel * ebytes
    nonzero_bytes += nonzero * ebytes

total_size_gb = total_bytes / (1024 ** 3)
nonzero_size_gb = nonzero_bytes / (1024 ** 3)

# print values
print(f"Model name:           {args.model_id}")
print(f"Model dtype:          {args.dtype}")
print(f"Model total size:     {total_size_gb:.2f} GB")
print(f"Model non-zero size:  {nonzero_size_gb:.2f} GB\n")

print(f"Total parameters:     {total_params:,}")
print(f"Non-zero parameters:  {nonzero_params:,}")
print(f"Sparsity:             {1 - (nonzero_params / total_params):.2%}")

