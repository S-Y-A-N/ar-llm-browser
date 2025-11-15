from transformers import pipeline
import torch
import json

pipe = pipeline(
    "text-generation", model="google/gemma-3-1b-it", device=0, dtype=torch.bfloat16
)

messages = [
    [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Write a poem on Hugging Face, the company"},
            ],
        },
    ],
]

output = pipe(messages, max_new_tokens=50)
print(json.dumps(output, indent=2))
print(torch.cuda.is_available())
