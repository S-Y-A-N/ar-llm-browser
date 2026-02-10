# Inference Speed (Throughput [tokens/sec]) and VRAM usage

import torch.cuda
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

prompts = [
    "Write a short story about a girl who overcame the shackles of sickness and achieved her dream of becoming a teacher.",
    "Write a python function that determines whether a number is prime or not.",
    "Solve the equation 3x² - 5x + 2 = 0 and explain each step clearly.",
    "List and explain the most common web security attacks.",
    "Breifly explain the probability of finding the correct passcode if the passcode consists of 6 digits ranging from 0 to 9.",
    "اكتب قصة قصيرة عن فتاة تمكنت من التغلب على قيود المرض وحققت حلمها لتصبح معلمة.",
    "اكتب دالة بايثون تحدد ما إذا كان الرقم أولياً أم لا.",
    "حل المعادلة 3x² - 5x + 2 = 0 واشرح كل خطوة بوضوح.",
    "اذكر واشرح أكثر الهجمات شيوعاً على أمن الويب.",
    "اشرح باختصار احتمال العثور على رمز مرور الصحيح إذا كان الرمز مكونًا من 6 أرقام تتراوح من 0 إلى 9.",
]

def bench(model_id):
    device = "cuda"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    torch.cuda.synchronize()

    # VRAM usage
    vram_bytes = torch.cuda.max_memory_allocated(device)
    print(f"\nModel VRAM usage (bytes): {vram_bytes}")

    elapsed, throughput = [], []
    GEN_TOKENS = 1024
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # warmup (ensures GPU kernels are initialized)
        outputs = model.generate(**inputs, max_new_tokens=64)
        torch.cuda.synchronize()

        start = time.time()
        outputs = model.generate(
            **inputs, max_new_tokens=GEN_TOKENS
        )
        torch.cuda.synchronize()
        end = time.time()

        elapsed.append(end - start)
        throughput.append(GEN_TOKENS / elapsed[-1])

        print(f"Generated {GEN_TOKENS} tokens in {elapsed[-1]:.3f} sec")
        print(f"Tokens/sec: {throughput[-1]:.2f}")

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n\nModel Output:", text)

    mean_tps = np.mean(throughput)
    std_tps = np.std(throughput)

    print(f"\nBenchamrk Results Summary:")
    print(f"\nModel Name: {model_id}")
    print(f"\nWeights Precision: {model.dtype}")
    print(f"\nModel VRAM usage (bytes): {vram_bytes}")
    print(f"\nAverage Throughput: {mean_tps:.2f} ± {std_tps:.2f} (tokens/sec)")
