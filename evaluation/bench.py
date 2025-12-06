# Inference Speed (Throughput [tokens/sec], Latency [sec/token]) and VRAM usage

import argparse
import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.helpers import parse_torch_dtype

def main():
    parser = argparse.ArgumentParser(
        prog="bench",
        description="%(prog)s is a command-line interface used to quickly benchmark a model's size and inference speed.",
    )

    parser.add_argument(
        "model_config_file",
        help="- Specify a model's configuration YAML file (must contain 'name').",
    )

    # TODO pass model to a quantization function if this flag is used
    parser.add_argument(
        "-q",
        "--quantize",
        help="- Apply quantization from the precision specified in model config (usually float16) to the given parameter (int4, int8).",
    )

    args = parser.parse_args()
    print(args)

    if args.model_config_file:
        import yaml

        with open(args.model_config_file) as f:
            model_config = yaml.safe_load(f)

    model_name = model_config.get("name")
    dtype = parse_torch_dtype(model_config.get("dtype"))
    use_cache = model_config.get("use_cache", True)
    bench(model_name, dtype, use_cache)


def bench(model_name, dtype, use_cache=True):
    device = "cuda"
    torch.cuda.reset_peak_memory_stats(device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        use_cache=use_cache,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    torch.cuda.synchronize()

    # VRAM usage
    vram_bytes = torch.cuda.max_memory_allocated(device)
    vram_gb = vram_bytes / (1024**3)
    print(f"Model VRAM usage: {vram_gb:.3f} GiB")

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

    elapsed, throughput, latency = [], [], []
    GEN_TOKENS = 1024
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # warmup (ensures GPU kernels are initialized)
        outputs = model.generate(**inputs, max_new_tokens=64)
        torch.cuda.synchronize()

        start = time.time()
        outputs = model.generate(
            **inputs, max_new_tokens=GEN_TOKENS, use_cache=use_cache
        )
        torch.cuda.synchronize()
        end = time.time()

        elapsed.append(end - start)
        throughput.append(GEN_TOKENS / elapsed[-1])
        latency.append(elapsed[-1] / GEN_TOKENS)

        print(f"Generated {GEN_TOKENS} tokens in {elapsed[-1]:.3f} sec")
        print(f"Tokens/sec: {throughput[-1]:.2f}")
        print(f"Latency (sec/token): {latency[-1]:.6f}")

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n\nModel Output:", text)

    mean_tps = np.mean(throughput)
    std_tps = np.std(throughput)

    mean_spt = np.mean(latency)
    std_spt = np.std(latency)

    print("\nBenchamrk Results Summary:")
    print(f"\nModel Name: {model_name}")
    print(f"\nWeights Precision: {dtype}")
    print(f"\nModel VRAM usage: {vram_gb:.3f} GiB")
    print(f"\nAverage Throughput: {mean_tps:.2f} ± {std_tps:.2f} (tokens/sec)")
    print(f"\nAverage Latency: {mean_spt:.2f} ± {std_spt:.2f} (sec/token)")


if __name__ == "__main__":
    main()
