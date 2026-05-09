import argparse


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        prog="llmini",
        description="%(prog)s is a command-line interface used to quickly compress and evaluate LLMs in one go.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "model_id",
        help="Specify a model's identifier on Hugging Face.",
    )

    parser.add_argument(
        "--bench",
        action="store_true",
        help="Benchmark a model's size in memory and token generation per second.",
    )

    quant_group = parser.add_argument_group("quantization options")
    quant_group.add_argument(
        "--quant",
        "-q",
        choices=["int4", "int8", "w4a16", "w8a16", "w8a8"],
        help="Choose a quantization algorithm.\n`int8` refers to the LLM.int8() algorithm.\n`int4` refers to QLoRa.\n`w4a16` and `w8a16` refer to AWQ weight-only int4 and int8 respectively.\n`w8a8` refer to AWQ wight+activation int8 with SmoothQuant.",
    )

    prune_group = parser.add_argument_group("pruning options")
    prune_group.add_argument(
        "--prune",
        "-p",
        choices=["sparsegpt", "wanda"],
        help="Choose a pruning algorithm.",
    )
    prune_group.add_argument(
        "--prune-config",
        type=str,
        default={},
        help="Specify pruning configuration as a comma-separated string. If not specified, defaults to unstructured 50%% sparsity. Example: --prune-config sparsity=0.7,mask_structure=2:4,sparsity_profile=owl",
    )

    args = parser.parse_args()
    if args.prune_config and not args.prune:
        parser.error("--prune-config cannot be used without --prune")

    return args


def main():
    args = parse_args()

    if args.model_id:
        if args.quant:
            apply_quantization(args.model_id, args.quant)
        elif args.prune:
            apply_pruning(args.model_id, args.prune, args.prune_config)
        elif args.bench:
            benchmark(args.model_id)


def apply_quantization(model_id, method):
    print(f"Applying quantization method `{method}` on `{model_id}`")

    from llmini.quantizaton.quantize import quantize

    quantize(model_id, method)


def apply_pruning(model_id: str, method: str, prune_config: str | None = None):
    """
    Args:
        model_id: Hugging Face model identifier.
        method: Pruning method to apply ('sparsegpt', 'wanda').
        prune_config: Optionally provide pruning config as comma-seperated string. Example: sparsity=0.7,mask_structure=2:4,sparsity_profile=owl
    """
    print(f"Applying pruning method `{method}` on `{model_id}`")

    if prune_config:
        prune_config = dict(config.split("=", 1) for config in prune_config.split(","))

    from llmini.pruning.prune import prune

    prune(model_id, method, **prune_config)


def benchmark(model_id):
    from llmini.evaluation.bench import bench

    bench(model_id)


if __name__ == "__main__":
    main()
