import argparse


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        prog="llmini",
        description="%(prog)s is a command-line interface used to quickly compress and evaluate LLMs in one go.",
    )

    parser.add_argument(
        "model_id",
        help="Specify a model's identifier on Hugging Face.",
    )

    parser.add_argument(
        "-q",
        "--quantize",
        choices=["int8", "int4"],
        help="Apply a quantization format.",
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
        if args.quantize:
            apply_quantization(args.model_id, args.quantize)
        elif args.prune:
            apply_pruning(args.model_id, args.prune, args.prune_config)


def apply_quantization(model_id, format):
    print(f"Applying quantization format `{format}` on `{model_id}`")


def apply_pruning(model_id: str, method: str, prune_config: str | None = None):
    """
    NOTE: currently, pruning `method` is not used.

    Args:
        model_id: Hugging Face model identifier.
        method: Pruning method to apply ('sparsegpt', 'wanda').
        recipe_path: Path to `llmcompressor` pruning recipe.
    """
    print(f"Applying pruning method `{method}` on `{model_id}`")

    if prune_config:
        prune_config = dict(config.split("=", 1) for config in prune_config.split(","))

    from llmini.pruning.prune import prune

    prune(model_id, method, **prune_config)


if __name__ == "__main__":
    main()
