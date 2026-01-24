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
        help="Apply a quantization format. Note: If pruning is specified, quantization is always applied first.",
    )

    prune_group = parser.add_argument_group('pruning options')
    prune_group.add_argument(
        "--prune",
        "-p",
        choices=["sparsegpt", "wanda"],
        help="Choose a pruning algorithm.",
    )
    prune_group.add_argument(
        "--recipe",
        "-r",
        type=str,
        help="Path to a pruning `llmcompressor` recipe YAML file.",
    )
    
    args = parser.parse_args()
    
    if args.prune and not args.recipe:
        parser.error("--prune or -p requires --recipe or -r")

    return args


def main():
    args = parse_args()

    if args.model_id:
        # quantize first
        if args.quantize:
            apply_quantization(args.model_id, args.quantize)
            # then prune
            if args.prune:
                apply_pruning(args.model_id, args.prune, args.recipe)
        # or prune without quantization
        elif args.prune:
            apply_pruning(args.model_id, args.prune, args.recipe)
    return


def apply_quantization(model_id, format):
    print(f"Applying quantization format `{format}` on `{model_id}`")
    return


def apply_pruning(model_id: str, method: str, recipe_path: str):
    """
    NOTE: currently, pruning `method` is not used.
    
    Args:
        model_id: Hugging Face model identifier.
        method: Pruning method to apply ('sparsegpt', 'wanda').
        recipe_path: Path to `llmcompressor` pruning recipe.
    """
    print(f"Applying pruning method `{method}` on `{model_id}`")
    from llmini.pruning.prune import prune
    prune(model_id, recipe_path)

if __name__ == "__main__":
    main()
