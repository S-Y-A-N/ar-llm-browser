import torch.nn
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
from llmcompressor.modifiers.pruning.wanda import WandaPruningModifier
from typing import Literal


PRUNE_METHOD = Literal["sparsegpt", "wanda"]


def load_calibration_dataset(dataset_id, data_files, num_samples=128, seed=42):
    datasets = [
        load_dataset(dataset_id, data_files=data_file, split="train")
        for data_file in tqdm(data_files, desc="Loading datasets", unit="file")
    ]
    return concatenate_datasets(datasets).shuffle(seed=seed).select(range(num_samples))


def get_block_name(model) -> str | None:
    """Extract the transformer block class name from a model. Used for `sequential_targets`.

    NOTE: Temporary solution due to incorrect inference of sequential targets. Likely to be removed or modified later.
    """
    paths = [
        ["model", "layers"],
        ["transformer", "h"],
        ["model", "decoder", "layers"],
    ]

    for path in paths:
        try:
            obj = model
            for attr in path:  # nesting attributes from path
                obj = getattr(obj, attr)

            if len(obj) > 0 and isinstance(
                obj, (torch.nn.ModuleList, torch.nn.Sequential)
            ):
                return obj[0].__class__.__name__
        except (AttributeError, TypeError):
            continue

    return None


def get_recipe(
    method,
    sparsity: float = 0.5,
    sparsity_profile: str | None = None,
    mask_structure: str = "0:0",
    targets: str | list[str] = ["Linear"],
    sequential_targets: str | list[str] | None = None,
    ignore: list[str] = ["re:.*lm_head"],
):
    prune_args = {
        "sparsity": sparsity,
        "sparsity_profile": sparsity_profile,
        "mask_structure": mask_structure,
        "sequential_targets": sequential_targets,
        "targets": targets,
        "ignore": ignore,
    }
    match method:
        case "sparsegpt":
            return [
                SparseGPTModifier(**prune_args, dampening_frac=0.01, block_size=128)
            ]
        case "wanda":
            return [WandaPruningModifier(**prune_args)]
        case _:
            raise NotImplementedError(
                f"The pruning method {method} is invalid."
                f"Supported methods: {PRUNE_METHOD.__args__}"
            )


def validate(model, tokenizer):
    print("\n========== SAMPLE GENERATION ==============")
    model.tie_weights()
    inputs = tokenizer("The biggest planet in our solar system is", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0]))
    print("==========================================\n")
