import torch.nn
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets


def load_calibration_dataset(dataset_id, data_files, num_samples=256, seed=42):
    datasets = [
        load_dataset(dataset_id, data_files=data_file, split="train")
        for data_file in tqdm(data_files, desc="Loading datasets", unit="file")
    ]
    return concatenate_datasets(datasets).shuffle(seed=seed).select(range(num_samples))


def get_block_name(model) -> str | None:
    """Extract the transformer block class name from a model. Used for `sequential_targets`.

    :NOTE: Temporary solution due to incorrect inference of sequential targets. Likely to be removed or modified later.
    """
    paths = [
        ["model", "layers"],
        ["transformer", "h"],
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
