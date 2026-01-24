import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot

from pathlib import Path
from datetime import datetime
from llmini.pruning.helpers import load_calibration_dataset, get_block_name

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Calibration data configuration
# as used in the SparseGPT paper:
DATASET_ID = "allenai/c4"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048
# NOTE: SparseGPT paper does random sampling for 128 samples from the first 'en' shard.
#       This code additionally samples from the first 'ar' shard, for a total of 256 samples.
DATA_FILES = [
    {
        "train": "multilingual/c4-ar.tfrecord-00000-of-01024.json.gz"
    },  # c4/ar, first shard
    {"train": "en/c4-train.00000-of-01024.json.gz"},  # c4/en, first shard
]


def prune(model_id, recipe_path):
    """Apply pruning on a model based on the provided `llmcompressor` recipe.

    Args:
        model_id: Model identifier on Hugging Face (e.g., `google/gemma-3-1b-it`)
        recipe_path: Path to `llmcompressor` recipe YAML file (e.g., `llmini/pruning/config/sparsegpt_50.yaml`)
    """
    output_dir = (
        "models/"
        + model_id
        + "_"
        + Path(recipe_path).stem
        + "_"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")
    logger.info(f"Loading model: {model_id}")

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

    logger.info(f"dtype: {model.dtype}")
    logger.info(model)
    logger.info(f"Loading tokenizer for: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_calibration_dataset(
        DATASET_ID, DATA_FILES, num_samples=NUM_CALIBRATION_SAMPLES
    )
    sequential_targets = get_block_name(model)

    logger.info(f"Sequential targets: {sequential_targets}")

    logger.info(f"Applying compression recipe: {recipe_path}")
    model = oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe_path,
        output_dir=output_dir,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        # sequential_targets=sequential_targets,
    )
    logger.info(f"Compressed model saved to: {output_dir}")

    validate(model, tokenizer)


def validate(model, tokenizer):
    print("\n========== SAMPLE GENERATION ==============")
    inputs = tokenizer("Who are you?", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0]))
    print("==========================================\n")
