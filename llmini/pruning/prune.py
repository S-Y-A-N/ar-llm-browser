import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier

from datetime import datetime
from llmini.pruning.helpers import (
    PRUNE_METHOD,
    load_calibration_dataset,
    get_block_name,
    get_recipe,
    validate,
)

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
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQUENCE_LENGTH = 2048
DATA_FILES = [
    {"train": "en/c4-train.00000-of-01024.json.gz"},  # c4/en, first shard
]


def prune(
    model_id: str,
    method: PRUNE_METHOD,
    sparsity: float = 0.5,
    sparsity_profile: str | None = None,
    mask_structure: str = "0:0",
):
    """Apply pruning on a model based on the provided recipe arguments.

    Args:
        model_id: Model identifier on Hugging Face (e.g., `google/gemma-3-1b-it`)
        method: Pruning method. Supported: 'wanda', 'sparsegpt'

    **Recipe Args:** Refer to [llmcompressor documentation](https://docs.vllm.ai/projects/llm-compressor/en/latest/reference/llmcompressor/modifiers/pruning/) for more information.

        Supported Arguments:
            - **sparsity:** Sparsity to compress the model to. Value between (0, 1). Default to 0.5 (50% sparsity).
            - **sparsity_profile:** Can be set to 'owl' to use Outlier Weighed Layerwise Sparsity (OWL), more information can be found in the paper https://arxiv.org/pdf/2310.05175
            - **mask_structure:** String to define the structure of the mask to apply. Must be of the form N:M where N, M are integers that define a custom block shape. Defaults to 0:0 which represents an unstructured mask.
    """
    print(locals())

    # create the pruned model directory name
    recipe_str = f"{method}_{int(sparsity * 100)}"

    if mask_structure != "0:0":
        recipe_str += f"_{mask_structure.replace(':', 'of')}"

    if sparsity_profile is not None:
        recipe_str += f"_{sparsity_profile}"

    output_dir = f"models/{model_id}_{recipe_str}_" + datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    # device information
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # model information
    logger.info(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
    logger.info(model)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # load calibration dataset
    dataset = load_calibration_dataset(
        DATASET_ID, DATA_FILES, num_samples=NUM_CALIBRATION_SAMPLES
    )

    # infer sequential targets and create pruning recipe
    sequential_targets = get_block_name(model)
    logger.info(f"Sequential targets: {sequential_targets}")
    recipe = get_recipe(
        method,
        sparsity=sparsity,
        sparsity_profile=sparsity_profile,
        mask_structure=mask_structure,
        sequential_targets=sequential_targets,
    )

    # apply pruning
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        output_dir=output_dir,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        # sequential_targets=sequential_targets,
    )
    logger.info(f"Compressed model saved to: {output_dir}")
    logger.info(model)

    # generate a sample by the pruned model
    validate(model, tokenizer)
