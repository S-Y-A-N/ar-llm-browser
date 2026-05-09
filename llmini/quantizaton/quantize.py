from typing import Literal
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

QUANT_METHOD = Literal[
    "int4",
    "awq-int4-w"  # QloRA vs AWQ W4A16
    "int8",
    "awq-int8-w",
    "awq-int8-wa",  # LLM.8bit() vs AWQ W8A16 vs AWQ W8A8
]

# For AWQ, we need calibration data to estimate activation scales
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


def quantize(model_id: str, method: QUANT_METHOD):
    quant_config = BitsAndBytesConfig()

    match method:
        case "int4":  # QLoRA (int4)
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
        case "int8":  # LLM.int8()
            quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6)
        case "awq-int4-w":  # W4A16 (weight only)
            recipe = [
                AWQModifier(scheme="W4A16", ignore=["lm_head"]),
            ]
        case "awq-int8-w":  # W8A16 (weight only)
            recipe = [
                AWQModifier(scheme="W8A16", ignore=["lm_head"]),
            ]
        case "awq-int8-wa":  # W8A8 (weight + activation)
            recipe = [
                SmoothQuantModifier(smoothing_strength=0.5),
                AWQModifier(scheme="W8A8", ignore=["lm_head"]),
            ]
        case _:
            raise NotImplementedError(
                f"The quantization method {method} is invalid."
                f"Supported methods: {QUANT_METHOD.__args__}"
            )

    output_dir = f"models/{model_id}_{method}_" + datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="auto", quantization_config=quant_config
    )

    # if using GPTQ or AWQ...
    if any(m in method for m in ["awq", "gptq"]):
        from llmcompressor import oneshot

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        calibration_data = load_calibration_data(
            DATASET_ID, tokenizer, num_samples=NUM_CALIBRATION_SAMPLES
        )

        oneshot(
            model=model,
            dataset=calibration_data,
            recipe=recipe,
            max_seq_length=MAX_SEQUENCE_LENGTH,
            num_calibration_samples=NUM_CALIBRATION_SAMPLES,
            output_dir=output_dir,
        )
        return
    model.save_pretrained(output_dir)


def load_calibration_data(
    dataset_id: str, tokenizer: AutoTokenizer, num_samples: int = 512
):
    # Load dataset
    ds = load_dataset(dataset_id, split=f"train_sft[:{num_samples}]")
    ds = ds.shuffle(seed=42)

    # Preprocess the data into the format the model is trained with
    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)

    # Tokenize the data (be careful with bos tokens - we need add_special_tokens=False since the chat_template already added it).
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    return ds
