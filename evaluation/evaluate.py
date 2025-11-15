import torch
from transformers import AutoModelForCausalLM
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import (
    TransformersModel,
    TransformersModelConfig,
)
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

import gc
import os
from dotenv import load_dotenv

load_dotenv()
hf_org = os.getenv("HF_ORG")
print("HF_ORG=", os.getenv("HF_ORG"))

def parse_torch_dtype(dtype_str):
    # remove "torch." prefix if present
    name = dtype_str.split(".")[-1]
    return getattr(torch, name)


def evaluate(model_config, tasks, max_samples=None, batch_size=1):
    gc.collect()
    torch.cuda.empty_cache()
    max_samples = None if max_samples == None else int(max_samples)

    evaluation_tracker = EvaluationTracker(
        output_dir="./evaluation/results",
        push_to_hub=True,
        hub_results_org=hf_org,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        max_samples=max_samples,
        load_tasks_multilingual=True,
    )

    model_name = model_config["name"]
    torch_dtype = parse_torch_dtype(model_config["torch_dtype"])
    batch_size = model_config["batch_size"] if batch_size == None else int(batch_size)

    # initialize Transformers model
    with torch.no_grad():            
        # dynamic batching to avoid CUDA OOM
        while batch_size >= 1:
            try:
                tf_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    use_cache=False
                )
                print("Model Name:", model_name)
                print("Batch Size:", batch_size)
                print("Precision (dtype):", tf_model.config.dtype)
                print("Model Instance:", tf_model)
                
                # in case of CUDA OOM, avoid rewrapping, only change BS
                tf_model_config = TransformersModelConfig(
                    model_name=model_name,
                    batch_size=batch_size,
                )
                tf_model = TransformersModel.from_model(tf_model, tf_model_config)
                print(f"Transformers Model Instance: {tf_model}")

                pipeline = Pipeline(
                    model=tf_model,
                    pipeline_parameters=pipeline_params,
                    evaluation_tracker=evaluation_tracker,
                    tasks=tasks,
                )
                print(f"Pipeline Instance: {pipeline}")

                print("Evaluating...")
                pipeline.evaluate()
                break
            # handle CUDA OOM exception
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"CUDA OOM. model='{model_name}', batch_size={batch_size}.")
                    batch_size = batch_size // 2
                    del tf_model
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    raise e
        else:
            print(f"model='{model_name}' could not complete evaluation.")
            exit()

    pipeline.show_results()
    pipeline.save_and_push_results()
    print("Evaluation Completed!")
