import torch
from transformers import AutoModelForCausalLM
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_package_available

if is_package_available("accelerate"):
    from datetime import timedelta
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
    print(f"Using Accelerator...")
else:
    accelerator = None
    
import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)

MODELS = config["models"]
BENCHMARKS = config["benchmarks"]
MAX_SAMPLES = config["params"]["max_samples"]

tasks = ",".join(BENCHMARKS)

evaluation_tracker = EvaluationTracker(output_dir="init-eval/results")
pipeline_params = PipelineParameters(
  launcher_type=ParallelismManager.ACCELERATE,
  max_samples=MAX_SAMPLES
)
  
for m in MODELS:
  model_name=m["name"]
  bs=m["batch_size"]
  
  # initialize model
  model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", dtype="auto"
  )
  print(f"Model Name: {model_name}")
  print(f"Batch Size: {bs}")
  print(f"Model Instance: {model}")
  
  # dynamic batching to avoid CUDA OOM
  while bs >= 1:
    try:
      # in case of CUDA OOM, avoid rewrapping, only change BS
      if isinstance(model, TransformersModel):
        model.config.batch_size = bs
        print(f"Retrying with batch_size={bs}")
      else:
        model_config = TransformersModelConfig(model_name=model_name, batch_size=bs)
        model = TransformersModel.from_model(model, model_config)
        print(f"Transformers Model Instance: {model}")

    
      pipeline = Pipeline(
        model=model,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        tasks=tasks,
      )
      print(f"Pipeline Instance: {pipeline}")

      print("Evaluating...")
      results = pipeline.evaluate()
      break
    # handle CUDA OOM exception
    except RuntimeError as e:
      if "out of memory" in str(e):
        torch.cuda.empty_cache()
        print(f"CUDA OOM. model='{model_name}', batch_size={bs}.")
        bs = bs // 2
      else:
        raise e
  else:
    print(f"model='{model_name}' could not complete evaluation on batch_size={bs}.")
    



  pipeline.show_results()
  results = pipeline.get_results()
  evaluation_tracker.save()
