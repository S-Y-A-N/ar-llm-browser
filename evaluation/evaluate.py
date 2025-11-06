import torch
from transformers import AutoModelForCausalLM
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
   
def evaluate(model_config, tasks, max_samples=100):
  evaluation_tracker = EvaluationTracker(output_dir="./init-eval/results")
  pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.ACCELERATE,
    max_samples=max_samples,
    load_tasks_multilingual=True
  )
  
  model_name = model_config['name']
  batch_size = model_config['batch_size']
    
  # initialize Transformers model
  tf_model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", dtype="auto"
  )
  print(f"Model Name: {model_name}")
  print(f"Batch Size: {batch_size}")
  print(f"Model Instance: {tf_model}")
  
  # dynamic batching to avoid CUDA OOM
  while batch_size >= 1:
    try:
      # in case of CUDA OOM, avoid rewrapping, only change BS
      if isinstance(tf_model, TransformersModel):
        tf_model.config.batch_size = batch_size
        print(f"Retrying with batch_size={batch_size}")
      else:
        tf_model_config = TransformersModelConfig(model_name=model_name, batch_size=batch_size)
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
        torch.cuda.empty_cache()
        print(f"CUDA OOM. model='{model_name}', batch_size={batch_size}.")
        batch_size = batch_size // 2
      else:
        raise e
  else:
    print(f"model='{model_name}' could not complete evaluation on batch_size={batch_size}.")
      
  pipeline.show_results()
  evaluation_tracker.save()
