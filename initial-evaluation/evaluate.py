import torch
from transformers import AutoModelForCausalLM

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters


MODELS = [
  "google/gemma-3-1b-it",                   # Gemma 3 (1B)
  "Qwen/Qwen3-8B",                          # Qwen3   (8B)
  "QCRI/Fanar-1-9B-Instruct",               # Fanar   (9B)
  "humain-ai/ALLaM-7B-Instruct-preview",    # ALLaM   (7B)
  "FreedomIntelligence/AceGPT-v2-8B-Chat",  # AceGPT  (8B)
]

BENCHMARKS = [
  "leaderboard|truthfulqa:mc|0",
  "leaderboard|hellaswag|0",
  "leaderboard|mmlu:5shot|0",
  # todo: issue with finding below tasks (see: https://github.com/huggingface/lighteval/issues/260)
  "lighteval|mmlu_ara_hybrid:arabic_language_general|0",
  "lighteval|mmlu_ara_hybrid:arabic_language_grammar|0",
  "lighteval|alghafa_arc_ara_hybrid:easy|0",
  "lighteval|alghafa_openbookqa_ara_hybrid|0",
  "lighteval|alghafa_piqa_ara_hybrid|0",
  "lighteval|alghafa_race_ara_hybrid|0",
  "lighteval|alghafa_sciqa_ara_hybrid|0",
  "lighteval|exams_ara_hybrid:biology|0",
  "lighteval|exams_ara_hybrid:islamic_studies|0",
  "lighteval|exams_ara_hybrid:physics|0",
  "lighteval|exams_ara_hybrid:science|0",
  "lighteval|exams_ara_hybrid:social|0",
]

BENCHMARKS = ",".join(BENCHMARKS)

for MODEL_NAME in MODELS:
  evaluation_tracker = EvaluationTracker(output_dir="./results")
  pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.NONE
  )

  model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", dtype=torch.float16,
  )
  
  batch_size=20
  min_batch_size=1
  while batch_size >= min_batch_size:
      try:
        config = TransformersModelConfig(model_name=MODEL_NAME, batch_size=batch_size)
        model = TransformersModel.from_model(model, config)
      
        pipeline = Pipeline(
          model=model,
          pipeline_parameters=pipeline_params,
          evaluation_tracker=evaluation_tracker,
          tasks=BENCHMARKS,
        )
          
        results = pipeline.evaluate()
        break
      except RuntimeError as e:
          if "out of memory" in str(e):
              torch.cuda.empty_cache()
              batch_size = batch_size // 2
              print(f"CUDA OOM, retrying with batch_size={batch_size}")
          else:
              raise e



  pipeline.show_results()
  results = pipeline.get_results()
  evaluation_tracker.save()
