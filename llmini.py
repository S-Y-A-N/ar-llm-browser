#!/usr/bin/env python3
import argparse
import os
from dotenv import load_dotenv
load_dotenv()
print("HF_HOME=", os.getenv("HF_HOME"))

def main():
  parser = argparse.ArgumentParser(
    prog="llmini",
    description="%(prog)s is a command-line interface used to quickly compress and evaluate LLMs in one go.",
  )

  parser.add_argument("model_config_file", help="- Specify a model's configuration YAML file (must contain 'name' and 'batch_size').")
  parser.add_argument("tasks", help="- Specify the evaluation tasks to run. Lighteval is the evaluation framewrok used. For task syntax and available tasks, refer to this page from Lighteval's documentation: https://huggingface.co/docs/lighteval/en/quicktour#task-specification")
  parser.add_argument("-s", "--samples", help="- Specify max samples for evaluation datasets.")
  parser.add_argument("-b", "--batch", help="- Specify batch size (number of samples to run per iteration).")
  parser.add_argument("-q", "--quantize", help="- Apply quantization.")
  
  args = parser.parse_args()
  print(args)
  
  if args.model_config_file and args.tasks:
    import evaluation.evaluate as eval
    import yaml
    with open(args.model_config_file) as f:
      model_config = yaml.safe_load(f)
    eval.evaluate(model_config, args.tasks, args.samples, args.batch)
    if args.quantize:
      quantized_model = apply_quantization(model_config)
      eval.evaluate(quantized_model)
  return

def apply_quantization(model_config):
  print("Applying quantization on {model_config.name}")
  return
  
  
if __name__ == "__main__":
  main()