# Arabic LLM in the Browser

This repository contains the code for our senior research project, titled "Browser-based Locally Hosted Arabic LLM Optimaization".

You will find our implementation of LLM evaluation and model compression methods such as quantization here.

## Installation

1.  Clone the repository into your local environment, then move inside it.
    ```sh
    git clone https://github.com/S-Y-A-N/ar-llm-browser.git
    cd ar-llm-browser
    ```

2. Install the dependencies using [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) (Recommended).
    ```sh
    uv sync
    ```
    alternatively, if you don't want to install or use uv, run:
    ```sh
    pip install -e .
    ```

## Running Evaluation

We use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to run model evaluations.

Example for running evaluation on `gemma-3-1b-it`:
```sh
lm_eval --config evaluation/config/gemma-3-1b-it.yaml \ # path to YAML config file
        --tasks metabench arabicmmlu \
        --log_samples \
        --output_path results
```