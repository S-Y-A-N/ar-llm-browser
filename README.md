# Arabic LLM in the Browser

This repository will contain our senior research project code, titled "Browser-based Locally Hosted Arabic LLM Optimaization". You will find our implementation of LLM evaluation and model compression methods such as quantization here.

## File Structure

```bash
├── evaluation
│   ├── evaluate.py   # evaluation script
│   └── tasks
│       └── tasks.txt # chosen evaluation tasks
├── models            # chosen models
├── requirements.txt  # pip dependencies
└── llmini.py         # helper script to perform evaluation
```
## Getting Started

1. Fork or clone the repository into your local environment, and make sure you are inside it.
```bash
git clone https://github.com/S-Y-A-N/ar-llm-browser.git && cd ar-llm-browser
```

2. Create a python virtual environment, then install the required pip dependencies.
```bash
python -m venv .venv && pip install -r requirements.txt
```

3. If you want to run an evalation, simply use the helper script `llmini.py`:
```bash
python llmini.py <path/to/model> <path/to/tasks> --options
```
or you can run `chmod +x llmini.py` to make it an executable and run:
```bash
./llmini.py <path/to/model> <path/to/tasks> --options
```
to find out about the available options, simple run `llmini.py` with `-h` or `--help`.
