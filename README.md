# SecretGuard-CLI  
**An LLM-Powered Hard-Coded Secret Scanner for Secure Codebases**

---

## Project Overview

**SecretGuard-CLI** is a lightweight command-line tool that detects hard-coded secrets in source code using a fine-tuned Large Language Model (LLM). Built on top of [`bigcode/starcoderbase`](https://huggingface.co/bigcode/starcoderbase) and fine-tuned into [`asudarshan/secretguard-starcoderbase-lora2`](https://huggingface.co/asudarshan/secretguard-starcoderbase-lora2), this model classifies snippets as either `SECRET` or `BENIGN` through instruction-based semantic understanding.

This project explores whether an LLM-only pipeline can offer context-aware security scanning beyond traditional static regex detectors.

---

## Team Members

- **Arvind Sudarshan**  
- **Krithika Naidu**  
- **Vaishnavi Kosuri**

---

## Repository Structure

```text
SecretGuardCLI/
├── CLI_Guard.py                # Main CLI script to classify code snippets
├── LICENSE.md                  # MIT License
├── README.md                   # Project documentation
├── dataset_generator.py        # Script to generate synthetic training data
├── format_dataset.py           # Formats data for instruction-style fine-tuning
├── requirements.txt            # Project dependencies
├── Datasets/                   # Contains synthetic SECRET and BENIGN examples

├── secretguard-evaluation/     # Evaluation and Metrics Submodule
│   ├── README.md
│   ├── requirements.txt
│   ├── .vscode/
│   │   └── settings.json
│   ├── data/
│   │   ├── test_samples.json           # Test input data
│   │   └── evaluation_results/         # Output prediction files and logs
│   ├── src/
│   │   ├── model_loader.py             # Loads fine-tuned model
│   │   ├── evaluator.py                # Runs evaluation loop
│   │   ├── metrics.py                  # Precision, recall, F1, specificity, AUC
│   │   └── utils.py                    # Helper functions
│   ├── scripts/
│   │   └── run_evaluation.py           # Script to launch full evaluation
│   ├── tests/
│   │   └── test_evaluation.py          # Unit tests for evaluation pipeline
│   └── notebooks/
│       └── analysis.ipynb              # Jupyter notebook for results exploration

```

## How to Use

### 1. Clone the Repository

git clone https://github.com/xf4c70r/SecretGuardCLI.git
cd SecretGuardCLI

2. Install dependencies

pip install -r requirements.txt

3. Authenticate with Hugging Face

huggingface-cli login

The model requires access to Hugging Face Hub and will not run without a valid API token. You can get one from your Hugging Face account settings.

