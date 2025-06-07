# SecretGuard Model Evaluation Framework

A comprehensive evaluation framework for the SecretGuard model (StarCoder-based LoRA fine-tuned for secret detection in code).

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ 
- MacBook Air M3 (optimized for Apple Silicon)
- 8GB+ RAM
- VS Code with Python extension

### Setup

1. **Clone and navigate to the project:**
```bash
git clone <your-repo-url>
cd secretguard-evaluation
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r eval_requirements.txt
```

4. **Set up VS Code:**
   - Open the project folder in VS Code
   - Select the virtual environment as Python interpreter (Cmd+Shift+P → "Python: Select Interpreter")
   - Install recommended extensions when prompted

### Quick Test

Run a quick test to verify everything works:

```bash
python scripts/run_evaluation.py --quick-test
```

### Full Evaluation

Run the complete evaluation suite:

```bash
python scripts/run_evaluation.py
```

## 📊 Evaluation Metrics

The framework provides comprehensive evaluation metrics:

### Core Metrics
- **Accuracy**: Overall correctness
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives  
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate
- **False Positive Rate**: Incorrectly flagged safe code
- **False Negative Rate**: Missed secrets

### Category-wise Analysis
Performance breakdown by secret type:
- AWS keys (access keys, secret keys)
- API keys (OpenAI, GitHub tokens)
- Database credentials
- JWT tokens
- Hardcoded passwords
- Safe code (control group)

### Performance Metrics
- Inference time per prediction
- Throughput (predictions per second)
- Error rate
- Memory usage

## 🏗️ Project Structure

```
secretguard-evaluation/
├── Evaluation_README.md                    # This file
├── eval_requirements.txt             # Python dependencies
├── .vscode/
│   └── settings.json           # VS Code configuration
├── data/
│   ├── test_samples.json       # Test dataset
│   └── evaluation_results/     # Output directory
├── src/
│   ├── model_loader.py         # Model loading and inference
│   ├── evaluator.py            # Main evaluation logic
│   ├── metrics.py              # Metrics calculation
│   └── utils.py                # Utility functions
├── tests/
│   └── test_evaluation.py      # Unit tests
├── scripts/
│   └── run_evaluation.py       # Main evaluation script
└── notebooks/
    └── analysis.ipynb          # Jupyter notebook for analysis
```

## 🔧 Configuration

### Model Configuration

The default configuration uses:
- **Base Model**: `bigcode/starcoderbase`
- **LoRA Adapter**: `asudarshan/secretguard-starcoderbase-lora2`
- **Quantization**: 4-bit (recommended for M3 Mac)

To modify:

```bash
python scripts/run_evaluation.py \
    --hf-repo "your-custom-repo" \
    --base-model "your-base-model" \
    --no-quantization  # Use full precision
```

### Test Data Format

Test cases in `data/test_samples.json`:

```json
{
  "test_cases": [
    {
      "id": "unique_id",
      "code": "const password = 'secret123';",
      "expected": "Yes",
      "category": "password",
      "description": "Hardcoded password example"
    }
  ]
}
```

## 📈 Understanding Results

### Example Console Output
```
SECRETGUARD MODEL EVALUATION SUMMARY
============================================================
Overall Performance:
  Accuracy:    0.950
  Precision:   0.823
  Recall:      0.960
  F1-Score:    0.841
  Specificity: 0.333

Performance by Category:
  password            | F1: 0.950 | Samples: 500
  api_key            | F1: 0.900 | Samples: 400
  safe               | F1: 0.967 | Samples: 800
```

### Output Files

1. **`evaluation_results.json`**: Complete detailed results
2. **`summary_results.json`**: Key metrics summary
3. **`confusion_matrix.png`**: Visual confusion matrix
4. **`metrics_by_category.png`**: Category performance charts

### Key Metrics Interpretation

- **High Precision (>0.9)**: Few false positives (safe code flagged as secrets)
- **High Recall (>0.9)**: Few false negatives (missed actual secrets)
- **Balanced F1-Score**: Good overall performance
- **Low False Positive Rate**: Won't annoy developers with false alarms

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_evaluation.py::TestEvaluationMetrics::test_basic_metrics_calculation -v
```

## 🚨 Common Issues & Solutions

### Memory Issues
If you encounter OOM errors:
```bash
python scripts/run_evaluation.py --no-quantization=false
```

### Model Loading Errors
1. Check internet connection for HuggingFace downloads
2. Verify HF repository exists and is accessible
3. Try without quantization if BitsAndBytes issues occur

### MPS Device Issues (M3 Mac)
If MPS errors occur:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
python scripts/run_evaluation.py
```

## Base Model Comparison
```bash
python scripts/compare_with_base.py
```
What it does: Evaluates both fine-tuned SecretGuard and base StarCoder models on the same test dataset, then generates a comparison report showing improvement metrics.
Output: Comparison table with accuracy, precision, recall, F1-score improvements + saves detailed results to data/comparison_results/

## 📊 Advanced Usage

### Custom Test Data

Create your own test cases:

```python
test_cases = [
    {
        "id": "custom_1",
        "code": "const mySecret = 'sk-custom123';",
        "expected": "Yes", 
        "category": "custom_api_key"
    }
]

# Save to JSON and run evaluation
```

### Programmatic Usage

```python
from src.evaluator import SecretGuardEvaluator

evaluator = SecretGuardEvaluator()
evaluator.load_model()

# Single prediction
result = evaluator.run_single_prediction("const pass = 'secret';")
print(f"Prediction: {result['prediction']}")

# Batch evaluation
results = evaluator.run_full_evaluation("path/to/test_data.json")
```

### Integration with CI/CD

Add to your GitHub Actions:

```yaml
- name: Run Security Evaluation
  run: |
    python scripts/run_evaluation.py --test-data tests/security_tests.json
    # Add threshold checks based on metrics
```


**Optimization Tips for M3**:
- Use quantization (default) to reduce memory usage
- Enable MPS acceleration for faster inference
- Batch size of 1 recommended for memory efficiency
- Close other applications during evaluation for best performance