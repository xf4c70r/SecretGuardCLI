#!/usr/bin/env python3
"""
Compare fine-tuned SecretGuard model with base StarCoder model
Quick comparison for course project
"""
import sys
import json
import time
from pathlib import Path
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluator import SecretGuardEvaluator
from src.metrics import EvaluationMetrics
import logging

logging.basicConfig(level=logging.WARNING)

class BaseModelEvaluator:
    """Evaluator for the base StarCoder model (without fine-tuning)"""
    
    def __init__(self, base_model="Salesforce/codegen-350M-mono", use_quantization=True):
        self.base_model = base_model
        self.use_quantization = use_quantization
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        
    def _get_device(self):
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def load_model(self):
        """Load base model without LoRA adapter"""
        print(f"Loading base model: {self.base_model}")
        
        # Check if quantization is supported
        has_gpu_support = False
        if self.use_quantization:
            try:
                import bitsandbytes as bnb
                # Check if we have GPU support
                if torch.cuda.is_available():
                    has_gpu_support = True
                elif torch.backends.mps.is_available():
                    print("⚠️  MPS device detected - quantization not supported, using full precision")
                    self.use_quantization = False
                else:
                    print("⚠️  No GPU detected - quantization not supported, using full precision")
                    self.use_quantization = False
            except ImportError:
                print("⚠️  bitsandbytes not available, falling back to full precision")
                self.use_quantization = False
        
        # Quantization config
        bnb_config = None
        if self.use_quantization and has_gpu_support:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            except Exception as e:
                print(f"⚠️  Failed to configure quantization: {e}")
                self.use_quantization = False
        
        # Load base model
        print(f"Loading model with quantization: {self.use_quantization}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto" if self.device != "mps" else None,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            trust_remote_code=True
        )
        
        if self.device == "mps":
            self.model = self.model.to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        self.model.eval()
        print("Base model loaded successfully!")
    
    def build_prompt(self, code_snippet):
        """Build prompt for base model - using same format as fine-tuned"""
        return f"""<|user|>
Does this code contain a secret?
<|code|>
{code_snippet}
<|assistant|>
"""
    
    def classify(self, code_snippet):
        """Classify using base model"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first. Call load_model() before classify()")
        
        prompt = self.build_prompt(code_snippet)
        
        try:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = decoded.split("<|assistant|>")[-1].strip()
            
            return result
            
        except Exception as e:
            print(f"Error during base model classification: {e}")
            return "Error"

def evaluate_both_models(test_data_path="data/evaluation_results/test_samples.json"):
    """Evaluate both base and fine-tuned models"""
    
    print("🔄 COMPARING BASE vs FINE-TUNED MODEL")
    print("=" * 60)
    
    # Load test data
    with open(test_data_path, 'r') as f:
        data = json.load(f)
    test_cases = data['test_cases']
    
    print(f"📊 Test dataset: {len(test_cases)} samples")
    
    results = {}
    
    # Evaluate Fine-tuned model
    print("\n🎯 Evaluating FINE-TUNED SecretGuard model...")
    start_time = time.time()
    
    finetuned_evaluator = SecretGuardEvaluator(
        hf_repo="Salesforce/codegen-350M-mono",
        base_model="Salesforce/codegen-350M-mono",
        use_quantization=True
    )
    
    finetuned_results = finetuned_evaluator.run_full_evaluation(
        test_data_path=test_data_path,
        output_dir="data/comparison_results/finetuned",
        save_detailed=True,
        generate_plots=False  # Skip plots for speed
    )
    
    finetuned_time = time.time() - start_time
    print(f"✅ Fine-tuned evaluation completed in {finetuned_time:.1f}s")
    
    # Evaluate Base model
    print("\n🏗️  Evaluating BASE StarCoder model...")
    start_time = time.time()
    
    base_evaluator = BaseModelEvaluator()
    base_evaluator.load_model()
    
    # Run predictions on base model
    base_predictions = []
    print("Running base model predictions...")
    
    for i, test_case in enumerate(test_cases):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(test_cases)}")
        
        try:
            prediction = base_evaluator.classify(test_case['code'])
            base_predictions.append(prediction)
        except Exception as e:
            print(f"Error on case {i}: {e}")
            base_predictions.append("Error")
    
    # Calculate base model metrics
    metrics_calculator = EvaluationMetrics()
    base_results = metrics_calculator.generate_detailed_report(test_cases, base_predictions)
    
    base_time = time.time() - start_time
    print(f"✅ Base model evaluation completed in {base_time:.1f}s")
    
    # Create comparison
    comparison = create_comparison_report(finetuned_results, base_results)
    
    # Save comparison results
    output_dir = Path("data/comparison_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "model_comparison.json", 'w') as f:
        json.dump({
            'finetuned_results': finetuned_results,
            'base_results': base_results,
            'comparison': comparison
        }, f, indent=2, default=str)
    
    print(f"\n💾 Comparison results saved to: {output_dir}/")
    
    return comparison

def create_comparison_report(finetuned_results, base_results):
    """Create detailed comparison report"""
    
    ft_metrics = finetuned_results['overall_metrics']
    base_metrics = base_results['overall_metrics']
    
    comparison = {
        'metric_comparison': {},
        'improvement': {},
        'summary': {}
    }
    
    # Compare each metric
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']:
        ft_value = ft_metrics[metric]
        base_value = base_metrics[metric]
        improvement = ft_value - base_value
        improvement_pct = (improvement / base_value * 100) if base_value > 0 else 0
        
        comparison['metric_comparison'][metric] = {
            'fine_tuned': ft_value,
            'base_model': base_value,
            'improvement': improvement,
            'improvement_percentage': improvement_pct
        }
    
    # Overall assessment
    avg_improvement = sum(comp['improvement'] for comp in comparison['metric_comparison'].values()) / len(comparison['metric_comparison'])
    
    comparison['summary'] = {
        'average_improvement': avg_improvement,
        'best_improvement': max(comparison['metric_comparison'].values(), key=lambda x: x['improvement']),
        'total_samples': finetuned_results['summary']['total_samples'],
        'recommendation': 'Fine-tuning beneficial' if avg_improvement > 0.05 else 'Marginal improvement'
    }
    
    return comparison

def print_comparison_summary(comparison):
    """Print formatted comparison summary"""
    
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Metric':<15} | {'Base Model':<12} | {'Fine-tuned':<12} | {'Improvement':<12} | {'% Change':<10}")
    print("-" * 75)
    
    for metric, values in comparison['metric_comparison'].items():
        print(f"{metric.title():<15} | "
              f"{values['base_model']:<12.3f} | "
              f"{values['fine_tuned']:<12.3f} | "
              f"{values['improvement']:<+12.3f} | "
              f"{values['improvement_percentage']:<+10.1f}%")
    
    print(f"\n📊 SUMMARY:")
    print(f"Average Improvement: {comparison['summary']['average_improvement']:+.3f}")
    print(f"Assessment: {comparison['summary']['recommendation']}")
    
    # Best improvements
    best_metric = max(comparison['metric_comparison'].items(), 
                     key=lambda x: x[1]['improvement'])
    print(f"Biggest Improvement: {best_metric[0].title()} ({best_metric[1]['improvement']:+.3f})")
    
    print("\n💡 INTERPRETATION:")
    avg_imp = comparison['summary']['average_improvement']
    if avg_imp > 0.1:
        print("🟢 EXCELLENT: Fine-tuning provides significant improvement!")
    elif avg_imp > 0.05:
        print("🟡 GOOD: Fine-tuning shows clear benefits")
    elif avg_imp > 0:
        print("🟠 MODERATE: Some improvement from fine-tuning")
    else:
        print("🔴 CONCERNING: Fine-tuning may not be effective")
    
    print("="*80)

def main():
    test_data_path = "data/evaluation_results/test_samples.json"
    
    if not Path(test_data_path).exists():
        print(f"❌ Test data not found: {test_data_path}")
        print("Please run the expanded evaluation first!")
        return 1
    
    try:
        comparison = evaluate_both_models(test_data_path)
        print_comparison_summary(comparison)
        
        print("\n🎯 FOR YOUR COURSE REPORT:")
        print("- Include the comparison table above")
        print("- Mention specific improvements in each metric") 
        print("- Discuss which categories benefited most from fine-tuning")
        print("- Use this to justify the value of your fine-tuning approach")
        
        return 0
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        print("\n🔧 Quick fix: If base model fails, you can:")
        print("1. Skip base model comparison for now")
        print("2. Use literature values for comparison")
        print("3. Compare against random baseline (50% accuracy)")
        return 1

if __name__ == "__main__":
    exit(main())