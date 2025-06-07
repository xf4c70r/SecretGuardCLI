"""
Main evaluator class for SecretGuard model
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

from .model_loader import SecretGuardModelLoader
from .metrics import EvaluationMetrics

logger = logging.getLogger(__name__)

class SecretGuardEvaluator:
    """Main evaluation class for SecretGuard model"""
    
    def __init__(self, 
                 hf_repo: str = "asudarshan/secretguard-starcoderbase-lora2",
                 base_model: str = "bigcode/starcoderbase",
                 use_quantization: bool = True):
        """
        Initialize the evaluator
        
        Args:
            hf_repo: HuggingFace repository for the LoRA adapter
            base_model: Base model name
            use_quantization: Whether to use quantization
        """
        self.model_loader = SecretGuardModelLoader(
            hf_repo=hf_repo,
            base_model=base_model,
            use_quantization=use_quantization
        )
        self.metrics_calculator = EvaluationMetrics()
        self.model = None
        self.tokenizer = None
        
    def load_model(self) -> None:
        """Load the model and tokenizer"""
        logger.info("Loading SecretGuard model...")
        self.model, self.tokenizer = self.model_loader.load_model()
        logger.info("Model loaded successfully!")
    
    def load_test_data(self, test_data_path: str) -> List[Dict[str, Any]]:
        """
        Load test data from JSON file
        
        Args:
            test_data_path: Path to test data JSON file
            
        Returns:
            List of test cases
        """
        try:
            with open(test_data_path, 'r') as f:
                data = json.load(f)
            
            test_cases = data.get('test_cases', [])
            logger.info(f"Loaded {len(test_cases)} test cases from {test_data_path}")
            return test_cases
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            raise
    
    def run_single_prediction(self, code_snippet: str) -> Dict[str, Any]:
        """
        Run prediction on a single code snippet
        
        Args:
            code_snippet: Code to analyze
            
        Returns:
            Prediction result with timing info
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        prediction = self.model_loader.classify(code_snippet)
        end_time = time.time()
        
        return {
            'prediction': prediction,
            'inference_time': end_time - start_time,
            'code_length': len(code_snippet)
        }
    
    def run_batch_evaluation(self, test_cases: List[Dict[str, Any]], 
                           batch_size: int = 1) -> Dict[str, Any]:
        """
        Run evaluation on a batch of test cases
        
        Args:
            test_cases: List of test cases
            batch_size: Batch size for processing (currently only supports 1)
            
        Returns:
            Evaluation results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Running evaluation on {len(test_cases)} test cases...")
        
        predictions = []
        inference_times = []
        errors = []
        
        # Process each test case
        for i, test_case in enumerate(tqdm(test_cases, desc="Evaluating")):
            try:
                result = self.run_single_prediction(test_case['code'])
                predictions.append(result['prediction'])
                inference_times.append(result['inference_time'])
                
            except Exception as e:
                logger.error(f"Error processing test case {i}: {e}")
                predictions.append("Error")
                inference_times.append(0.0)
                errors.append({
                    'test_case_id': test_case.get('id', f'case_{i}'),
                    'error': str(e)
                })
        
        # Calculate metrics
        evaluation_report = self.metrics_calculator.generate_detailed_report(
            test_cases, predictions
        )
        
        # Add timing and performance info
        evaluation_report['performance'] = {
            'total_inference_time': sum(inference_times),
            'average_inference_time': sum(inference_times) / len(inference_times) if inference_times else 0,
            'predictions_per_second': len(predictions) / sum(inference_times) if sum(inference_times) > 0 else 0,
            'errors': errors,
            'error_rate': len(errors) / len(test_cases) if test_cases else 0
        }
        
        # Add model info
        evaluation_report['model_info'] = self.model_loader.get_model_info()
        
        return evaluation_report
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save evaluation results to JSON file
        
        Args:
            results: Evaluation results
            output_path: Path to save results
        """
        try:
            # Create output directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                return obj
            
            # Save results
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=convert_numpy)
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def generate_visualizations(self, results: Dict[str, Any], 
                              output_dir: str) -> None:
        """
        Generate and save visualization plots
        
        Args:
            results: Evaluation results
            output_dir: Directory to save plots
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Extract data for plotting
            test_cases = []
            predictions = []
            
            for detail in results['detailed_results']:
                test_cases.append({
                    'expected': detail['expected'],
                    'category': detail['category']
                })
                predictions.append(detail['predicted'])
            
            # Normalize data
            y_true = self.metrics_calculator.normalize_ground_truth(
                [case['expected'] for case in test_cases]
            )
            y_pred = self.metrics_calculator.normalize_predictions(predictions)
            
            # Generate confusion matrix plot
            cm_fig = self.metrics_calculator.plot_confusion_matrix(
                y_true, y_pred, 
                save_path=str(output_path / "confusion_matrix.png")
            )
            plt.close(cm_fig)
            
            # Generate category metrics plot
            if results['category_metrics']:
                cat_fig = self.metrics_calculator.plot_metrics_by_category(
                    results['category_metrics'],
                    save_path=str(output_path / "metrics_by_category.png")
                )
                plt.close(cat_fig)
            
            logger.info(f"Visualizations saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            raise
    
    def run_full_evaluation(self, 
                          test_data_path: str,
                          output_dir: str = "data/evaluation_results",
                          save_detailed: bool = True,
                          generate_plots: bool = True) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline
        
        Args:
            test_data_path: Path to test data JSON file
            output_dir: Directory to save results
            save_detailed: Whether to save detailed results
            generate_plots: Whether to generate visualization plots
            
        Returns:
            Evaluation results
        """
        logger.info("Starting full evaluation pipeline...")
        
        # Load test data
        test_cases = self.load_test_data(test_data_path)
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Run evaluation
        results = self.run_batch_evaluation(test_cases)
        
        # Print summary
        self.metrics_calculator.print_summary(results)
        
        # Save results
        if save_detailed:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            self.save_results(results, str(output_path / "evaluation_results.json"))
            
            # Save summary metrics only
            summary_results = {
                'overall_metrics': results['overall_metrics'],
                'category_metrics': results['category_metrics'],
                'summary': results['summary'],
                'model_info': results['model_info'],
                'performance': results['performance']
            }
            self.save_results(summary_results, str(output_path / "summary_results.json"))
        
        # Generate visualizations
        if generate_plots:
            self.generate_visualizations(results, output_dir)
        
        logger.info("Evaluation completed successfully!")
        return results
    
    def quick_test(self, code_snippets: List[str]) -> List[Dict[str, Any]]:
        """
        Quick test on a list of code snippets
        
        Args:
            code_snippets: List of code snippets to test
            
        Returns:
            List of prediction results
        """
        if self.model is None:
            self.load_model()
        
        results = []
        for i, code in enumerate(code_snippets):
            try:
                result = self.run_single_prediction(code)
                results.append({
                    'index': i,
                    'code': code,
                    'prediction': result['prediction'],
                    'inference_time': result['inference_time'],
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'code': code,
                    'prediction': 'Error',
                    'inference_time': 0.0,
                    'error': str(e),
                    'status': 'error'
                })
        
        return results