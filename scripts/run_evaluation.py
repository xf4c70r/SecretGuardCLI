#!/usr/bin/env python3
"""
Main script to run SecretGuard model evaluation
Usage: python scripts/run_evaluation.py
"""
import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluator import SecretGuardEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

def main():
    parser = argparse.ArgumentParser(description='Evaluate SecretGuard Model')
    parser.add_argument(
        '--test-data', 
        type=str, 
        default='data/test_samples.json',
        help='Path to test data JSON file'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='data/evaluation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--hf-repo', 
        type=str, 
        default='Salesforce/codegen-350M-mono',
        help='HuggingFace repository for model'
    )
    parser.add_argument(
        '--base-model', 
        type=str, 
        default='Salesforce/codegen-350M-mono',
        help='Base model name'
    )
    parser.add_argument(
        '--no-quantization', 
        action='store_true',
        help='Disable quantization (use full precision)'
    )
    parser.add_argument(
        '--no-plots', 
        action='store_true',
        help='Skip generating visualization plots'
    )
    parser.add_argument(
        '--quick-test', 
        action='store_true',
        help='Run quick test with sample code snippets'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SecretGuardEvaluator(
        hf_repo=args.hf_repo,
        base_model=args.base_model,
        use_quantization=not args.no_quantization
    )
    
    try:
        if args.quick_test:
            # Quick test mode
            print("Running quick test...")
            
            test_snippets = [
                'const password = "mySecretPassword123";',
                'const apiKey = "sk-1234567890abcdef";',
                'const userName = "john_doe";',
                'function calculateSum(a, b) { return a + b; }',
                'const awsKey = "AKIA1234567890EXAMPLE";'
            ]
            
            results = evaluator.quick_test(test_snippets)
            
            print("\nQuick Test Results:")
            print("-" * 60)
            for result in results:
                print(f"Code: {result['code'][:50]}...")
                print(f"Prediction: {result['prediction']}")
                print(f"Time: {result['inference_time']:.3f}s")
                print(f"Status: {result['status']}")
                print("-" * 60)
        
        else:
            # Full evaluation mode
            print(f"Running full evaluation...")
            print(f"Test data: {args.test_data}")
            print(f"Output directory: {args.output_dir}")
            print(f"HF Repository: {args.hf_repo}")
            print(f"Quantization: {not args.no_quantization}")
            
            # Check if test data exists
            if not Path(args.test_data).exists():
                print(f"Error: Test data file not found: {args.test_data}")
                print("Please create the test data file first.")
                return 1
            
            # Run evaluation
            results = evaluator.run_full_evaluation(
                test_data_path=args.test_data,
                output_dir=args.output_dir,
                save_detailed=True,
                generate_plots=not args.no_plots
            )
            
            print(f"\nEvaluation completed!")
            print(f"Results saved to: {args.output_dir}")
            print(f"Overall accuracy: {results['overall_metrics']['accuracy']:.3f}")
            print(f"F1-Score: {results['overall_metrics']['f1_score']:.3f}")
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error during evaluation: {e}")
        logging.exception("Evaluation failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())