"""
Utility functions for SecretGuard evaluation
"""
import os
import json
import time
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for performance monitoring
    
    Returns:
        Dictionary with system information
    """
    try:
        import platform
        import torch
        
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'torch_version': torch.__version__,
        }
        
        # Check for Apple Silicon
        if 'arm64' in platform.machine() or 'M1' in platform.processor() or 'M2' in platform.processor() or 'M3' in platform.processor():
            info['apple_silicon'] = True
            info['mps_available'] = torch.backends.mps.is_available()
        else:
            info['apple_silicon'] = False
            
        # CUDA info
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
            
        return info
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {'error': str(e)}

def monitor_performance(func):
    """
    Decorator to monitor function performance
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function with performance monitoring
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**3)
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / (1024**3)
            
            performance_info = {
                'execution_time': end_time - start_time,
                'memory_used_gb': end_memory - start_memory,
                'peak_memory_gb': end_memory,
                'function_name': func.__name__
            }
            
            logger.info(f"Performance: {func.__name__} took {performance_info['execution_time']:.2f}s, "
                       f"memory delta: {performance_info['memory_used_gb']:.2f}GB")
            
            if hasattr(result, '__dict__'):
                result.performance_info = performance_info
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
            
    return wrapper

def create_evaluation_config(
    hf_repo: str = "asudarshan/secretguard-starcoderbase-lora2",
    base_model: str = "bigcode/starcoderbase",
    use_quantization: bool = True,
    max_length: int = 512,
    max_new_tokens: int = 10,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Create evaluation configuration
    
    Args:
        hf_repo: HuggingFace repository
        base_model: Base model name
        use_quantization: Whether to use quantization
        max_length: Maximum input length
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Configuration dictionary
    """
    config = {
        'model': {
            'hf_repo': hf_repo,
            'base_model': base_model,
            'use_quantization': use_quantization
        },
        'generation': {
            'max_length': max_length,
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'do_sample': temperature > 0.0
        },
        'evaluation': {
            'batch_size': 1,
            'save_detailed_results': True,
            'generate_plots': True
        },
        'system': get_system_info(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return config

def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
            
        logger.info(f"Configuration saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise

def validate_test_data(test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate test data format and content
    
    Args:
        test_data: List of test cases
        
    Returns:
        Validation report
    """
    report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    required_fields = ['id', 'code', 'expected']
    optional_fields = ['category', 'description']
    
    categories = set()
    expected_values = set()
    
    for i, test_case in enumerate(test_data):
        # Check required fields
        for field in required_fields:
            if field not in test_case:
                report['errors'].append(f"Test case {i}: Missing required field '{field}'")
                report['valid'] = False
        
        # Check expected values
        if 'expected' in test_case:
            expected = test_case['expected'].lower()
            expected_values.add(expected)
            if expected not in ['yes', 'no', 'true', 'false', '1', '0']:
                report['warnings'].append(f"Test case {i}: Unusual expected value '{test_case['expected']}'")
        
        # Check code content
        if 'code' in test_case:
            if len(test_case['code']) == 0:
                report['errors'].append(f"Test case {i}: Empty code field")
                report['valid'] = False
            elif len(test_case['code']) > 2000:
                report['warnings'].append(f"Test case {i}: Very long code snippet ({len(test_case['code'])} chars)")
        
        # Collect categories
        if 'category' in test_case:
            categories.add(test_case['category'])
    
    # Statistics
    report['statistics'] = {
        'total_cases': len(test_data),
        'categories': list(categories),
        'category_count': len(categories),
        'expected_values': list(expected_values),
        'positive_cases': sum(1 for case in test_data if case.get('expected', '').lower() in ['yes', 'true', '1']),
        'negative_cases': sum(1 for case in test_data if case.get('expected', '').lower() in ['no', 'false', '0'])
    }
    
    return report

def format_results_table(results: Dict[str, Any]) -> str:
    """
    Format evaluation results as a readable table
    
    Args:
        results: Evaluation results
        
    Returns:
        Formatted table string
    """
    lines = []
    lines.append("="*80)
    lines.append("SECRETGUARD EVALUATION RESULTS")
    lines.append("="*80)
    
    # Overall metrics
    if 'overall_metrics' in results:
        lines.append("\nOVERALL PERFORMANCE:")
        lines.append("-" * 40)
        metrics = results['overall_metrics']
        for metric, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"{metric.replace('_', ' ').title():20}: {value:.3f}")
            else:
                lines.append(f"{metric.replace('_', ' ').title():20}: {value}")
    
    # Category metrics
    if 'category_metrics' in results and results['category_metrics']:
        lines.append("\nPERFORMANCE BY CATEGORY:")
        lines.append("-" * 40)
        lines.append(f"{'Category':<20} | {'F1':<6} | {'Precision':<9} | {'Recall':<6} | {'Samples':<7}")
        lines.append("-" * 60)
        
        for category, metrics in results['category_metrics'].items():
            lines.append(
                f"{category:<20} | "
                f"{metrics.get('f1_score', 0):.3f} | "
                f"{metrics.get('precision', 0):.3f}     | "
                f"{metrics.get('recall', 0):.3f} | "
                f"{metrics.get('sample_count', 0):<7}"
            )
    
    # Performance info
    if 'performance' in results:
        lines.append("\nPERFORMANCE METRICS:")
        lines.append("-" * 40)
        perf = results['performance']
        lines.append(f"Total Inference Time:    {perf.get('total_inference_time', 0):.2f}s")
        lines.append(f"Average per Prediction:  {perf.get('average_inference_time', 0):.3f}s")
        lines.append(f"Predictions per Second:  {perf.get('predictions_per_second', 0):.1f}")
        lines.append(f"Error Rate:              {perf.get('error_rate', 0):.1%}")
    
    # Summary
    if 'summary' in results:
        lines.append("\nSUMMARY:")
        lines.append("-" * 40)
        summary = results['summary']
        lines.append(f"Total Test Cases:        {summary.get('total_samples', 0)}")
        lines.append(f"Correct Predictions:     {summary.get('correct_predictions', 0)}")
        lines.append(f"Categories Tested:       {summary.get('categories_tested', 0)}")
        lines.append(f"Overall Accuracy:        {summary.get('accuracy', 0):.1%}")
    
    lines.append("\n" + "="*80)
    
    return "\n".join(lines)

def cleanup_temp_files(directory: str, pattern: str = "*.tmp") -> int:
    """
    Clean up temporary files
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match
        
    Returns:
        Number of files cleaned
    """
    try:
        path = Path(directory)
        if not path.exists():
            return 0
        
        files_removed = 0
        for file in path.glob(pattern):
            if file.is_file():
                file.unlink()
                files_removed += 1
        
        logger.info(f"Cleaned up {files_removed} temporary files")
        return files_removed
        
    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")
        return 0

def estimate_memory_requirements(
    model_name: str,
    use_quantization: bool = True,
    batch_size: int = 1
) -> Dict[str, float]:
    """
    Estimate memory requirements for model
    
    Args:
        model_name: Name of the model
        use_quantization: Whether quantization is used
        batch_size: Batch size
        
    Returns:
        Memory requirement estimates in GB
    """
    # Rough estimates for StarCoder models
    base_params = {
        'bigcode/starcoderbase': 15.5,  # ~15.5B parameters
        'bigcode/starcoder': 15.5,
    }
    
    params = base_params.get(model_name, 15.5)  # Default to StarCoder size
    
    # Memory estimates
    if use_quantization:
        # 4-bit quantization: ~0.5 bytes per parameter
        model_memory = params * 0.5
    else:
        # FP16: 2 bytes per parameter
        model_memory = params * 2
    
    # Add overhead for activations, gradients, etc.
    overhead = model_memory * 0.3
    
    # Batch size impact (minimal for inference)
    batch_overhead = batch_size * 0.1
    
    estimates = {
        'model_memory_gb': model_memory,
        'overhead_gb': overhead,
        'batch_overhead_gb': batch_overhead,
        'total_estimated_gb': model_memory + overhead + batch_overhead,
        'recommended_system_memory_gb': (model_memory + overhead + batch_overhead) * 1.5
    }
    
    return estimates