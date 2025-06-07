"""
Test suite for SecretGuard evaluation
"""
import pytest
import sys
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.metrics import EvaluationMetrics
from src.model_loader import SecretGuardModelLoader

class TestEvaluationMetrics:
    """Test cases for evaluation metrics"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.metrics = EvaluationMetrics()
        
    def test_normalize_predictions(self):
        """Test prediction normalization"""
        predictions = ["Yes", "No", "yes", "no", "True", "False", "Secret found", "Safe"]
        expected = [1, 0, 1, 0, 1, 0, 1, 0]
        
        result = self.metrics.normalize_predictions(predictions)
        assert result == expected
        
    def test_normalize_ground_truth(self):
        """Test ground truth normalization"""
        ground_truth = ["Yes", "No", "yes", "no", "1", "0"]
        expected = [1, 0, 1, 0, 1, 0]
        
        result = self.metrics.normalize_ground_truth(ground_truth)
        assert result == expected
        
    def test_basic_metrics_calculation(self):
        """Test basic metrics calculation"""
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1, 0]
        
        metrics = self.metrics.calculate_basic_metrics(y_true, y_pred)
        
        # Check that all expected metrics are present
        expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 
                        'specificity', 'false_positive_rate', 'false_negative_rate']
        
        for key in expected_keys:
            assert key in metrics
            assert 0 <= metrics[key] <= 1
            
    def test_generate_detailed_report(self):
        """Test detailed report generation"""
        test_data = [
            {
                "id": "test1",
                "code": "const password = 'secret';",
                "expected": "Yes",
                "category": "password"
            },
            {
                "id": "test2", 
                "code": "const user = 'john';",
                "expected": "No",
                "category": "safe"
            }
        ]
        
        predictions = ["Yes", "No"]
        
        report = self.metrics.generate_detailed_report(test_data, predictions)
        
        # Check report structure
        assert 'overall_metrics' in report
        assert 'category_metrics' in report
        assert 'detailed_results' in report
        assert 'summary' in report
        assert 'confusion_matrix' in report
        
        # Check summary
        assert report['summary']['total_samples'] == 2
        assert report['summary']['correct_predictions'] == 2
        assert report['summary']['accuracy'] == 1.0

class TestModelLoader:
    """Test cases for model loader"""
    
    def test_device_detection(self):
        """Test device detection for M3 Mac"""
        loader = SecretGuardModelLoader()
        device = loader._get_device()
        
        # Should detect MPS on M3 Mac, but fallback to CPU is acceptable
        assert device in ['mps', 'cuda', 'cpu']
        
    def test_prompt_building(self):
        """Test prompt building"""
        loader = SecretGuardModelLoader()
        code = "const password = 'test';"
        
        prompt = loader.build_prompt(code)
        
        assert "<|user|>" in prompt
        assert "<|code|>" in prompt
        assert "<|assistant|>" in prompt
        assert code in prompt
        
    def test_quantization_config(self):
        """Test quantization configuration"""
        loader = SecretGuardModelLoader(use_quantization=True)
        config = loader._setup_quantization_config()
        
        # Config might be None if bitsandbytes not available
        if config is not None:
            assert hasattr(config, 'load_in_4bit')
            assert config.load_in_4bit == True

class TestIntegration:
    """Integration tests"""
    
    def test_sample_data_loading(self):
        """Test loading sample test data"""
        # Create temporary test data
        test_data = {
            "test_cases": [
                {
                    "id": "test1",
                    "code": "const password = 'secret';",
                    "expected": "Yes",
                    "category": "password"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            # Test data should load without errors
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert 'test_cases' in loaded_data
            assert len(loaded_data['test_cases']) == 1
            assert loaded_data['test_cases'][0]['id'] == 'test1'
            
        finally:
            Path(temp_path).unlink()
            
    def test_metrics_with_sample_data(self):
        """Test metrics calculation with sample data"""
        test_data = [
            {"id": "1", "code": "password='secret'", "expected": "Yes", "category": "password"},
            {"id": "2", "code": "user='john'", "expected": "No", "category": "safe"},
            {"id": "3", "code": "api_key='sk-123'", "expected": "Yes", "category": "api_key"},
            {"id": "4", "code": "function test() {}", "expected": "No", "category": "safe"}
        ]
        
        predictions = ["Yes", "No", "Yes", "No"]
        
        metrics = EvaluationMetrics()
        report = metrics.generate_detailed_report(test_data, predictions)
        
        # Perfect predictions should give perfect scores
        assert report['overall_metrics']['accuracy'] == 1.0
        assert report['overall_metrics']['precision'] == 1.0
        assert report['overall_metrics']['recall'] == 1.0
        assert report['overall_metrics']['f1_score'] == 1.0

# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])