import torch
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SecretGuardEvaluator:
    def __init__(self, hf_repo: str, base_model: str):
        self.hf_repo = hf_repo
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer"""
        print("Loading model...")
        
        # Setup quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_repo)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, self.hf_repo)
        self.model.eval()
        print("Model loaded successfully!")
    
    def build_prompt(self, code_snippet: str) -> str:
        """Build prompt for the model"""
        return f"""<|user|>
Does this code contain a secret?
<|code|>
{code_snippet}
<|assistant|>
"""
    
    def classify(self, code_snippet: str) -> str:
        """Classify a single code snippet"""
        prompt = self.build_prompt(code_snippet)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = decoded.split("<|assistant|>")[-1].strip()
        return result
    
    def parse_prediction(self, prediction: str) -> int:
        """Parse model prediction to binary classification (1 for secret, 0 for no secret)"""
        prediction = prediction.lower().strip()
        
        # Define positive indicators for secrets
        positive_indicators = ['yes', 'true', 'contains', 'secret', '1']
        negative_indicators = ['no', 'false', 'does not', "doesn't", 'clean', '0']
        
        # Check for positive indicators first
        for indicator in positive_indicators:
            if indicator in prediction:
                return 1
        
        # Check for negative indicators
        for indicator in negative_indicators:
            if indicator in prediction:
                return 0
        
        # Default to 0 if unclear
        return 0
    
    def evaluate_dataset(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Evaluate the model on a test dataset"""
        predictions = []
        true_labels = []
        raw_predictions = []
        
        print("Evaluating model on test dataset...")
        for item in tqdm(test_data):
            code = item['code']
            true_label = item['label']  # 1 for secret, 0 for no secret
            
            # Get model prediction
            raw_pred = self.classify(code)
            parsed_pred = self.parse_prediction(raw_pred)
            
            predictions.append(parsed_pred)
            true_labels.append(true_label)
            raw_predictions.append(raw_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Classification report
        report = classification_report(true_labels, predictions, 
                                     target_names=['No Secret', 'Secret'], 
                                     output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': predictions,
            'true_labels': true_labels,
            'raw_predictions': raw_predictions
        }
        
        return results
    
    def print_evaluation_results(self, results: Dict[str, Any]):
        """Print evaluation results in a formatted way"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1_score']:.4f}")
        
        print("\nConfusion Matrix:")
        print("                Predicted")
        print("                No Secret  Secret")
        print(f"Actual No Secret    {results['confusion_matrix'][0][0]:4d}      {results['confusion_matrix'][0][1]:4d}")
        print(f"Actual Secret       {results['confusion_matrix'][1][0]:4d}      {results['confusion_matrix'][1][1]:4d}")
        
        print("\nDetailed Classification Report:")
        report = results['classification_report']
        for class_name in ['No Secret', 'Secret']:
            if class_name in report:
                metrics = report[class_name]
                print(f"{class_name:12} - Precision: {metrics['precision']:.4f}, "
                      f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1-score']:.4f}")
    
    def plot_confusion_matrix(self, results: Dict[str, Any], save_path: str = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=['No Secret', 'Secret'],
                   yticklabels=['No Secret', 'Secret'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_errors(self, test_data: List[Dict], results: Dict[str, Any]):
        """Analyze prediction errors"""
        predictions = results['predictions']
        true_labels = results['true_labels']
        raw_predictions = results['raw_predictions']
        
        errors = []
        for i, (pred, true_label, raw_pred) in enumerate(zip(predictions, true_labels, raw_predictions)):
            if pred != true_label:
                errors.append({
                    'index': i,
                    'code': test_data[i]['code'],
                    'true_label': true_label,
                    'predicted_label': pred,
                    'raw_prediction': raw_pred,
                    'error_type': 'False Positive' if pred == 1 else 'False Negative'
                })
        
        print(f"\nError Analysis: {len(errors)} errors out of {len(test_data)} samples")
        print("="*50)
        
        # Group errors by type
        false_positives = [e for e in errors if e['error_type'] == 'False Positive']
        false_negatives = [e for e in errors if e['error_type'] == 'False Negative']
        
        print(f"False Positives: {len(false_positives)}")
        print(f"False Negatives: {len(false_negatives)}")
        
        # Show some examples
        if false_positives:
            print("\nFalse Positive Examples:")
            for i, error in enumerate(false_positives[:3]):  # Show first 3
                print(f"\nExample {i+1}:")
                print(f"Code: {error['code']}")
                print(f"Model said: {error['raw_prediction']}")
        
        if false_negatives:
            print("\nFalse Negative Examples:")
            for i, error in enumerate(false_negatives[:3]):  # Show first 3
                print(f"\nExample {i+1}:")
                print(f"Code: {error['code']}")
                print(f"Model said: {error['raw_prediction']}")
        
        return errors

def create_test_dataset() -> List[Dict]:
    """Create a comprehensive test dataset for secret detection"""
    
    # Test cases with secrets (label = 1)
    secret_examples = [
        # API Keys
        {
            'code': 'const apiKey = "AKIA1234567890EXAMPLE";',
            'label': 1,
            'description': 'AWS API Key'
        },
        {
            'code': 'API_KEY = "sk-1234567890abcdef1234567890abcdef12345678"',
            'label': 1,
            'description': 'OpenAI API Key'
        },
        {
            'code': 'github_token = "ghp_1234567890abcdef1234567890abcdef12345678"',
            'label': 1,
            'description': 'GitHub Personal Access Token'
        },
        {
            'code': 'STRIPE_SECRET_KEY = "sk_live_1234567890abcdef1234567890abcdef"',
            'label': 1,
            'description': 'Stripe Secret Key'
        },
        
        # Database Credentials
        {
            'code': 'password = "MySecretPassword123!"',
            'label': 1,
            'description': 'Database Password'
        },
        {
            'code': 'DB_CONNECTION = "postgresql://user:password123@localhost:5432/mydb"',
            'label': 1,
            'description': 'Database Connection String'
        },
        {
            'code': 'mysql_config = {"user": "admin", "password": "secret123", "host": "localhost"}',
            'label': 1,
            'description': 'MySQL Configuration'
        },
        
        # SSH Keys
        {
            'code': 'private_key = "-----BEGIN RSA PRIVATE KEY-----\\nMIIEpAIBAAKCAQEA..."',
            'label': 1,
            'description': 'SSH Private Key'
        },
        
        # JWT Tokens
        {
            'code': 'jwt_secret = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ"',
            'label': 1,
            'description': 'JWT Token'
        },
        
        # OAuth Tokens
        {
            'code': 'access_token = "ya29.1234567890abcdef1234567890abcdef12345678"',
            'label': 1,
            'description': 'OAuth Access Token'
        },
        
        # Webhooks and URLs with secrets
        {
            'code': 'webhook_url = "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"',
            'label': 1,
            'description': 'Slack Webhook URL'
        },
        
        # Credit Card Numbers
        {
            'code': 'credit_card = "4532-1234-5678-9012"',
            'label': 1,
            'description': 'Credit Card Number'
        },
        
        # Encryption Keys
        {
            'code': 'encryption_key = "a1b2c3d4e5f6789012345678901234567890abcdef"',
            'label': 1,
            'description': 'Encryption Key'
        },
        
        # Mixed with code logic
        {
            'code': '''
def connect_to_api():
    api_key = "sk-proj-1234567890abcdef"
    headers = {"Authorization": f"Bearer {api_key}"}
    return requests.get("https://api.example.com", headers=headers)
''',
            'label': 1,
            'description': 'API Key in Function'
        },
        
        # Environment-like but hardcoded
        {
            'code': 'SECRET_KEY = "django-insecure-1234567890abcdef"',
            'label': 1,
            'description': 'Django Secret Key'
        }
    ]
    
    # Test cases without secrets (label = 0)
    clean_examples = [
        # Environment Variables (proper way)
        {
            'code': 'api_key = os.getenv("API_KEY")',
            'label': 0,
            'description': 'Environment Variable Usage'
        },
        {
            'code': 'password = os.environ.get("PASSWORD")',
            'label': 0,
            'description': 'Environment Variable Access'
        },
        {
            'code': 'config = {"api_key": os.getenv("API_KEY", "default")}',
            'label': 0,
            'description': 'Config with Environment Variable'
        },
        
        # Placeholders and Examples
        {
            'code': 'api_key = "your-api-key-here"',
            'label': 0,
            'description': 'Placeholder Text'
        },
        {
            'code': 'password = "example_password"',
            'label': 0,
            'description': 'Example Placeholder'
        },
        {
            'code': 'secret = "TODO: add your secret here"',
            'label': 0,
            'description': 'TODO Placeholder'
        },
        
        # Mock/Test Data
        {
            'code': 'mock_api_key = "test_key_123"',
            'label': 0,
            'description': 'Mock API Key'
        },
        {
            'code': 'test_password = "password123"  # This is just for testing',
            'label': 0,
            'description': 'Test Password with Comment'
        },
        
        # Regular Variables
        {
            'code': 'username = "john_doe"',
            'label': 0,
            'description': 'Username (not secret)'
        },
        {
            'code': 'app_name = "MyApplication"',
            'label': 0,
            'description': 'Application Name'
        },
        {
            'code': 'version = "1.2.3"',
            'label': 0,
            'description': 'Version String'
        },
        
        # URLs without secrets
        {
            'code': 'api_url = "https://api.example.com/v1"',
            'label': 0,
            'description': 'API URL without secrets'
        },
        {
            'code': 'database_host = "localhost"',
            'label': 0,
            'description': 'Database Host'
        },
        
        # Configuration files
        {
            'code': 'config = {"timeout": 30, "retries": 3}',
            'label': 0,
            'description': 'Configuration Settings'
        },
        
        # File paths
        {
            'code': 'log_file = "/var/log/application.log"',
            'label': 0,
            'description': 'File Path'
        },
        
        # Constants
        {
            'code': 'MAX_RETRIES = 5',
            'label': 0,
            'description': 'Constant Value'
        },
        
        # Function definitions
        {
            'code': '''
def authenticate(api_key):
    # API key is passed as parameter
    return validate_key(api_key)
''',
            'label': 0,
            'description': 'Function with Parameter'
        },
        
        # Comments and documentation
        {
            'code': '# Set your API key in the environment variable API_KEY',
            'label': 0,
            'description': 'Documentation Comment'
        },
        
        # Empty or None values
        {
            'code': 'secret_key = None',
            'label': 0,
            'description': 'None Value'
        },
        {
            'code': 'api_key = ""',
            'label': 0,
            'description': 'Empty String'
        },
        
        # Template strings
        {
            'code': 'connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"',
            'label': 0,
            'description': 'Template String'
        }
    ]
    
    # Combine all examples
    all_examples = secret_examples + clean_examples
    
    # Shuffle the dataset
    import random
    random.shuffle(all_examples)
    
    return all_examples

def main():
    """Main evaluation function"""
    # Configuration
    HF_REPO = "asudarshan/secretguard-starcoderbase-lora2"
    BASE_MODEL = "bigcode/starcoderbase"
    
    # Create test dataset
    print("Creating test dataset...")
    test_data = create_test_dataset()
    print(f"Created test dataset with {len(test_data)} examples")
    print(f"Secrets: {sum(1 for item in test_data if item['label'] == 1)}")
    print(f"Clean code: {sum(1 for item in test_data if item['label'] == 0)}")
    
    # Initialize evaluator
    evaluator = SecretGuardEvaluator(HF_REPO, BASE_MODEL)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(test_data)
    
    # Print results
    evaluator.print_evaluation_results(results)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(results, save_path="confusion_matrix.png")
    
    # Analyze errors
    errors = evaluator.analyze_errors(test_data, results)
    
    # Save detailed results
    detailed_results = {
        'test_data': test_data,
        'results': {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'confusion_matrix': results['confusion_matrix'].tolist(),
        },
        'errors': errors
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nDetailed results saved to 'evaluation_results.json'")
    print("Evaluation completed!")

if __name__ == "__main__":
    main()