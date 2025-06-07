"""
Model loader for SecretGuard evaluation
Optimized for MacBook Air M3 chip
"""
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
import logging
from typing import Optional, Tuple
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecretGuardModelLoader:
    def __init__(self, 
                 hf_repo: str = "Salesforce/codegen-350M-mono",
                 base_model: str = "Salesforce/codegen-350M-mono",
                 use_quantization: bool = True):
        """
        Initialize the model loader
        
        Args:
            hf_repo: HuggingFace repository for the model
            base_model: Base model name
            use_quantization: Whether to use 4-bit quantization (recommended for M3)
        """
        self.hf_repo = hf_repo
        self.base_model = base_model
        self.use_quantization = use_quantization
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """Get the appropriate device for M3 MacBook Air"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the model and tokenizer
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model on device: {self.device}")
        
        try:
            # Load tokenizer
            logger.info(f"Loading tokenizer from: {self.hf_repo}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.hf_repo,
                trust_remote_code=True
            )
            
            # Configure tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            
            # Load model
            logger.info(f"Loading model from: {self.hf_repo}")
            model = AutoModelForCausalLM.from_pretrained(
                self.hf_repo,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                trust_remote_code=True
            )
            
            # Move to appropriate device
            model = model.to(self.device)
            model.eval()
            
            # Store references
            self.model = model
            self.tokenizer = tokenizer
            
            logger.info("Model loaded successfully!")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def build_prompt(self, code_snippet: str) -> str:
        """
        Build the prompt for the model
        
        Args:
            code_snippet: Code to analyze
            
        Returns:
            Formatted prompt string
        """
        return f"""Analyze if this code contains secrets or sensitive information.
Code:
{code_snippet}

Question: Does this code contain any secrets or sensitive information?
Answer with exactly one word: Yes or No.

Answer:"""
    
    def classify(self, code_snippet: str) -> str:
        """
        Classify a code snippet
        
        Args:
            code_snippet: Code to analyze
            
        Returns:
            Classification result
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        prompt = self.build_prompt(code_snippet)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,  # Limit to short response
                    do_sample=False,
                    temperature=0.0,
                    num_beams=5,  # Use beam search
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = decoded.split("Answer:")[-1].strip().lower()
            
            # Clean and normalize the response
            if result.startswith('yes'):
                return "Yes"
            elif result.startswith('no'):
                return "No"
            else:
                # Default to No for ambiguous cases (conservative approach)
                logger.warning(f"Ambiguous response: {result}. Defaulting to No.")
                return "No"
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return "No"  # Default to No on error
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "device": str(self.model.device),
            "hf_repo": self.hf_repo,
            "base_model": self.base_model,
            "quantization": self.use_quantization,
            "model_type": type(self.model).__name__
        }