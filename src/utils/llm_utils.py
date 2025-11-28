"""
LLM API utilities for Gemini and local Qwen inference.
"""

import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class GeminiAPI:
    """
    Wrapper for Google Gemini API with retry logic and error handling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Gemini API client.
        
        Args:
            config: Configuration dictionary with API settings
        """
        self.api_key = config.get('api_key')
        self.model_name = config.get('model', 'gemini-1.5-pro')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2048)
        self.timeout = config.get('timeout', 60)
        self.retry_attempts = config.get('retry_attempts', 3)
        
        # Initialize Gemini client
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
            self.genai = genai
            logger.info(f"Gemini API initialized with model: {self.model_name}")
        except ImportError:
            logger.error("google-generativeai not installed. Run: pip install google-generativeai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            raise
        
        # Track API usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text using Gemini API with retry logic.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens to generate (overrides default)
            stop_sequences: Optional stop sequences
            
        Returns:
            Generated text
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        generation_config = self.genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            stop_sequences=stop_sequences,
        )
        
        for attempt in range(self.retry_attempts):
            try:
                response = self.client.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                
                # Extract text
                if response.text:
                    output = response.text
                else:
                    logger.warning("Empty response from Gemini API")
                    output = ""
                
                # Track usage (approximate since Gemini doesn't provide exact token counts)
                self.total_input_tokens += len(prompt.split()) * 1.3  # Rough estimate
                self.total_output_tokens += len(output.split()) * 1.3
                self.total_calls += 1
                
                logger.debug(f"Gemini API call successful (attempt {attempt + 1})")
                return output
                
            except Exception as e:
                logger.warning(f"Gemini API call failed (attempt {attempt + 1}/{self.retry_attempts}): {e}")
                
                if attempt < self.retry_attempts - 1:
                    # Exponential backoff
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"All retry attempts failed for Gemini API")
                    raise
    
    def generate_json(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate JSON output using Gemini API.
        
        Args:
            prompt: Input prompt (should request JSON output)
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Parsed JSON as dictionary
        """
        # Add JSON formatting instruction
        json_prompt = f"{prompt}\n\nOutput valid JSON only."
        
        output = self.generate(json_prompt, temperature, max_tokens)
        
        # Try to parse JSON
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in output:
                output = output.split("```json")[1].split("```")[0].strip()
            elif "```" in output:
                output = output.split("```")[1].split("```")[0].strip()
            
            return json.loads(output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemini output: {e}")
            logger.error(f"Raw output: {output}")
            raise ValueError(f"Invalid JSON response from Gemini: {output}")
    
    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get API usage statistics.
        
        Returns:
            Dictionary with usage stats
        """
        return {
            'total_calls': self.total_calls,
            'total_input_tokens': int(self.total_input_tokens),
            'total_output_tokens': int(self.total_output_tokens),
            'total_tokens': int(self.total_input_tokens + self.total_output_tokens),
        }


class QwenLocal:
    """
    Local inference with fine-tuned Qwen models.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize local Qwen model.
        
        Args:
            model_path: Path to fine-tuned model or HuggingFace model ID
            device: Device to load model on ('auto', 'cuda', 'cpu')
            load_in_8bit: Whether to load in 8-bit precision
            load_in_4bit: Whether to load in 4-bit precision
        """
        self.model_path = model_path
        self.device = device
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading Qwen model from: {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            
            # Load model
            load_kwargs = {
                'trust_remote_code': True,
                'device_map': device,
            }
            
            if load_in_8bit:
                load_kwargs['load_in_8bit'] = True
            elif load_in_4bit:
                load_kwargs['load_in_4bit'] = True
            else:
                load_kwargs['torch_dtype'] = torch.bfloat16
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs
            )
            
            self.model.eval()
            
            logger.info("Qwen model loaded successfully")
            
        except ImportError:
            logger.error("transformers not installed. Run: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text using local Qwen model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        import torch
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if self.device != "cpu":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        
        return generated_text
    
    def generate_json(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Generate JSON output using local Qwen model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens
            temperature: Sampling temperature
            
        Returns:
            Parsed JSON dictionary
        """
        json_prompt = f"{prompt}\n\nOutput valid JSON only."
        
        output = self.generate(json_prompt, max_new_tokens, temperature)
        
        # Parse JSON
        try:
            if "```json" in output:
                output = output.split("```json")[1].split("```")[0].strip()
            elif "```" in output:
                output = output.split("```")[1].split("```")[0].strip()
            
            return json.loads(output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Qwen output: {e}")
            logger.error(f"Raw output: {output}")
            raise ValueError(f"Invalid JSON response from Qwen: {output}")


def create_llm_client(provider: str, config: Dict[str, Any]):
    """
    Factory function to create appropriate LLM client.
    
    Args:
        provider: 'gemini' or 'qwen'
        config: Configuration dictionary
        
    Returns:
        LLM client instance
    """
    if provider == "gemini":
        return GeminiAPI(config)
    elif provider == "qwen":
        return QwenLocal(**config)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


if __name__ == "__main__":
    # Test Gemini API
    from config import load_config
    
    config = load_config()
    gemini = GeminiAPI(config.api_keys.gemini.to_dict())
    
    response = gemini.generate("Write a short poem about AI tutoring.")
    print("Gemini response:")
    print(response)
    print("\nUsage stats:", gemini.get_usage_stats())