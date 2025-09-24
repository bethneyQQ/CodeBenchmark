"""
DeepSeek API Model Implementation for lm-evaluation-harness

This module provides integration with DeepSeek's API for text generation tasks.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import retry_on_specific_exceptions


eval_logger = logging.getLogger(__name__)


def deepseek_completion(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop: List[str],
    **kwargs: Any,
) -> str:
    """Wrapper function around the DeepSeek API with retry handling.

    Args:
        api_key: DeepSeek API key
        model: str - DeepSeek model name (e.g., 'deepseek-chat', 'deepseek-coder')
        prompt: str - Input prompt
        max_tokens: int - Maximum tokens to generate
        temperature: float - Sampling temperature
        stop: List[str] - Stop sequences
        **kwargs: Additional arguments

    Returns:
        Generated text completion
    """
    try:
        import openai  # DeepSeek uses OpenAI-compatible API
    except ModuleNotFoundError as exception:
        raise type(exception)(
            "attempted to use 'deepseek' LM type, but package `openai` is not installed. "
            "Please install openai via `pip install openai`"
        )

    def _completion_helper():
        try:
            eval_logger.debug(f"Initializing OpenAI client for model: {model}")
            eval_logger.debug(f"Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1")
            
            # Initialize OpenAI client for DeepSeek API
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            
            # Prepare messages for chat completion
            messages = [{"role": "user", "content": prompt}]
            
            # Prepare generation parameters
            gen_params = {
                'model': model,
                'messages': messages,
                'max_tokens': max_tokens,
                'temperature': temperature,
            }
            
            if stop:
                gen_params['stop'] = stop
                
            # Special handling for DeepSeek-V3.1 thinking mode
            if 'enable_thinking' in kwargs:
                gen_params['extra_body'] = {'enable_thinking': kwargs.pop('enable_thinking')}
            
            # Add any additional parameters
            gen_params.update(kwargs)
            
            eval_logger.debug(f"Calling DeepSeek API with params: {gen_params}")
            response = client.chat.completions.create(**gen_params)
            eval_logger.debug(f"Received response: {response}")
            
            if response.choices and response.choices[0].message:
                return response.choices[0].message.content
            else:
                eval_logger.error("DeepSeek API returned empty response")
                raise RuntimeError("DeepSeek API returned empty response")
                
        except openai.APIError as e:
            eval_logger.error(f"DeepSeek API error: {e}")
            raise e
        except openai.RateLimitError as e:
            eval_logger.error(f"DeepSeek API rate limit exceeded: {e}")
            raise e
        except openai.AuthenticationError as e:
            eval_logger.error(f"DeepSeek API authentication error: {e}. Check your DASHSCOPE_API_KEY.")
            raise e
        except Exception as e:
            eval_logger.error(f"Unexpected error calling DeepSeek API: {e}")
            raise e

    # Apply retry logic with exponential backoff
    @retry_on_specific_exceptions(
        on_exceptions=[Exception],  # Catch all exceptions for now
        max_retries=3,
        backoff_time=1.0,
        backoff_multiplier=2.0,
    )
    def _retry_completion():
        return _completion_helper()
    
    return _retry_completion()


@register_model("deepseek")
class DeepSeekLM(LM):
    """Implementation of DeepSeek API model for lm-eval."""

    def __init__(
        self,
        model: str = "deepseek-v3.1",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.6,
        batch_size: int = 1,
        **kwargs,
    ):
        """Initialize DeepSeek model.

        Args:
            model: DeepSeek model name (e.g., 'deepseek-v3.1', 'deepseek-v3', 'deepseek-r1')
            api_key: DeepSeek API key (can also be set via DASHSCOPE_API_KEY env var)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (default 0.6 as per API docs)
            batch_size: Batch size for requests
            **kwargs: Additional arguments (e.g., enable_thinking=True for v3.1)
        """
        super().__init__()
        
        # Set API key
        if api_key:
            self.api_key = api_key
        elif "DASHSCOPE_API_KEY" in os.environ:
            self.api_key = os.environ["DASHSCOPE_API_KEY"]
        elif "DEEPSEEK_API_KEY" in os.environ:
            # Backward compatibility
            self.api_key = os.environ["DEEPSEEK_API_KEY"]
        else:
            raise ValueError(
                "DeepSeek API key must be provided either through 'api_key' parameter "
                "or 'DASHSCOPE_API_KEY' environment variable. "
                "Get your API key from: https://bailian.console.alibabacloud.com/"
            )
        
        # Validate model name
        valid_models = [
            'deepseek-v3.1', 'deepseek-v3', 'deepseek-r1', 'deepseek-r1-0528',
            'deepseek-chat', 'deepseek-coder'  # Legacy names for backward compatibility
        ]
        if model not in valid_models:
            eval_logger.warning(f"Model '{model}' not in known models {valid_models}. Proceeding anyway...")
            
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.extra_kwargs = kwargs
        
        eval_logger.info(f"Initialized DeepSeek model: {self.model}")
        if 'enable_thinking' in kwargs:
            eval_logger.info(f"Thinking mode enabled: {kwargs['enable_thinking']}")

    @property
    def max_length(self) -> int:
        """Return maximum context length for the model."""
        # DeepSeek models typically support long contexts
        return 32768

    @property
    def max_gen_toks(self) -> int:
        """Return maximum generation tokens."""
        return self.max_tokens

    @property
    def batch_size_per_gpu(self) -> int:
        """Return batch size per GPU."""
        return self.batch_size

    @property
    def device(self) -> str:
        """Return device (API-based, so no specific device)."""
        return "api"

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        """Encode a string into tokens.
        
        Note: DeepSeek API doesn't expose tokenization, so we use approximation.
        """
        # Rough approximation: ~4 characters per token
        return list(range(len(string) // 4 + 1))

    def tok_decode(self, tokens: List[int], **kwargs) -> str:
        """Decode tokens into a string.
        
        Note: DeepSeek API doesn't expose tokenization, so we use approximation.
        """
        return " ".join([f"token_{i}" for i in tokens])

    def generate_until(
        self, requests: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate completions for the given requests."""
        if not requests:
            return []
        
        results = []
        
        # Process requests in batches
        for i in tqdm(range(0, len(requests), self.batch_size), desc="DeepSeek Generation"):
            batch = requests[i : i + self.batch_size]
            batch_results = []
            
            for req in batch:
                # Extract context and generation kwargs from request
                # Handle both Instance objects (from lm_eval) and dict formats
                if hasattr(req, 'args'):
                    # Instance object from lm-eval
                    context = req.args[0]
                    gen_kwargs = req.args[1] if len(req.args) > 1 else {}
                else:
                    # Dictionary format
                    context = req.get("context", "")
                    gen_kwargs = req.get("gen_kwargs", {})
                
                # Extract generation parameters
                max_tokens = gen_kwargs.get("max_gen_toks", self.max_tokens)
                temperature = gen_kwargs.get("temperature", self.temperature)
                until = gen_kwargs.get("until", [])
                
                # Handle stop sequences
                stop_sequences = []
                if until:
                    if isinstance(until, str):
                        stop_sequences = [until]
                    elif isinstance(until, list):
                        stop_sequences = until
                
                try:
                    # Call DeepSeek API with extra kwargs
                    # Also remove parameters not supported by DeepSeek API
                    unsupported_params = ['max_gen_toks', 'temperature', 'until', 'do_sample', 
                                         'max_batch_size', 'device', 'dtype', 'quantization',
                                         'load_in_8bit', 'load_in_4bit', 'trust_remote_code',
                                         'use_accelerate', 'low_cpu_mem_usage', 'use_cache']
                    
                    # Clean extra_kwargs from unsupported parameters
                    api_kwargs = {k: v for k, v in self.extra_kwargs.items() 
                                 if k not in unsupported_params}
                    
                    # Remove conflicting parameters from gen_kwargs to avoid duplication
                    gen_kwargs_clean = {k: v for k, v in gen_kwargs.items() 
                                      if k not in unsupported_params}
                    api_kwargs.update(gen_kwargs_clean)
                    
                    response = deepseek_completion(
                        api_key=self.api_key,
                        model=self.model,
                        prompt=context,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=stop_sequences,
                        **api_kwargs
                    )
                    # Ensure we append a string, not None
                    if response is not None:
                        batch_results.append(response)
                    else:
                        batch_results.append("")
                    
                except Exception as e:
                    eval_logger.error(f"Error generating completion: {e}")
                    batch_results.append("")  # Return empty string on error
            
            results.extend(batch_results)
        
        return results

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """Compute log-likelihood of completions.
        
        Note: DeepSeek API doesn't directly support logprobs computation,
        so this method raises NotImplementedError.
        """
        raise NotImplementedError(
            "DeepSeek API does not support loglikelihood computation. "
            "This model can only be used for generation tasks."
        )

    def loglikelihood_rolling(self, requests) -> List[float]:
        """Compute rolling log-likelihood.
        
        Note: DeepSeek API doesn't support logprobs computation,
        so this method raises NotImplementedError.
        """
        raise NotImplementedError(
            "DeepSeek API does not support loglikelihood computation. "
            "This model can only be used for generation tasks."
        )

    @classmethod
    def create_from_arg_string(cls, arg_string: str, additional_config: Optional[Dict] = None) -> "DeepSeekLM":
        """Create model instance from argument string.
        
        Args:
            arg_string: Comma-separated string of key=value pairs
            additional_config: Additional configuration dictionary
            
        Returns:
            DeepSeekLM instance
        """
        args = {}
        if arg_string:
            for arg in arg_string.split(","):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert value types
                    if key in ["max_tokens", "batch_size"]:
                        args[key] = int(value)
                    elif key == "temperature":
                        args[key] = float(value)
                    else:
                        args[key] = value
        
        if additional_config:
            args.update(additional_config)
            
        return cls(**args)