import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import retry_on_specific_exceptions


eval_logger = logging.getLogger(__name__)


def dashscope_completion(
    client,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop: List[str],
    **kwargs: Any,
) -> str:
    """Wrapper function around the DashScope SDK with retry handling.

    Args:
        client: dashscope client instance
        model: str - DashScope model name (e.g., 'qwen-turbo', 'qwen-plus')
        prompt: str - Input prompt
        max_tokens: int - Maximum tokens to generate
        temperature: float - Sampling temperature
        stop: List[str] - Stop sequences
        **kwargs: Additional arguments

    Returns:
        Generated text completion
    """
    try:
        import dashscope
        from dashscope import Generation
    except ModuleNotFoundError as exception:
        raise type(exception)(
            "attempted to use 'dashscope' LM type, but package `dashscope` is not installed. "
            "Please install dashscope via `pip install dashscope`"
        )

    # Prepare generation parameters
    gen_params = {
        'model': model,
        'prompt': prompt,
        'max_tokens': max_tokens,
        'temperature': temperature,
    }
    
    if stop:
        gen_params['stop'] = stop
    
    # Add any additional parameters
    gen_params.update(kwargs)
    
    def _completion_helper():
        try:
            response = Generation.call(**gen_params)
            if response.status_code == 200:
                return response.output.text
            else:
                eval_logger.error(f"DashScope API error: {response.code} - {response.message}")
                raise RuntimeError(f"DashScope API error: {response.code} - {response.message}")
        except Exception as e:
            eval_logger.error(f"Error calling DashScope API: {e}")
            raise e

    # Apply retry logic with exponential backoff
    @retry_on_specific_exceptions(
        on_exceptions=[Exception],  # Catch all exceptions for now, DashScope specific exceptions TBD
        max_retries=3,
        backoff_time=1.0,
        backoff_multiplier=2.0,
    )
    def _retry_completion():
        return _completion_helper()
    
    return _retry_completion()


@register_model("dashscope")
class DashScopeLM(LM):
    """Implementation of DashScope API model for lm-eval."""

    def __init__(
        self,
        model: str = "qwen-turbo",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        batch_size: int = 1,
        **kwargs,
    ):
        """Initialize DashScope model.

        Args:
            model: DashScope model name (e.g., 'qwen-turbo', 'qwen-plus', 'qwen-max')
            api_key: DashScope API key (can also be set via DASHSCOPE_API_KEY env var)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            batch_size: Batch size for requests
            **kwargs: Additional arguments
        """
        super().__init__()
        
        try:
            import dashscope
            from dashscope import Generation
        except ModuleNotFoundError as exception:
            raise type(exception)(
                "attempted to use 'dashscope' LM type, but package `dashscope` is not installed. "
                "Please install dashscope via `pip install dashscope`"
            )
        
        # Set API key
        if api_key:
            dashscope.api_key = api_key
        elif "DASHSCOPE_API_KEY" in os.environ:
            dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
        else:
            raise ValueError(
                "DashScope API key must be provided either through 'api_key' parameter "
                "or 'DASHSCOPE_API_KEY' environment variable"
            )
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.client = dashscope
        self.generation = Generation
        
        eval_logger.info(f"Initialized DashScope model: {self.model}")

    @property
    def eot_token_id(self):
        # DashScope doesn't expose tokenizer details directly
        return None

    @property
    def max_length(self):
        # Return a reasonable default - DashScope models typically support 2K-8K context
        return 8192

    @property
    def max_gen_toks(self):
        return self.max_tokens

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @property
    def device(self):
        return "api"

    def tok_encode(self, string: str) -> List[int]:
        """Tokenize string. 
        
        Note: DashScope doesn't expose tokenization directly,
        so we'll return a placeholder implementation.
        """
        # Rough approximation: ~4 characters per token for Chinese/English text
        return list(range(len(string) // 4 + 1))

    def tok_decode(self, tokens: List[int]) -> str:
        """Decode tokens to string.
        
        Note: DashScope doesn't expose tokenization directly,
        so we'll return a placeholder implementation.
        """
        return "".join([f"<tok_{t}>" for t in tokens])

    def generate_until(self, requests) -> List[str]:
        """Generate text completions for multiple requests."""
        if not requests:
            return []
            
        results = []
        
        # Process requests in batches
        for i in tqdm(
            range(0, len(requests), self.batch_size),
            desc="DashScope Generation",
            disable=len(requests) < 10
        ):
            batch = requests[i : i + self.batch_size]
            batch_results = []
            
            for req in batch:
                context = req.args[0]
                gen_kwargs = req.args[1] if len(req.args) > 1 else {}
                
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
                    # Call DashScope API
                    response = dashscope_completion(
                        client=self.client,
                        model=self.model,
                        prompt=context,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=stop_sequences,
                    )
                    batch_results.append(response)
                    
                except Exception as e:
                    eval_logger.error(f"Error generating completion: {e}")
                    batch_results.append("")  # Return empty string on error
            
            results.extend(batch_results)
        
        return results

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """Compute log-likelihood of completions.
        
        Note: DashScope API doesn't directly support logprobs computation,
        so this method raises NotImplementedError.
        """
        raise NotImplementedError(
            "DashScope API does not support loglikelihood computation. "
            "This model can only be used for generation tasks."
        )

    def loglikelihood_rolling(self, requests) -> List[float]:
        """Compute rolling log-likelihood.
        
        Note: DashScope API doesn't directly support logprobs computation,
        so this method raises NotImplementedError.
        """
        raise NotImplementedError(
            "DashScope API does not support rolling loglikelihood computation. "
            "This model can only be used for generation tasks."
        )

    @classmethod
    def create_from_arg_string(cls, arg_string: Union[str, dict], additional_config: Optional[dict] = None):
        """Create model instance from argument string or dict."""
        args = {}
        
        if isinstance(arg_string, dict):
            args.update(arg_string)
        elif isinstance(arg_string, str) and arg_string:
            for arg in arg_string.split(","):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    # Try to convert to appropriate types
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    else:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Keep as string
                    args[key.strip()] = value
        
        if additional_config:
            args.update(additional_config)
            
        return cls(**args)