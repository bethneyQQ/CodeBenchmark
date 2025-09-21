import logging
import os
from typing import Any, Dict, List, Tuple, Union

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import retry_on_specific_exceptions


eval_logger = logging.getLogger(__name__)


def claude_local_completion(
    client,  #: anthropic.Anthropic,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stop: List[str],
    **kwargs: Any,
) -> str:
    """Local wrapper function around the Anthropic SDK with exponential back-off
    in case of RateLimitError.

    params:
        client: anthropic.Anthropic
            Anthropic SDK client (local instance)
        model: str
            Anthropic model e.g. 'claude-3-haiku-20240307', 'claude-3-5-sonnet-20241022'
        prompt: str
            Prompt to feed to the model
        max_tokens: int
            Maximum number of tokens to generate
        temperature: float
            Sampling temperature
        stop: List[str]
            List of stop sequences
        kwargs: Any
            Additional model_args to pass to the SDK
    """

    try:
        import anthropic
    except ModuleNotFoundError as exception:
        raise type(exception)(
            "attempted to use 'claude-local' LM type, but package `anthropic` is not installed. \
please install anthropic via `pip install 'lm-eval[anthropic]'` or `pip install -e '.[anthropic]'`",
        )

    def _exception_callback(e: Exception, sleep_time: float) -> None:
        eval_logger.warning(
            f"RateLimitError occurred: {e.__cause__}\n Retrying in {sleep_time} seconds"
        )

    @retry_on_specific_exceptions(
        on_exceptions=[
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.APIStatusError,
        ],
        max_retries=3,  # Limited retries for local usage
        on_exception_callback=_exception_callback,
    )
    def messages():
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.content[0].text

    return messages()


@register_model("claude-local")
class ClaudeLocal(LM):
    """Local Claude model wrapper using Anthropic SDK directly."""
    
    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        api_key: str = None,
        **kwargs,
    ) -> None:
        """Local Claude SDK wrapper.

        :param model: str
            Claude model name e.g. 'claude-3-haiku-20240307', 'claude-3-5-sonnet-20241022'
        :param max_tokens: int
            Maximum number of tokens to generate
        :param temperature: float
            Sampling temperature
        :param api_key: str
            Anthropic API key (if None, uses environment variable)
        :param kwargs: Any
            Additional model_args to pass to the SDK
        """
        super().__init__()

        try:
            import anthropic
        except ModuleNotFoundError as exception:
            raise type(exception)(
                "attempted to use 'claude-local' LM type, but package `anthropic` is not installed. \
please install anthropic via `pip install 'lm-eval[anthropic]'` or `pip install -e '.[anthropic]'`",
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.kwargs = kwargs
        
        # Initialize Anthropic client
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            # Uses ANTHROPIC_API_KEY environment variable
            self.client = anthropic.Anthropic()
        
        eval_logger.info(f"Initialized Claude Local model: {model}")

    @property
    def eot_token_id(self):
        # Claude doesn't expose tokenization details
        raise NotImplementedError("Claude models don't expose tokenization details.")

    @property
    def max_length(self) -> int:
        # Claude 3 models have different context lengths
        model_context_lengths = {
            "claude-3-haiku-20240307": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-5-sonnet-20241022": 200000,
            "claude-3-opus-20240229": 200000,
        }
        return model_context_lengths.get(self.model, 200000)

    @property
    def max_gen_toks(self) -> int:
        return self.max_tokens

    @property
    def batch_size(self):
        # Claude API doesn't support batching
        return 1

    @property
    def device(self):
        # Not applicable for API models
        return "api"

    def tok_encode(self, string: str) -> List[int]:
        # Claude doesn't provide direct tokenization access
        # Return approximate token count based on character length
        # This is a rough approximation: ~4 characters per token
        return list(range(len(string) // 4 + 1))

    def tok_decode(self, tokens: List[int]) -> str:
        # Since we can't actually decode tokens without the tokenizer,
        # this method should not be used for Claude models
        raise NotImplementedError("Token decoding not available for Claude models")

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("Claude models don't support loglikelihood computation.")

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        """Generate text using Claude SDK locally."""
        try:
            import anthropic
        except ModuleNotFoundError as exception:
            raise type(exception)(
                "attempted to use 'claude-local' LM type, but package `anthropic` is not installed. \
please install anthropic via `pip install 'lm-eval[anthropic]'` or `pip install -e '.[anthropic]'`",
            )

        if not requests:
            return []

        _requests: List[Tuple[str, dict]] = [req.args for req in requests]

        res = []
        for request in tqdm(_requests, disable=disable_tqdm, desc="Claude Local Generation"):
            try:
                inp = request[0]
                request_args = request[1]
                
                # Extract generation parameters
                until = request_args.get("until", [])
                max_gen_toks = request_args.get("max_gen_toks", self.max_tokens)
                temperature = request_args.get("temperature", self.temperature)
                
                # Filter out lm_eval specific parameters that shouldn't go to Anthropic API
                api_kwargs = {k: v for k, v in self.kwargs.items() 
                             if k not in ['batch_size', 'limit', 'num_fewshot']}
                
                # Generate response using Claude SDK
                response = claude_local_completion(
                    client=self.client,
                    model=self.model,
                    prompt=inp,
                    max_tokens=max_gen_toks,
                    temperature=temperature,
                    stop=until,
                    **api_kwargs,
                )
                res.append(response)

                # Cache the result
                self.cache_hook.add_partial("generate_until", request, response)
                
            except anthropic.APIConnectionError as e:
                eval_logger.critical(f"Claude API connection error: {e}")
                res.append("")
            except anthropic.APIStatusError as e:
                eval_logger.critical(f"Claude API error {e.status_code}: {e.message}")
                res.append("")
            except Exception as e:
                eval_logger.error(f"Unexpected error with Claude Local: {e}")
                res.append("")

        return res

    def _model_call(self, inps):
        # Not used because we override generate_until
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Not used because we override generate_until
        raise NotImplementedError()

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("Claude models don't support loglikelihood computation.")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("Claude models don't support loglikelihood computation.")