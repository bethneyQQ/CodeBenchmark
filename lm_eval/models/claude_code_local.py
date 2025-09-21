import logging
import asyncio
from typing import Any, Dict, List, Tuple, Union

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


eval_logger = logging.getLogger(__name__)


@register_model("claude-code-local")
class ClaudeCodeLocal(LM):
    """Local Claude Code model wrapper using Claude Code SDK."""
    
    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 3000,
        temperature: float = 0.0,
        cwd: str = None,
        allowed_tools: List[str] = None,
        debug: bool = False,
        multi_turn: bool = False,
        permission_mode: str = "bypassPermissions",
        **kwargs,
    ) -> None:
        """Local Claude Code SDK wrapper.

        :param model: str
            Claude model name e.g. 'claude-3-haiku-20240307', 'claude-3-5-sonnet-20241022', 'claude-sonnet-4-20250514'
        :param max_tokens: int
            Maximum number of tokens to generate (not used by Claude Code SDK directly)
        :param temperature: float
            Sampling temperature (not used by Claude Code SDK directly)
        :param cwd: str
            Current working directory for Claude Code (optional)
        :param allowed_tools: List[str]
            List of allowed tools for Claude Code (optional)
        :param debug: bool
            Enable debug mode to print Claude Code responses (default: False)
        :param multi_turn: bool
            Enable multi-turn conversation support (default: False)
        :param permission_mode: str
            Permission mode for file operations (default: "bypassPermissions")
        :param kwargs: Any
            Additional model_args
        """
        super().__init__()

        try:
            import claude_code_sdk
        except ModuleNotFoundError as exception:
            raise type(exception)(
                "attempted to use 'claude-code-local' LM type, but package `claude-code-sdk` is not installed. \
please install claude-code-sdk via `pip install claude-code-sdk`",
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cwd = cwd
        self.allowed_tools = allowed_tools or []
        self.debug = debug
        self.multi_turn = multi_turn
        self.permission_mode = permission_mode
        self.kwargs = kwargs
        
        # Multi-turn conversation state
        self.conversation_history = []
        self.current_client = None
        
        # Import Claude Code SDK components
        self.claude_code_sdk = claude_code_sdk
        
        eval_logger.info(f"Initialized Claude Code Local model: {model}")
        if self.cwd:
            eval_logger.info(f"Working directory: {self.cwd}")
        if self.allowed_tools:
            eval_logger.info(f"Allowed tools: {self.allowed_tools}")
        if self.debug:
            eval_logger.info("Debug mode enabled - will print Claude Code responses")
        if self.multi_turn:
            eval_logger.info("Multi-turn mode enabled - will maintain conversation state")
        eval_logger.info(f"Permission mode: {self.permission_mode}")

    @property
    def eot_token_id(self):
        # Claude doesn't expose tokenization details
        raise NotImplementedError("Claude models don't expose tokenization details.")

    @property
    def max_length(self) -> int:
        # Claude models have different context lengths
        model_context_lengths = {
            "claude-3-haiku-20240307": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-5-haiku-20241022": 200000,
            "claude-sonnet-4-20250514": 200000,
            "claude-3-opus-20240229": 200000,
        }
        return model_context_lengths.get(self.model, 200000)

    @property
    def max_gen_toks(self) -> int:
        return self.max_tokens

    @property
    def batch_size(self):
        # Claude Code SDK doesn't support batching
        return 1

    @property
    def device(self):
        # Not applicable for SDK models
        return "claude-code-sdk"

    def tok_encode(self, string: str) -> List[int]:
        # Claude doesn't provide direct tokenization access
        # Return approximate token count based on character length
        # This is a rough approximation: ~4 characters per token
        return list(range(len(string) // 4 + 1))

    def tok_decode(self, tokens: List[int]) -> str:
        # Since we can't actually decode tokens without the tokenizer,
        # this method should not be used for Claude models
        raise NotImplementedError("Token decoding not available for Claude Code models")

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("Claude Code models don't support loglikelihood computation.")

    async def _async_query_claude_code_multiturn(self, prompt: str, **kwargs) -> str:
        """Async function for multi-turn conversation with Claude Code SDK."""
        try:
            if self.current_client is None:
                # Initialize new conversation
                options_dict = {
                    'model': self.model,
                    'permission_mode': self.permission_mode,
                }
                
                if self.cwd:
                    options_dict['cwd'] = self.cwd
                    
                if self.allowed_tools:
                    options_dict['allowed_tools'] = self.allowed_tools
                
                filtered_kwargs = {k: v for k, v in kwargs.items() 
                                 if k not in ['max_tokens', 'batch_size', 'limit', 'num_fewshot', 'temperature', 'debug', 'multi_turn', 'permission_mode']}
                options_dict.update(filtered_kwargs)
                
                options = self.claude_code_sdk.ClaudeCodeOptions(**options_dict)
                self.current_client = self.claude_code_sdk.ClaudeSDKClient(options=options)
                await self.current_client.__aenter__()
            
            # Send message and get response
            await self.current_client.query(prompt)
            
            response_parts = []
            async for response in self.current_client.receive_response():
                if self.debug:
                    print(f"\n🐛 DEBUG - Multi-turn Claude Code Response: {response}")
                
                if hasattr(response, 'content'):
                    for content in response.content:
                        if hasattr(content, 'text'):
                            response_parts.append(content.text)
                elif hasattr(response, 'text'):
                    response_parts.append(response.text)
                else:
                    response_parts.append(str(response))
            
            final_response = ''.join(response_parts)
            
            if self.debug:
                print(f"\n🐛 DEBUG - Multi-turn final response:")
                print(f"{'='*50}")
                print(final_response)
                print(f"{'='*50}")
            
            return final_response
            
        except Exception as e:
            eval_logger.error(f"Claude Code SDK multi-turn error: {e}")
            # Cleanup on error
            if self.current_client:
                try:
                    await self.current_client.__aexit__(None, None, None)
                except Exception as cleanup_error:
                    eval_logger.warning(f"Error during cleanup: {cleanup_error}")
                finally:
                    self.current_client = None
            return ""

    async def _async_query_claude_code(self, prompt: str, **kwargs) -> str:
        """Async function to query Claude Code SDK."""
        try:
            # Create Claude Code options with correct parameters
            options_dict = {
                'model': self.model,
            }
            
            # Add working directory if specified (use 'cwd' not 'working_directory')
            if self.cwd:
                options_dict['cwd'] = self.cwd
                
            # Add allowed tools if specified
            if self.allowed_tools:
                options_dict['allowed_tools'] = self.allowed_tools
            
            # Add permission mode
            options_dict['permission_mode'] = self.permission_mode
            
            # Add any additional kwargs (but filter out incompatible ones)
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ['max_tokens', 'batch_size', 'limit', 'num_fewshot', 'temperature', 'debug', 'multi_turn', 'permission_mode']}
            options_dict.update(filtered_kwargs)
            
            options = self.claude_code_sdk.ClaudeCodeOptions(**options_dict)
            
            # Use the query function for simple queries (correct keyword-only API)
            response_parts = []
            async for response in self.claude_code_sdk.query(prompt=prompt, options=options):
                if self.debug:
                    print(f"\n🐛 DEBUG - Claude Code Response: {response}")
                    print(f"🐛 DEBUG - Response type: {type(response)}")
                    if hasattr(response, '__dict__'):
                        print(f"🐛 DEBUG - Response attributes: {response.__dict__}")
                
                if hasattr(response, 'content'):
                    for content in response.content:
                        if hasattr(content, 'text'):
                            response_parts.append(content.text)
                elif hasattr(response, 'text'):
                    response_parts.append(response.text)
                else:
                    response_parts.append(str(response))
            
            final_response = ''.join(response_parts)
            
            if self.debug:
                print(f"\n🐛 DEBUG - Final combined response:")
                print(f"{'='*50}")
                print(final_response)
                print(f"{'='*50}")
            
            return final_response
            
        except Exception as e:
            eval_logger.error(f"Claude Code SDK error: {e}")
            return ""

    def _sync_query_claude_code_multiturn(self, prompt: str, **kwargs) -> str:
        """Synchronous wrapper for async Claude Code multi-turn query."""
        try:
            # Always use a new event loop for multi-turn to avoid conflicts
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(self._async_query_claude_code_multiturn(prompt, **kwargs))
                )
                return future.result(timeout=300)  # 5 minute timeout
        except concurrent.futures.TimeoutError:
            eval_logger.error("Claude Code multi-turn query timed out after 5 minutes")
            # Force cleanup on timeout
            if self.current_client:
                self.current_client = None
            return ""
        except Exception as e:
            eval_logger.error(f"Error in sync Claude Code multi-turn query: {e}")
            # Force cleanup on error
            if self.current_client:
                self.current_client = None
            return ""

    def _sync_query_claude_code(self, prompt: str, **kwargs) -> str:
        """Synchronous wrapper for async Claude Code query."""
        try:
            # Always use a new event loop to avoid conflicts
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(self._async_query_claude_code(prompt, **kwargs))
                )
                return future.result(timeout=300)  # 5 minute timeout
        except concurrent.futures.TimeoutError:
            eval_logger.error("Claude Code query timed out after 5 minutes")
            return ""
        except Exception as e:
            eval_logger.error(f"Error in sync Claude Code query: {e}")
            return ""

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        """Generate text using Claude Code SDK."""
        try:
            import claude_code_sdk
        except ModuleNotFoundError as exception:
            raise type(exception)(
                "attempted to use 'claude-code-local' LM type, but package `claude-code-sdk` is not installed. \
please install claude-code-sdk via `pip install claude-code-sdk`",
            )

        if not requests:
            return []

        _requests: List[Tuple[str, dict]] = [req.args for req in requests]

        res = []
        for request in tqdm(_requests, disable=disable_tqdm, desc="Claude Code Local Generation"):
            try:
                inp = request[0]
                request_args = request[1]
                
                # Extract generation parameters
                until = request_args.get("until", [])
                max_gen_toks = request_args.get("max_gen_toks", self.max_tokens)
                temperature = request_args.get("temperature", self.temperature)
                
                # Filter out lm_eval specific parameters and debug
                api_kwargs = {k: v for k, v in self.kwargs.items() 
                             if k not in ['batch_size', 'limit', 'num_fewshot', 'debug', 'multi_turn', 'permission_mode']}
                
                # Claude Code SDK doesn't support temperature or max_tokens in options
                # These are handled by the underlying Anthropic API
                
                # Generate response using Claude Code SDK
                if self.multi_turn:
                    response = self._sync_query_claude_code_multiturn(inp, **api_kwargs)
                else:
                    response = self._sync_query_claude_code(inp, **api_kwargs)
                res.append(response)

                # Cache the result
                self.cache_hook.add_partial("generate_until", request, response)
                
                # Force cleanup after each request to prevent hanging
                if self.multi_turn and self.current_client:
                    try:
                        self.reset_conversation()
                        eval_logger.debug("Cleaned up multi-turn client after request")
                    except Exception as e:
                        eval_logger.warning(f"Error during post-request cleanup: {e}")
                
            except Exception as e:
                eval_logger.error(f"Unexpected error with Claude Code Local: {e}")
                res.append("")

        return res

    def _model_call(self, inps):
        # Not used because we override generate_until
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Not used because we override generate_until
        raise NotImplementedError()

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("Claude Code models don't support loglikelihood computation.")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("Claude Code models don't support loglikelihood computation.")

    def reset_conversation(self):
        """Reset multi-turn conversation state."""
        if self.current_client:
            try:
                # Try to close the client properly
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is running, schedule cleanup for later
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                lambda: asyncio.run(self.current_client.__aexit__(None, None, None))
                            )
                            future.result(timeout=5)  # 5 second timeout
                    else:
                        loop.run_until_complete(self.current_client.__aexit__(None, None, None))
                except RuntimeError:
                    # No event loop, create one
                    asyncio.run(self.current_client.__aexit__(None, None, None))
                except Exception as e:
                    eval_logger.warning(f"Error during client cleanup: {e}")
            except Exception as e:
                eval_logger.warning(f"Error closing Claude Code client: {e}")
            finally:
                self.current_client = None
        self.conversation_history = []

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'current_client') and self.current_client:
            try:
                self.reset_conversation()
            except Exception as e:
                eval_logger.warning(f"Error in destructor cleanup: {e}")