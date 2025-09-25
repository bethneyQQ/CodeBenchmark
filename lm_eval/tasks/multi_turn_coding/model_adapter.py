"""
Universal Model Adapter for Multi-Turn Coding Evaluation

This module provides a unified interface for different model backends,
allowing the multi-turn coding evaluation to work with various models
including DeepSeek, Claude Code, OpenAI, and others.
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Standardized response from any model."""
    content: str
    model_name: str
    execution_time: float
    token_usage: Optional[Dict[str, int]] = None
    error: Optional[str] = None


class UniversalModelAdapter:
    """Universal adapter to interface with different model backends."""
    
    def __init__(self, model_name: str, model_args: Dict[str, Any]):
        self.model_name = model_name
        self.model_args = model_args
        self.model = None
        self._setup_model()
    
    def _setup_model(self):
        """Setup the specific model backend based on model_name."""
        logger.info(f"Setting up model: {self.model_name}")
        
        if self.model_name in ["claude-code-local", "claude-code"]:
            self._setup_claude_code()
        elif self.model_name in ["deepseek", "deepseek-v3"]:
            self._setup_deepseek()
        elif self.model_name in ["dashscope"]:
            self._setup_dashscope()
        elif self.model_name.startswith("gpt"):
            self._setup_openai()
        elif self.model_name.startswith("claude"):
            self._setup_anthropic()
        else:
            logger.warning(f"Unknown model backend: {self.model_name}")
            self._setup_generic_lm_eval()
    
    def _setup_claude_code(self):
        """Setup Claude Code SDK model."""
        try:
            from lm_eval.models.claude_code_local import ClaudeCodeLocal
            
            # Extract Claude Code specific parameters
            model_id = self.model_args.get('model', 'claude-3-haiku-20240307')
            multi_turn = self.model_args.get('multi_turn', True)
            
            self.model = ClaudeCodeLocal(
                model=model_id,
                multi_turn=multi_turn,
                permission_mode='bypassPermissions'
            )
            logger.info(f"✅ Claude Code model initialized: {model_id}")
            
        except ImportError:
            logger.error("Claude Code SDK not available. Install with: pip install claude-code-sdk")
            raise
        except Exception as e:
            logger.error(f"Failed to setup Claude Code: {e}")
            raise
    
    def _setup_deepseek(self):
        """Setup DeepSeek model."""
        try:
            from lm_eval.models.deepseek_model import DeepSeekLM
            
            model_id = self.model_args.get('model', 'deepseek-v3.1')
            api_key = self.model_args.get('api_key') or os.environ.get('DASHSCOPE_API_KEY')
            
            self.model = DeepSeekLM(
                model=model_id,
                api_key=api_key,
                **{k: v for k, v in self.model_args.items() if k not in ['model', 'api_key']}
            )
            logger.info(f"✅ DeepSeek model initialized: {model_id}")
            
        except ImportError:
            logger.error("DeepSeek model not available. Check lm_eval installation.")
            raise
        except Exception as e:
            logger.error(f"Failed to setup DeepSeek: {e}")
            raise
    
    def _setup_dashscope(self):
        """Setup DashScope model."""
        try:
            from lm_eval.models.dashscope_model import DashScopeLM
            
            model_id = self.model_args.get('model', 'qwen-turbo')
            api_key = self.model_args.get('api_key') or os.environ.get('DASHSCOPE_API_KEY')
            
            self.model = DashScopeLM(
                model=model_id,
                api_key=api_key,
                **{k: v for k, v in self.model_args.items() if k not in ['model', 'api_key']}
            )
            logger.info(f"✅ DashScope model initialized: {model_id}")
            
        except ImportError:
            logger.error("DashScope model not available. Check lm_eval installation.")
            raise
        except Exception as e:
            logger.error(f"Failed to setup DashScope: {e}")
            raise
    
    def _setup_openai(self):
        """Setup OpenAI model."""
        try:
            from lm_eval.models.openai_completions import OpenAILM
            
            model_id = self.model_args.get('model', 'gpt-3.5-turbo')
            api_key = self.model_args.get('api_key') or os.environ.get('OPENAI_API_KEY')
            
            self.model = OpenAILM(
                model=model_id,
                api_key=api_key,
                **{k: v for k, v in self.model_args.items() if k not in ['model', 'api_key']}
            )
            logger.info(f"✅ OpenAI model initialized: {model_id}")
            
        except ImportError:
            logger.error("OpenAI model not available. Install openai package.")
            raise
        except Exception as e:
            logger.error(f"Failed to setup OpenAI: {e}")
            raise
    
    def _setup_anthropic(self):
        """Setup Anthropic Claude API model.""" 
        try:
            from lm_eval.models.anthropic_llms import AnthropicLM
            
            model_id = self.model_args.get('model', 'claude-3-haiku-20240307')
            api_key = self.model_args.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
            
            self.model = AnthropicLM(
                model=model_id,
                api_key=api_key,
                **{k: v for k, v in self.model_args.items() if k not in ['model', 'api_key']}
            )
            logger.info(f"✅ Anthropic model initialized: {model_id}")
            
        except ImportError:
            logger.error("Anthropic model not available. Install anthropic package.")
            raise
        except Exception as e:
            logger.error(f"Failed to setup Anthropic: {e}")
            raise
    
    def _setup_generic_lm_eval(self):
        """Setup generic lm-eval model as fallback."""
        logger.warning(f"Using generic lm-eval setup for {self.model_name}")
        # This would need to be implemented based on lm-eval's model loading mechanism
        pass
    
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using the model with unified interface."""
        import time
        
        start_time = time.time()
        
        try:
            # Prepare generation parameters
            gen_kwargs = {
                'temperature': kwargs.get('temperature', 0.0),
                'max_gen_toks': kwargs.get('max_gen_toks', 3000),
                'until': kwargs.get('until', []),
                'do_sample': kwargs.get('do_sample', False)
            }
            
            # Create request in lm-eval format
            if hasattr(self.model, 'generate_until'):
                # Standard lm-eval interface
                requests = [{
                    "context": prompt,
                    "gen_kwargs": gen_kwargs
                }]
                
                responses = self.model.generate_until(requests)
                content = responses[0] if responses else ""
                
            elif self.model_name in ["claude-code-local", "claude-code"]:
                # Claude Code specific interface
                content = self._generate_claude_code(prompt, **gen_kwargs)
                
            else:
                # Fallback generic generation
                content = self._generate_generic(prompt, **gen_kwargs)
            
            execution_time = time.time() - start_time
            
            return ModelResponse(
                content=content,
                model_name=self.model_name,
                execution_time=execution_time,
                token_usage=None  # Could be implemented per model
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error generating with {self.model_name}: {e}")
            
            return ModelResponse(
                content="",
                model_name=self.model_name,
                execution_time=execution_time,
                error=str(e)
            )
    
    def _generate_claude_code(self, prompt: str, **kwargs) -> str:
        """Special handling for Claude Code model."""
        try:
            # Claude Code has its own interface
            if hasattr(self.model, 'generate_until'):
                requests = [{"context": prompt, "gen_kwargs": kwargs}]
                responses = self.model.generate_until(requests)
                return responses[0] if responses else ""
            else:
                logger.warning("Claude Code model doesn't support expected interface")
                return ""
        except Exception as e:
            logger.error(f"Claude Code generation error: {e}")
            return ""
    
    def _generate_generic(self, prompt: str, **kwargs) -> str:
        """Generic generation for unknown model types."""
        logger.warning(f"Using generic generation for {self.model_name}")
        return f"[Generated by {self.model_name}] {prompt[:100]}..."
    
    def is_available(self) -> bool:
        """Check if the model is properly initialized and available."""
        return self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            'model_name': self.model_name,
            'model_args': self.model_args,
            'is_available': self.is_available(),
            'model_type': type(self.model).__name__ if self.model else None
        }


def create_model_adapter(model_name: str, model_args: Dict[str, Any]) -> UniversalModelAdapter:
    """Factory function to create appropriate model adapter."""
    return UniversalModelAdapter(model_name, model_args)


def parse_model_args(model_args_string: str) -> Dict[str, Any]:
    """Parse model arguments from string format like 'model=claude-3-haiku,multi_turn=true'."""
    args = {}
    if not model_args_string:
        return args
    
    for arg in model_args_string.split(','):
        if '=' in arg:
            key, value = arg.strip().split('=', 1)
            
            # Type conversion
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
            
            args[key.strip()] = value
    
    return args


# Example usage and testing
if __name__ == "__main__":
    # Test different model setups
    test_models = [
        ("claude-code-local", {"model": "claude-3-haiku-20240307", "multi_turn": True}),
        ("deepseek", {"model": "deepseek-v3.1"}),
        ("dashscope", {"model": "qwen-turbo"}),
    ]
    
    for model_name, model_args in test_models:
        try:
            adapter = create_model_adapter(model_name, model_args)
            info = adapter.get_model_info()
            print(f"✅ {model_name}: {info}")
            
            if adapter.is_available():
                response = adapter.generate("Hello, world!", temperature=0.0)
                print(f"   Response: {response.content[:100]}...")
            
        except Exception as e:
            print(f"❌ {model_name}: {e}")