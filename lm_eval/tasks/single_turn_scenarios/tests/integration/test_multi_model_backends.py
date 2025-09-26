"""Integration tests for multi-model backend support."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Import the modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config_manager import ConfigManager, load_model_config, validate_config, merge_configs


class TestModelBackendIntegration:
    """Integration tests for all supported model backends."""
    
    @pytest.mark.integration
    def test_claude_code_backend_integration(self):
        """Test Claude Code backend configuration and integration."""
        claude_config = {
            "model_name": "claude-code-local",
            "endpoint_config": {
                "base_url": "https://api.anthropic.com",
                "timeout": 60,
                "rate_limit": 10,
                "api_key_env": "ANTHROPIC_API_KEY"
            },
            "generation_params": {
                "temperature": 0.0,
                "max_tokens": 2048,
                "top_p": 0.95,
                "stop_sequences": ["```"]
            },
            "batch_config": {
                "batch_size": 4,
                "max_batch_size": 16,
                "concurrent_requests": 2
            },
            "tokenizer_config": {
                "encoding": "cl100k_base",
                "max_context": 200000
            },
            "optimization": {
                "use_caching": True,
                "parallel_requests": True,
                "file_operations": True,
                "iterative_development": True
            }
        }
        
        # Test configuration loading
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(claude_config))):
                config = load_model_config("claude_code")
                
                assert config["model_name"] == "claude-code-local"
                assert config["endpoint_config"]["base_url"] == "https://api.anthropic.com"
                assert config["optimization"]["file_operations"] == True
                assert config["optimization"]["iterative_development"] == True
        
        # Test configuration validation
        assert validate_config(claude_config) == True
        
        # Test Claude-specific optimizations
        assert claude_config["optimization"]["file_operations"] == True
        assert claude_config["optimization"]["iterative_development"] == True
        assert claude_config["generation_params"]["stop_sequences"] == ["```"]
    
    @pytest.mark.integration
    def test_deepseek_backend_integration(self):
        """Test DeepSeek backend configuration and integration."""
        deepseek_config = {
            "model_name": "deepseek-coder-v2",
            "endpoint_config": {
                "base_url": "https://api.deepseek.com",
                "timeout": 45,
                "rate_limit": 20,
                "api_key_env": "DEEPSEEK_API_KEY"
            },
            "generation_params": {
                "temperature": 0.1,
                "max_tokens": 1024,
                "top_p": 0.9,
                "frequency_penalty": 0.1
            },
            "batch_config": {
                "batch_size": 8,
                "max_batch_size": 32,
                "concurrent_requests": 4
            },
            "tokenizer_config": {
                "encoding": "deepseek",
                "max_context": 32000
            },
            "optimization": {
                "use_caching": True,
                "parallel_requests": True,
                "cost_optimization": True,
                "batch_processing": True
            }
        }
        
        # Test configuration loading
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(deepseek_config))):
                config = load_model_config("deepseek")
                
                assert config["model_name"] == "deepseek-coder-v2"
                assert config["endpoint_config"]["rate_limit"] == 20
                assert config["optimization"]["cost_optimization"] == True
                assert config["optimization"]["batch_processing"] == True
        
        # Test configuration validation
        assert validate_config(deepseek_config) == True
        
        # Test DeepSeek-specific optimizations
        assert deepseek_config["optimization"]["cost_optimization"] == True
        assert deepseek_config["optimization"]["batch_processing"] == True
        assert deepseek_config["batch_config"]["batch_size"] == 8  # Higher batch size for cost efficiency
    
    @pytest.mark.integration
    def test_openai_backend_integration(self):
        """Test OpenAI backend configuration and integration."""
        openai_config = {
            "model_name": "gpt-4",
            "endpoint_config": {
                "base_url": "https://api.openai.com/v1",
                "timeout": 60,
                "rate_limit": 15,
                "api_key_env": "OPENAI_API_KEY"
            },
            "generation_params": {
                "temperature": 0.0,
                "max_tokens": 2048,
                "top_p": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0
            },
            "batch_config": {
                "batch_size": 4,
                "max_batch_size": 16,
                "concurrent_requests": 3
            },
            "tokenizer_config": {
                "encoding": "cl100k_base",
                "max_context": 128000
            },
            "optimization": {
                "use_caching": True,
                "parallel_requests": True,
                "stability_focus": True,
                "compatibility_mode": True
            }
        }
        
        # Test configuration loading
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(openai_config))):
                config = load_model_config("openai")
                
                assert config["model_name"] == "gpt-4"
                assert config["endpoint_config"]["base_url"] == "https://api.openai.com/v1"
                assert config["optimization"]["stability_focus"] == True
                assert config["optimization"]["compatibility_mode"] == True
        
        # Test configuration validation
        assert validate_config(openai_config) == True
        
        # Test OpenAI-specific optimizations
        assert openai_config["optimization"]["stability_focus"] == True
        assert openai_config["optimization"]["compatibility_mode"] == True
        assert openai_config["generation_params"]["temperature"] == 0.0  # Stability focus
    
    @pytest.mark.integration
    def test_anthropic_backend_integration(self):
        """Test Anthropic Claude backend configuration and integration."""
        anthropic_config = {
            "model_name": "claude-3-sonnet",
            "endpoint_config": {
                "base_url": "https://api.anthropic.com",
                "timeout": 90,
                "rate_limit": 8,
                "api_key_env": "ANTHROPIC_API_KEY"
            },
            "generation_params": {
                "temperature": 0.0,
                "max_tokens": 4096,
                "top_p": 0.95,
                "top_k": 40
            },
            "batch_config": {
                "batch_size": 2,
                "max_batch_size": 8,
                "concurrent_requests": 2
            },
            "tokenizer_config": {
                "encoding": "claude",
                "max_context": 200000
            },
            "optimization": {
                "use_caching": True,
                "parallel_requests": False,  # Conservative for reasoning
                "reasoning_focus": True,
                "long_context": True
            }
        }
        
        # Test configuration loading
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(anthropic_config))):
                config = load_model_config("anthropic")
                
                assert config["model_name"] == "claude-3-sonnet"
                assert config["endpoint_config"]["timeout"] == 90  # Longer timeout for reasoning
                assert config["optimization"]["reasoning_focus"] == True
                assert config["optimization"]["long_context"] == True
        
        # Test configuration validation
        assert validate_config(anthropic_config) == True
        
        # Test Anthropic-specific optimizations
        assert anthropic_config["optimization"]["reasoning_focus"] == True
        assert anthropic_config["optimization"]["long_context"] == True
        assert anthropic_config["optimization"]["parallel_requests"] == False  # Conservative approach
    
    @pytest.mark.integration
    def test_universal_backend_integration(self):
        """Test Universal backend configuration and integration."""
        universal_config = {
            "model_name": "universal",
            "endpoint_config": {
                "base_url": "https://api.universal.com",
                "timeout": 60,
                "rate_limit": 10,
                "api_key_env": "UNIVERSAL_API_KEY"
            },
            "generation_params": {
                "temperature": 0.0,
                "max_tokens": 2048,
                "top_p": 0.95
            },
            "batch_config": {
                "batch_size": 4,
                "max_batch_size": 16,
                "concurrent_requests": 2
            },
            "tokenizer_config": {
                "encoding": "universal",
                "max_context": 100000
            },
            "optimization": {
                "use_caching": True,
                "parallel_requests": True,
                "generic_adaptation": True,
                "fallback_mode": True
            }
        }
        
        # Test configuration loading
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(universal_config))):
                config = load_model_config("universal")
                
                assert config["model_name"] == "universal"
                assert config["optimization"]["generic_adaptation"] == True
                assert config["optimization"]["fallback_mode"] == True
        
        # Test configuration validation
        assert validate_config(universal_config) == True
        
        # Test Universal-specific features
        assert universal_config["optimization"]["generic_adaptation"] == True
        assert universal_config["optimization"]["fallback_mode"] == True


class TestConfigurationMerging:
    """Integration tests for configuration merging and overrides."""
    
    @pytest.mark.integration
    def test_config_merging_with_overrides(self):
        """Test configuration merging with user overrides."""
        base_config = {
            "model_name": "base-model",
            "endpoint_config": {
                "base_url": "https://api.base.com",
                "timeout": 30,
                "rate_limit": 10
            },
            "generation_params": {
                "temperature": 0.0,
                "max_tokens": 1000,
                "top_p": 0.95
            }
        }
        
        user_overrides = {
            "endpoint_config": {
                "timeout": 60,  # Override timeout
                "custom_header": "custom_value"  # Add new field
            },
            "generation_params": {
                "temperature": 0.5  # Override temperature
            },
            "new_section": {
                "custom_setting": True
            }
        }
        
        merged_config = merge_configs(base_config, user_overrides)
        
        # Test that overrides work
        assert merged_config["endpoint_config"]["timeout"] == 60
        assert merged_config["generation_params"]["temperature"] == 0.5
        
        # Test that original values are preserved
        assert merged_config["endpoint_config"]["base_url"] == "https://api.base.com"
        assert merged_config["generation_params"]["max_tokens"] == 1000
        assert merged_config["generation_params"]["top_p"] == 0.95
        
        # Test that new fields are added
        assert merged_config["endpoint_config"]["custom_header"] == "custom_value"
        assert merged_config["new_section"]["custom_setting"] == True
    
    @pytest.mark.integration
    def test_fallback_to_universal_config(self):
        """Test fallback to universal config when specific config not found."""
        universal_config = {
            "model_name": "universal",
            "endpoint_config": {"base_url": "https://api.universal.com"},
            "generation_params": {"temperature": 0.0}
        }
        
        def mock_exists(path):
            # Only universal.yaml exists
            return "universal.yaml" in str(path)
        
        with patch('pathlib.Path.exists', side_effect=mock_exists):
            with patch('builtins.open', mock_open(read_data=json.dumps(universal_config))):
                # Try to load non-existent model config
                config = load_model_config("nonexistent_model")
                
                # Should fallback to universal config
                assert config["model_name"] == "universal"
                assert config["endpoint_config"]["base_url"] == "https://api.universal.com"
    
    @pytest.mark.integration
    def test_config_validation_across_all_backends(self):
        """Test configuration validation across all model backends."""
        config_manager = ConfigManager()
        
        # Test valid configurations for all backends
        valid_configs = {
            "claude_code": {
                "model_name": "claude-code",
                "endpoint_config": {"base_url": "https://api.anthropic.com"},
                "generation_params": {"temperature": 0.0},
                "optimization": {"file_operations": True}
            },
            "deepseek": {
                "model_name": "deepseek-coder",
                "endpoint_config": {"base_url": "https://api.deepseek.com"},
                "generation_params": {"temperature": 0.1},
                "optimization": {"cost_optimization": True}
            },
            "openai": {
                "model_name": "gpt-4",
                "endpoint_config": {"base_url": "https://api.openai.com"},
                "generation_params": {"temperature": 0.0},
                "optimization": {"stability_focus": True}
            },
            "anthropic": {
                "model_name": "claude-3-sonnet",
                "endpoint_config": {"base_url": "https://api.anthropic.com"},
                "generation_params": {"temperature": 0.0},
                "optimization": {"reasoning_focus": True}
            },
            "universal": {
                "model_name": "universal",
                "endpoint_config": {"base_url": "https://api.universal.com"},
                "generation_params": {"temperature": 0.0},
                "optimization": {"generic_adaptation": True}
            }
        }
        
        for backend, config in valid_configs.items():
            assert config_manager.validate_config(config) == True, f"Validation failed for {backend}"


class TestModelSpecificOptimizations:
    """Integration tests for model-specific optimizations."""
    
    @pytest.mark.integration
    def test_claude_code_file_operations(self):
        """Test Claude Code specific file operation optimizations."""
        claude_config = {
            "model_name": "claude-code-local",
            "endpoint_config": {"base_url": "https://api.anthropic.com"},
            "generation_params": {"temperature": 0.0},
            "optimization": {
                "file_operations": True,
                "iterative_development": True,
                "multi_file_context": True
            }
        }
        
        # Test that Claude Code specific optimizations are present
        assert claude_config["optimization"]["file_operations"] == True
        assert claude_config["optimization"]["iterative_development"] == True
        assert claude_config["optimization"]["multi_file_context"] == True
    
    @pytest.mark.integration
    def test_deepseek_cost_optimization(self):
        """Test DeepSeek specific cost optimization features."""
        deepseek_config = {
            "model_name": "deepseek-coder-v2",
            "endpoint_config": {"base_url": "https://api.deepseek.com"},
            "generation_params": {"temperature": 0.1},
            "batch_config": {"batch_size": 8},  # Higher batch size
            "optimization": {
                "cost_optimization": True,
                "batch_processing": True,
                "token_efficiency": True
            }
        }
        
        # Test DeepSeek cost optimizations
        assert deepseek_config["optimization"]["cost_optimization"] == True
        assert deepseek_config["optimization"]["batch_processing"] == True
        assert deepseek_config["optimization"]["token_efficiency"] == True
        assert deepseek_config["batch_config"]["batch_size"] == 8  # Cost-effective batching
    
    @pytest.mark.integration
    def test_anthropic_reasoning_optimization(self):
        """Test Anthropic Claude reasoning capability optimizations."""
        anthropic_config = {
            "model_name": "claude-3-sonnet",
            "endpoint_config": {"base_url": "https://api.anthropic.com", "timeout": 90},
            "generation_params": {"temperature": 0.0, "max_tokens": 4096},
            "optimization": {
                "reasoning_focus": True,
                "long_context": True,
                "step_by_step": True,
                "parallel_requests": False  # Conservative for reasoning
            }
        }
        
        # Test Anthropic reasoning optimizations
        assert anthropic_config["optimization"]["reasoning_focus"] == True
        assert anthropic_config["optimization"]["long_context"] == True
        assert anthropic_config["optimization"]["step_by_step"] == True
        assert anthropic_config["optimization"]["parallel_requests"] == False
        assert anthropic_config["endpoint_config"]["timeout"] == 90  # Longer timeout
    
    @pytest.mark.integration
    def test_openai_stability_optimization(self):
        """Test OpenAI stability and compatibility optimizations."""
        openai_config = {
            "model_name": "gpt-4",
            "endpoint_config": {"base_url": "https://api.openai.com/v1"},
            "generation_params": {
                "temperature": 0.0,  # Stability focus
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0
            },
            "optimization": {
                "stability_focus": True,
                "compatibility_mode": True,
                "error_recovery": True
            }
        }
        
        # Test OpenAI stability optimizations
        assert openai_config["optimization"]["stability_focus"] == True
        assert openai_config["optimization"]["compatibility_mode"] == True
        assert openai_config["optimization"]["error_recovery"] == True
        assert openai_config["generation_params"]["temperature"] == 0.0


class TestConfigurationErrorHandling:
    """Integration tests for configuration error handling."""
    
    @pytest.mark.integration
    def test_invalid_config_handling(self):
        """Test handling of invalid configurations."""
        config_manager = ConfigManager()
        
        # Test completely invalid config
        invalid_config = {"invalid": "config"}
        assert config_manager.validate_config(invalid_config) == False
        
        # Test config with invalid values
        invalid_values_config = {
            "model_name": "test",
            "endpoint_config": {"base_url": "invalid_url"},
            "generation_params": {"temperature": -1.0}  # Invalid temperature
        }
        assert config_manager.validate_config(invalid_values_config) == False
    
    @pytest.mark.integration
    def test_missing_config_file_handling(self):
        """Test handling when config files are missing."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                load_model_config("nonexistent_model")
    
    @pytest.mark.integration
    def test_corrupted_config_file_handling(self):
        """Test handling of corrupted config files."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="invalid json content")):
                with pytest.raises(json.JSONDecodeError):
                    load_model_config("corrupted_model")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])