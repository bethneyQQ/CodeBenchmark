"""Unit tests for config_manager.py module."""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config_manager import (
    ConfigManager, load_model_config, validate_config,
    merge_configs, get_default_config
)


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    def test_config_manager_creation(self):
        """Test ConfigManager creation."""
        manager = ConfigManager()
        assert manager.configs == {}
        assert manager.base_path is not None
    
    def test_load_config_success(self):
        """Test successful config loading."""
        mock_config = {
            "model_name": "test-model",
            "endpoint_config": {"base_url": "https://api.test.com"},
            "generation_params": {"temperature": 0.0}
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(mock_config))):
                manager = ConfigManager()
                config = manager.load_config("test_model")
                
                assert config == mock_config
                assert "test_model" in manager.configs
    
    def test_load_config_file_not_found(self):
        """Test config loading when file doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            manager = ConfigManager()
            
            with pytest.raises(FileNotFoundError):
                manager.load_config("nonexistent_model")
    
    def test_load_config_invalid_json(self):
        """Test config loading with invalid JSON."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="invalid json")):
                manager = ConfigManager()
                
                with pytest.raises(json.JSONDecodeError):
                    manager.load_config("invalid_model")
    
    def test_get_config_cached(self):
        """Test getting cached config."""
        manager = ConfigManager()
        test_config = {"model_name": "cached-model"}
        manager.configs["cached_model"] = test_config
        
        config = manager.get_config("cached_model")
        assert config == test_config
    
    def test_get_config_load_on_demand(self):
        """Test loading config on demand."""
        mock_config = {"model_name": "on-demand-model"}
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(mock_config))):
                manager = ConfigManager()
                config = manager.get_config("on_demand_model")
                
                assert config == mock_config
    
    def test_validate_config_success(self):
        """Test successful config validation."""
        manager = ConfigManager()
        valid_config = {
            "model_name": "test-model",
            "endpoint_config": {"base_url": "https://api.test.com"},
            "generation_params": {"temperature": 0.0}
        }
        
        assert manager.validate_config(valid_config) == True
    
    def test_validate_config_missing_required_fields(self):
        """Test config validation with missing required fields."""
        manager = ConfigManager()
        invalid_config = {
            "model_name": "test-model"
            # Missing endpoint_config and generation_params
        }
        
        assert manager.validate_config(invalid_config) == False
    
    def test_merge_configs_success(self):
        """Test successful config merging."""
        manager = ConfigManager()
        base_config = {
            "model_name": "base-model",
            "endpoint_config": {"base_url": "https://api.base.com"},
            "generation_params": {"temperature": 0.0, "max_tokens": 1000}
        }
        override_config = {
            "generation_params": {"temperature": 0.5},
            "new_field": "new_value"
        }
        
        merged = manager.merge_configs(base_config, override_config)
        
        assert merged["model_name"] == "base-model"
        assert merged["generation_params"]["temperature"] == 0.5
        assert merged["generation_params"]["max_tokens"] == 1000
        assert merged["new_field"] == "new_value"
    
    def test_get_all_configs(self):
        """Test getting all available configs."""
        with patch('pathlib.Path.glob') as mock_glob:
            mock_files = [
                MagicMock(stem="model1"),
                MagicMock(stem="model2"),
                MagicMock(stem="model3")
            ]
            mock_glob.return_value = mock_files
            
            manager = ConfigManager()
            configs = manager.get_all_configs()
            
            assert len(configs) == 3
            assert "model1" in configs
            assert "model2" in configs
            assert "model3" in configs


class TestLoadModelConfig:
    """Test cases for load_model_config function."""
    
    def test_load_model_config_success(self):
        """Test successful model config loading."""
        mock_config = {
            "model_name": "test-model",
            "endpoint_config": {"base_url": "https://api.test.com"}
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(mock_config))):
                config = load_model_config("test_model")
                assert config == mock_config
    
    def test_load_model_config_fallback_to_universal(self):
        """Test fallback to universal config when specific config not found."""
        universal_config = {
            "model_name": "universal",
            "endpoint_config": {"base_url": "https://api.universal.com"}
        }
        
        def mock_exists(path):
            return "universal.yaml" in str(path)
        
        with patch('pathlib.Path.exists', side_effect=mock_exists):
            with patch('builtins.open', mock_open(read_data=json.dumps(universal_config))):
                config = load_model_config("nonexistent_model")
                assert config == universal_config
    
    def test_load_model_config_no_fallback(self):
        """Test when neither specific nor universal config exists."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                load_model_config("nonexistent_model")


class TestValidateConfig:
    """Test cases for validate_config function."""
    
    def test_validate_config_valid_complete(self):
        """Test validation with complete valid config."""
        valid_config = {
            "model_name": "test-model",
            "endpoint_config": {
                "base_url": "https://api.test.com",
                "timeout": 60,
                "rate_limit": 10
            },
            "generation_params": {
                "temperature": 0.0,
                "max_tokens": 2048,
                "top_p": 0.95
            },
            "batch_config": {
                "batch_size": 4,
                "max_batch_size": 16
            },
            "tokenizer_config": {
                "encoding": "cl100k_base",
                "max_context": 200000
            },
            "optimization": {
                "use_caching": True,
                "parallel_requests": True
            }
        }
        
        assert validate_config(valid_config) == True
    
    def test_validate_config_minimal_valid(self):
        """Test validation with minimal valid config."""
        minimal_config = {
            "model_name": "test-model",
            "endpoint_config": {"base_url": "https://api.test.com"},
            "generation_params": {"temperature": 0.0}
        }
        
        assert validate_config(minimal_config) == True
    
    def test_validate_config_missing_model_name(self):
        """Test validation with missing model_name."""
        invalid_config = {
            "endpoint_config": {"base_url": "https://api.test.com"},
            "generation_params": {"temperature": 0.0}
        }
        
        assert validate_config(invalid_config) == False
    
    def test_validate_config_missing_endpoint_config(self):
        """Test validation with missing endpoint_config."""
        invalid_config = {
            "model_name": "test-model",
            "generation_params": {"temperature": 0.0}
        }
        
        assert validate_config(invalid_config) == False
    
    def test_validate_config_missing_generation_params(self):
        """Test validation with missing generation_params."""
        invalid_config = {
            "model_name": "test-model",
            "endpoint_config": {"base_url": "https://api.test.com"}
        }
        
        assert validate_config(invalid_config) == False
    
    def test_validate_config_invalid_temperature(self):
        """Test validation with invalid temperature value."""
        invalid_config = {
            "model_name": "test-model",
            "endpoint_config": {"base_url": "https://api.test.com"},
            "generation_params": {"temperature": -1.0}  # Invalid negative temperature
        }
        
        assert validate_config(invalid_config) == False
    
    def test_validate_config_invalid_max_tokens(self):
        """Test validation with invalid max_tokens value."""
        invalid_config = {
            "model_name": "test-model",
            "endpoint_config": {"base_url": "https://api.test.com"},
            "generation_params": {"temperature": 0.0, "max_tokens": -100}  # Invalid negative tokens
        }
        
        assert validate_config(invalid_config) == False


class TestMergeConfigs:
    """Test cases for merge_configs function."""
    
    def test_merge_configs_simple(self):
        """Test simple config merging."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        
        result = merge_configs(base, override)
        
        assert result == {"a": 1, "b": 3, "c": 4}
    
    def test_merge_configs_nested(self):
        """Test nested config merging."""
        base = {
            "endpoint_config": {"base_url": "https://api.base.com", "timeout": 30},
            "generation_params": {"temperature": 0.0, "max_tokens": 1000}
        }
        override = {
            "endpoint_config": {"timeout": 60},
            "generation_params": {"temperature": 0.5}
        }
        
        result = merge_configs(base, override)
        
        assert result["endpoint_config"]["base_url"] == "https://api.base.com"
        assert result["endpoint_config"]["timeout"] == 60
        assert result["generation_params"]["temperature"] == 0.5
        assert result["generation_params"]["max_tokens"] == 1000
    
    def test_merge_configs_deep_nested(self):
        """Test deep nested config merging."""
        base = {
            "level1": {
                "level2": {
                    "level3": {"value": "base"}
                }
            }
        }
        override = {
            "level1": {
                "level2": {
                    "level3": {"value": "override"}
                }
            }
        }
        
        result = merge_configs(base, override)
        
        assert result["level1"]["level2"]["level3"]["value"] == "override"
    
    def test_merge_configs_empty_override(self):
        """Test merging with empty override."""
        base = {"a": 1, "b": 2}
        override = {}
        
        result = merge_configs(base, override)
        
        assert result == base
    
    def test_merge_configs_empty_base(self):
        """Test merging with empty base."""
        base = {}
        override = {"a": 1, "b": 2}
        
        result = merge_configs(base, override)
        
        assert result == override


class TestGetDefaultConfig:
    """Test cases for get_default_config function."""
    
    def test_get_default_config_structure(self):
        """Test default config has required structure."""
        default = get_default_config()
        
        assert "model_name" in default
        assert "endpoint_config" in default
        assert "generation_params" in default
        assert "batch_config" in default
        assert "tokenizer_config" in default
        assert "optimization" in default
    
    def test_get_default_config_values(self):
        """Test default config has reasonable values."""
        default = get_default_config()
        
        assert default["generation_params"]["temperature"] >= 0.0
        assert default["generation_params"]["max_tokens"] > 0
        assert default["batch_config"]["batch_size"] > 0
        assert default["tokenizer_config"]["max_context"] > 0
    
    def test_get_default_config_validation(self):
        """Test default config passes validation."""
        default = get_default_config()
        assert validate_config(default) == True


class TestConfigErrorHandling:
    """Test cases for config error handling."""
    
    def test_config_manager_with_permission_error(self):
        """Test ConfigManager handles permission errors gracefully."""
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            manager = ConfigManager()
            
            with pytest.raises(PermissionError):
                manager.load_config("test_model")
    
    def test_config_manager_with_corrupted_file(self):
        """Test ConfigManager handles corrupted files gracefully."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="\x00\x01\x02")):  # Binary data
                manager = ConfigManager()
                
                with pytest.raises((json.JSONDecodeError, UnicodeDecodeError)):
                    manager.load_config("corrupted_model")
    
    def test_validate_config_with_none_input(self):
        """Test config validation with None input."""
        assert validate_config(None) == False
    
    def test_validate_config_with_non_dict_input(self):
        """Test config validation with non-dict input."""
        assert validate_config("not a dict") == False
        assert validate_config(123) == False
        assert validate_config([]) == False
    
    def test_merge_configs_with_none_inputs(self):
        """Test config merging with None inputs."""
        base = {"a": 1}
        
        result = merge_configs(base, None)
        assert result == base
        
        result = merge_configs(None, base)
        assert result == base
        
        result = merge_configs(None, None)
        assert result == {}


# Test fixtures
@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model_name": "test-model",
        "endpoint_config": {
            "base_url": "https://api.test.com",
            "timeout": 60,
            "rate_limit": 10
        },
        "generation_params": {
            "temperature": 0.0,
            "max_tokens": 2048,
            "top_p": 0.95
        },
        "batch_config": {
            "batch_size": 4,
            "max_batch_size": 16
        },
        "tokenizer_config": {
            "encoding": "cl100k_base",
            "max_context": 200000
        },
        "optimization": {
            "use_caching": True,
            "parallel_requests": True
        }
    }


@pytest.fixture
def temp_config_dir():
    """Temporary directory for config files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])