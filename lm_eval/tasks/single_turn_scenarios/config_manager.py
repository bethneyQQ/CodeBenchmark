"""
Configuration management for single_turn_scenarios task.
Handles loading, validation, and merging of model configurations.
Includes secure environment variable support for API keys and sensitive data.

Requirements: 12.1, 12.4
"""

import os
import yaml
import logging
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration data class with validation."""
    model_name: str
    model_type: str
    endpoint_config: Dict[str, Any]
    generation_params: Dict[str, Any]
    batch_config: Dict[str, Any]
    tokenizer_config: Dict[str, Any]
    optimization: Dict[str, Any]
    scenario_specific: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_required_fields()
        self._validate_endpoint_config()
        self._validate_generation_params()
        self._validate_batch_config()
        self._validate_tokenizer_config()

    def _validate_required_fields(self):
        """Validate that all required fields are present."""
        required_fields = [
            'model_name', 'model_type', 'endpoint_config', 
            'generation_params', 'batch_config', 'tokenizer_config', 'optimization'
        ]
        for field_name in required_fields:
            if not getattr(self, field_name):
                raise ValueError(f"Required field '{field_name}' is missing or empty")

    def _validate_endpoint_config(self):
        """Validate endpoint configuration parameters."""
        required_keys = ['base_url', 'timeout', 'rate_limit', 'max_retries']
        for key in required_keys:
            if key not in self.endpoint_config:
                raise ValueError(f"Missing required endpoint config key: {key}")
        
        # Validate types and ranges
        if not isinstance(self.endpoint_config['timeout'], (int, float)) or self.endpoint_config['timeout'] <= 0:
            raise ValueError("timeout must be a positive number")
        
        if not isinstance(self.endpoint_config['rate_limit'], (int, float)) or self.endpoint_config['rate_limit'] <= 0:
            raise ValueError("rate_limit must be a positive number")
        
        if not isinstance(self.endpoint_config['max_retries'], int) or self.endpoint_config['max_retries'] < 0:
            raise ValueError("max_retries must be a non-negative integer")

    def _validate_generation_params(self):
        """Validate generation parameters."""
        required_keys = ['temperature', 'max_tokens', 'top_p']
        for key in required_keys:
            if key not in self.generation_params:
                raise ValueError(f"Missing required generation param key: {key}")
        
        # Validate ranges
        temp = self.generation_params['temperature']
        if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
            raise ValueError("temperature must be between 0 and 2")
        
        max_tokens = self.generation_params['max_tokens']
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        
        top_p = self.generation_params['top_p']
        if not isinstance(top_p, (int, float)) or top_p <= 0 or top_p > 1:
            raise ValueError("top_p must be between 0 and 1")

    def _validate_batch_config(self):
        """Validate batch configuration parameters."""
        required_keys = ['batch_size', 'max_batch_size', 'parallel_requests']
        for key in required_keys:
            if key not in self.batch_config:
                raise ValueError(f"Missing required batch config key: {key}")
        
        batch_size = self.batch_config['batch_size']
        max_batch_size = self.batch_config['max_batch_size']
        
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        
        if not isinstance(max_batch_size, int) or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        
        if batch_size > max_batch_size:
            raise ValueError("batch_size cannot be greater than max_batch_size")
        
        if not isinstance(self.batch_config['parallel_requests'], bool):
            raise ValueError("parallel_requests must be a boolean")

    def _validate_tokenizer_config(self):
        """Validate tokenizer configuration parameters."""
        required_keys = ['encoding', 'max_context', 'context_window']
        for key in required_keys:
            if key not in self.tokenizer_config:
                raise ValueError(f"Missing required tokenizer config key: {key}")
        
        max_context = self.tokenizer_config['max_context']
        context_window = self.tokenizer_config['context_window']
        
        if not isinstance(max_context, int) or max_context <= 0:
            raise ValueError("max_context must be a positive integer")
        
        if not isinstance(context_window, int) or context_window <= 0:
            raise ValueError("context_window must be a positive integer")


class SecureConfigurationManager:
    """Manages loading, validation, and merging of model configurations with secure environment variable support."""
    
    # Sensitive configuration keys that should use environment variables
    SENSITIVE_KEYS = {
        'api_key', 'secret_key', 'access_token', 'password', 'token',
        'private_key', 'client_secret', 'auth_token'
    }
    
    # Environment variable patterns for API keys
    ENV_VAR_PATTERNS = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY', 
        'claude': 'ANTHROPIC_API_KEY',
        'deepseek': 'DEEPSEEK_API_KEY',
        'cohere': 'COHERE_API_KEY',
        'huggingface': 'HUGGINGFACE_API_KEY'
    }
    
    def __init__(self, config_dir: Optional[str] = None, env_file: Optional[str] = None):
        """Initialize secure configuration manager.
        
        Args:
            config_dir: Directory containing model configuration files.
                       If None, uses default model_configs directory.
            env_file: Path to .env file for loading environment variables.
                     If None, uses .env in the same directory as this file.
        """
        if config_dir is None:
            # Default to model_configs directory relative to this file
            self.config_dir = Path(__file__).parent / "model_configs"
        else:
            self.config_dir = Path(config_dir)
        
        # Set up environment file path
        if env_file is None:
            self.env_file = Path(__file__).parent / ".env"
        else:
            self.env_file = Path(env_file)
        
        self._config_cache: Dict[str, ModelConfig] = {}
        self._available_configs: Optional[List[str]] = None
        self._env_loaded = False
        
        # Load environment variables
        self._load_environment()
    
    def _load_environment(self):
        """Load environment variables from .env file if it exists."""
        if not self._env_loaded:
            if self.env_file.exists():
                try:
                    load_dotenv(self.env_file)
                    logger.info(f"Loaded environment variables from {self.env_file}")
                except Exception as e:
                    logger.warning(f"Failed to load .env file: {e}")
            else:
                logger.info("No .env file found, using system environment variables")
            
            self._env_loaded = True
    
    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a configuration key contains sensitive data."""
        key_lower = key.lower()
        return any(sensitive in key_lower for sensitive in self.SENSITIVE_KEYS)
    
    def _resolve_environment_variables(self, config_data: Dict[str, Any], model_name: str = "") -> Dict[str, Any]:
        """Resolve environment variables in configuration data.
        
        Args:
            config_data: Configuration dictionary that may contain environment variable references.
            model_name: Name of the model for automatic API key resolution.
        
        Returns:
            Configuration dictionary with environment variables resolved.
        """
        resolved_config = {}
        
        for key, value in config_data.items():
            if isinstance(value, dict):
                # Recursively resolve nested dictionaries
                resolved_config[key] = self._resolve_environment_variables(value, model_name)
            elif isinstance(value, str):
                # Check for environment variable patterns
                resolved_value = self._resolve_env_var_string(value, key, model_name)
                resolved_config[key] = resolved_value
            else:
                resolved_config[key] = value
        
        return resolved_config
    
    def _resolve_env_var_string(self, value: str, key: str, model_name: str) -> str:
        """Resolve environment variables in a string value.
        
        Args:
            value: String value that may contain environment variable references.
            key: Configuration key name.
            model_name: Model name for automatic resolution.
        
        Returns:
            String with environment variables resolved.
        """
        # Pattern 1: ${ENV_VAR} or ${ENV_VAR:default_value}
        env_pattern = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')
        
        def replace_env_var(match):
            env_var = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            
            env_value = os.getenv(env_var, default_value)
            
            # Security check: warn if sensitive data might be in config file
            if self._is_sensitive_key(key) and env_value == default_value and default_value:
                logger.warning(f"Using default value for sensitive key '{key}'. "
                             f"Consider setting environment variable '{env_var}'")
            
            return env_value
        
        resolved_value = env_pattern.sub(replace_env_var, value)
        
        # Pattern 2: Automatic API key resolution for known providers
        if self._is_sensitive_key(key) and resolved_value in ["", "your_api_key_here", "placeholder"]:
            auto_env_var = self._get_auto_env_var(model_name, key)
            if auto_env_var:
                auto_value = os.getenv(auto_env_var)
                if auto_value:
                    logger.info(f"Auto-resolved API key for {model_name} from {auto_env_var}")
                    resolved_value = auto_value
                else:
                    logger.warning(f"API key not found in environment variable {auto_env_var} for {model_name}")
        
        return resolved_value
    
    def _get_auto_env_var(self, model_name: str, key: str) -> Optional[str]:
        """Get automatic environment variable name for API keys.
        
        Args:
            model_name: Name of the model.
            key: Configuration key.
        
        Returns:
            Environment variable name or None if not found.
        """
        model_lower = model_name.lower()
        
        # Check direct matches
        for provider, env_var in self.ENV_VAR_PATTERNS.items():
            if provider in model_lower:
                return env_var
        
        # Check if key contains provider name
        key_lower = key.lower()
        for provider, env_var in self.ENV_VAR_PATTERNS.items():
            if provider in key_lower:
                return env_var
        
        return None
    
    def _validate_sensitive_data_security(self, config_data: Dict[str, Any], config_name: str):
        """Validate that sensitive data is properly secured.
        
        Args:
            config_data: Configuration data to validate.
            config_name: Name of the configuration for logging.
        """
        def check_dict(data: Dict[str, Any], path: str = ""):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                
                if isinstance(value, dict):
                    check_dict(value, current_path)
                elif isinstance(value, str) and self._is_sensitive_key(key):
                    # Check for hardcoded sensitive values
                    if value and not value.startswith("${") and value not in ["", "placeholder", "your_api_key_here"]:
                        # This might be a hardcoded sensitive value
                        if len(value) > 10:  # Likely a real key
                            logger.error(f"Potential hardcoded sensitive data in {config_name}:{current_path}")
                            raise ValueError(f"Sensitive data should not be hardcoded in configuration files. "
                                           f"Use environment variables for {current_path}")
        
        check_dict(config_data)


class ConfigurationManager(SecureConfigurationManager):
    """Manages loading, validation, and merging of model configurations."""
    
    def __init__(self, config_dir: Optional[str] = None, env_file: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory containing model configuration files.
                       If None, uses default model_configs directory.
            env_file: Path to .env file for loading environment variables.
        """
        super().__init__(config_dir, env_file)
        self._config_cache: Dict[str, ModelConfig] = {}
        self._available_configs: Optional[List[str]] = None

    def get_available_configs(self) -> List[str]:
        """Get list of available configuration names.
        
        Returns:
            List of configuration names (without .yaml extension).
        """
        if self._available_configs is None:
            self._available_configs = []
            if self.config_dir.exists():
                for config_file in self.config_dir.glob("*.yaml"):
                    self._available_configs.append(config_file.stem)
        
        return self._available_configs.copy()

    def load_config(self, config_name: str, use_cache: bool = True, validate_security: bool = True) -> ModelConfig:
        """Load and validate a model configuration with secure environment variable resolution.
        
        Args:
            config_name: Name of the configuration (without .yaml extension).
            use_cache: Whether to use cached configuration if available.
            validate_security: Whether to validate security of sensitive data.
        
        Returns:
            Validated ModelConfig instance.
        
        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            ValueError: If configuration is invalid or contains security issues.
            yaml.YAMLError: If YAML parsing fails.
        """
        if use_cache and config_name in self._config_cache:
            return self._config_cache[config_name]
        
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            available = self.get_available_configs()
            raise FileNotFoundError(
                f"Configuration '{config_name}' not found. "
                f"Available configurations: {available}"
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                raise ValueError(f"Configuration file '{config_path}' is empty")
            
            # Validate security before processing
            if validate_security:
                self._validate_sensitive_data_security(config_data, config_name)
            
            # Resolve environment variables
            model_name = config_data.get('model_name', config_name)
            resolved_config = self._resolve_environment_variables(config_data, model_name)
            
            # Create and validate ModelConfig
            model_config = ModelConfig(**resolved_config)
            
            # Cache the validated configuration
            if use_cache:
                self._config_cache[config_name] = model_config
            
            logger.info(f"Successfully loaded configuration: {config_name}")
            return model_config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML in '{config_path}': {e}")
        except TypeError as e:
            raise ValueError(f"Invalid configuration structure in '{config_path}': {e}")

    def merge_configs(self, base_config: str, override_config: Dict[str, Any]) -> ModelConfig:
        """Merge a base configuration with override parameters.
        
        Args:
            base_config: Name of the base configuration to load.
            override_config: Dictionary of parameters to override.
        
        Returns:
            Merged ModelConfig instance.
        """
        base = self.load_config(base_config)
        
        # Convert base config to dictionary for merging
        base_dict = {
            'model_name': base.model_name,
            'model_type': base.model_type,
            'endpoint_config': base.endpoint_config.copy(),
            'generation_params': base.generation_params.copy(),
            'batch_config': base.batch_config.copy(),
            'tokenizer_config': base.tokenizer_config.copy(),
            'optimization': base.optimization.copy(),
            'scenario_specific': base.scenario_specific.copy(),
            'features': base.features.copy(),
            'metadata': base.metadata.copy()
        }
        
        # Deep merge override config
        merged_dict = self._deep_merge(base_dict, override_config)
        
        # Create and validate merged configuration
        return ModelConfig(**merged_dict)

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries.
        
        Args:
            base: Base dictionary.
            override: Override dictionary.
        
        Returns:
            Merged dictionary.
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

    def validate_all_configs(self) -> Dict[str, bool]:
        """Validate all available configurations.
        
        Returns:
            Dictionary mapping config names to validation status (True if valid).
        """
        results = {}
        available_configs = self.get_available_configs()
        
        for config_name in available_configs:
            try:
                self.load_config(config_name, use_cache=False)
                results[config_name] = True
                logger.info(f"Configuration '{config_name}' is valid")
            except Exception as e:
                results[config_name] = False
                logger.error(f"Configuration '{config_name}' is invalid: {e}")
        
        return results

    def get_config_for_scenario(self, config_name: str, scenario: str) -> Dict[str, Any]:
        """Get scenario-specific configuration parameters.
        
        Args:
            config_name: Name of the configuration.
            scenario: Scenario name (e.g., 'code_completion', 'system_design').
        
        Returns:
            Dictionary of scenario-specific parameters, or default parameters if
            scenario-specific ones are not available.
        """
        config = self.load_config(config_name)
        
        # Check for scenario-specific configuration
        if scenario in config.scenario_specific:
            scenario_params = config.scenario_specific[scenario].copy()
        elif 'default' in config.scenario_specific:
            scenario_params = config.scenario_specific['default'].copy()
        else:
            # Use base generation parameters as fallback
            scenario_params = {}
        
        # Merge with base generation parameters
        merged_params = config.generation_params.copy()
        merged_params.update(scenario_params)
        
        return merged_params

    def validate_environment_security(self) -> Dict[str, Any]:
        """Validate security of environment variables and configuration.
        
        Returns:
            Dictionary containing security validation results.
        """
        results = {
            'env_file_exists': self.env_file.exists(),
            'env_vars_loaded': self._env_loaded,
            'missing_api_keys': [],
            'security_issues': [],
            'recommendations': []
        }
        
        # Check for required API keys
        for provider, env_var in self.ENV_VAR_PATTERNS.items():
            if not os.getenv(env_var):
                results['missing_api_keys'].append(env_var)
        
        # Check .env file security
        if self.env_file.exists():
            try:
                # Check file permissions (Unix-like systems)
                import stat
                file_stat = self.env_file.stat()
                if file_stat.st_mode & stat.S_IROTH:
                    results['security_issues'].append(f".env file is readable by others: {self.env_file}")
                if file_stat.st_mode & stat.S_IWOTH:
                    results['security_issues'].append(f".env file is writable by others: {self.env_file}")
            except Exception:
                # File permission checking not available on this system
                pass
        
        # Check if .env is in .gitignore
        gitignore_path = self.env_file.parent.parent.parent.parent / '.gitignore'
        if gitignore_path.exists():
            try:
                gitignore_content = gitignore_path.read_text()
                if '.env' not in gitignore_content:
                    results['security_issues'].append(".env file not found in .gitignore")
            except Exception as e:
                results['security_issues'].append(f"Could not check .gitignore: {e}")
        
        # Generate recommendations
        if results['missing_api_keys']:
            results['recommendations'].append("Set missing API keys in environment variables")
        if results['security_issues']:
            results['recommendations'].append("Fix security issues with .env file")
        if not results['env_file_exists']:
            results['recommendations'].append("Create .env file from .env.template")
        
        return results
    
    def get_secure_config_summary(self, config_name: str) -> Dict[str, Any]:
        """Get a summary of configuration with sensitive data masked.
        
        Args:
            config_name: Name of the configuration.
        
        Returns:
            Dictionary with configuration summary and masked sensitive data.
        """
        config = self.load_config(config_name)
        
        def mask_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively mask sensitive data in configuration."""
            masked = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    masked[key] = mask_sensitive_data(value)
                elif self._is_sensitive_key(key) and isinstance(value, str) and value:
                    # Mask sensitive values
                    if len(value) > 8:
                        masked[key] = f"{value[:4]}...{value[-4:]}"
                    else:
                        masked[key] = "***"
                else:
                    masked[key] = value
            return masked
        
        # Convert config to dict and mask sensitive data
        config_dict = {
            'model_name': config.model_name,
            'model_type': config.model_type,
            'endpoint_config': config.endpoint_config,
            'generation_params': config.generation_params,
            'batch_config': config.batch_config,
            'tokenizer_config': config.tokenizer_config,
            'optimization': config.optimization,
            'scenario_specific': config.scenario_specific,
            'features': config.features,
            'metadata': config.metadata
        }
        
        return mask_sensitive_data(config_dict)

    def clear_cache(self):
        """Clear the configuration cache."""
        self._config_cache.clear()
        self._available_configs = None
        logger.info("Configuration cache cleared")


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager


def load_model_config(config_name: str) -> ModelConfig:
    """Convenience function to load a model configuration.
    
    Args:
        config_name: Name of the configuration to load.
    
    Returns:
        Validated ModelConfig instance.
    """
    return get_config_manager().load_config(config_name)


def get_scenario_config(config_name: str, scenario: str) -> Dict[str, Any]:
    """Convenience function to get scenario-specific configuration.
    
    Args:
        config_name: Name of the configuration.
        scenario: Scenario name.
    
    Returns:
        Dictionary of scenario-specific parameters.
    """
    return get_config_manager().get_config_for_scenario(config_name, scenario)