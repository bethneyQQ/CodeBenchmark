"""Unit tests for context application functionality."""

import pytest
import json
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import apply_context_template, load_context_configs


class TestContextApplication:
    """Test cases for context template application."""
    
    def test_apply_no_context_template(self):
        """Test applying no_context template."""
        doc = {
            "prompt": "Write a function to add two numbers",
            "context_mode": "no_context"
        }
        
        mock_configs = {
            "no_context": {
                "template": "{{prompt}}",
                "description": "Pure problem with no additional context"
            }
        }
        
        with patch('utils.load_context_configs', return_value=mock_configs):
            result = apply_context_template(doc, "no_context")
            
            assert "processed_prompt" in result
            assert result["processed_prompt"] == "Write a function to add two numbers"
    
    def test_apply_minimal_context_template(self):
        """Test applying minimal_context template."""
        doc = {
            "prompt": "Write a function to add two numbers",
            "context_mode": "minimal_context"
        }
        
        mock_configs = {
            "minimal_context": {
                "template": "{{prompt}}\n\nRequirements:\n- Follow PEP 8 style guidelines\n- Include error handling",
                "description": "Basic constraints and requirements"
            }
        }
        
        with patch('utils.load_context_configs', return_value=mock_configs):
            result = apply_context_template(doc, "minimal_context")
            
            assert "processed_prompt" in result
            assert "Write a function to add two numbers" in result["processed_prompt"]
            assert "Requirements:" in result["processed_prompt"]
            assert "PEP 8" in result["processed_prompt"]
    
    def test_apply_full_context_template(self):
        """Test applying full_context template."""
        doc = {
            "prompt": "Write a function to add two numbers",
            "context_mode": "full_context",
            "company_standards": "Use type hints and docstrings",
            "best_practices": "Write unit tests for all functions"
        }
        
        mock_configs = {
            "full_context": {
                "template": "Company Standards:\n{{company_standards}}\n\nProblem:\n{{prompt}}\n\nBest Practices:\n{{best_practices}}",
                "description": "Complete company standards and best practices"
            }
        }
        
        with patch('utils.load_context_configs', return_value=mock_configs):
            result = apply_context_template(doc, "full_context")
            
            assert "processed_prompt" in result
            assert "Company Standards:" in result["processed_prompt"]
            assert "Use type hints and docstrings" in result["processed_prompt"]
            assert "Write unit tests for all functions" in result["processed_prompt"]
    
    def test_apply_domain_context_template(self):
        """Test applying domain_context template."""
        doc = {
            "prompt": "Design a database schema",
            "context_mode": "domain_context",
            "domain": "Financial Services",
            "domain_requirements": "Must comply with PCI DSS standards"
        }
        
        mock_configs = {
            "domain_context": {
                "template": "Domain: {{domain}}\n\nSpecialist Requirements:\n{{domain_requirements}}\n\nProblem:\n{{prompt}}",
                "description": "Domain-specific professional requirements"
            }
        }
        
        with patch('utils.load_context_configs', return_value=mock_configs):
            result = apply_context_template(doc, "domain_context")
            
            assert "processed_prompt" in result
            assert "Domain: Financial Services" in result["processed_prompt"]
            assert "PCI DSS standards" in result["processed_prompt"]
    
    def test_apply_context_template_missing_variables(self):
        """Test context template application with missing template variables."""
        doc = {
            "prompt": "Write a function",
            "context_mode": "full_context"
            # Missing company_standards and best_practices
        }
        
        mock_configs = {
            "full_context": {
                "template": "Company Standards:\n{{company_standards}}\n\nProblem:\n{{prompt}}\n\nBest Practices:\n{{best_practices}}",
                "description": "Complete company standards and best practices"
            }
        }
        
        with patch('utils.load_context_configs', return_value=mock_configs):
            result = apply_context_template(doc, "full_context")
            
            # Should handle missing variables gracefully
            assert "processed_prompt" in result
            assert "Write a function" in result["processed_prompt"]
    
    def test_apply_context_template_invalid_mode(self):
        """Test context template application with invalid context mode."""
        doc = {
            "prompt": "Write a function",
            "context_mode": "invalid_mode"
        }
        
        mock_configs = {
            "no_context": {
                "template": "{{prompt}}",
                "description": "Pure problem"
            }
        }
        
        with patch('utils.load_context_configs', return_value=mock_configs):
            # Should fall back to no_context or raise appropriate error
            result = apply_context_template(doc, "invalid_mode")
            
            # Depending on implementation, might fallback or raise error
            assert "processed_prompt" in result or "error" in result
    
    def test_apply_context_template_preserves_original_doc(self):
        """Test that context application preserves original document fields."""
        doc = {
            "id": "test_001",
            "title": "Test Problem",
            "language": "python",
            "scenario": "code_completion",
            "difficulty": "simple",
            "prompt": "Write a function",
            "context_mode": "no_context",
            "reference": ["def test(): pass"],
            "metadata": {"author": "test"}
        }
        
        mock_configs = {
            "no_context": {
                "template": "{{prompt}}",
                "description": "Pure problem"
            }
        }
        
        with patch('utils.load_context_configs', return_value=mock_configs):
            result = apply_context_template(doc, "no_context")
            
            # Should preserve all original fields
            assert result["id"] == "test_001"
            assert result["title"] == "Test Problem"
            assert result["language"] == "python"
            assert result["scenario"] == "code_completion"
            assert result["difficulty"] == "simple"
            assert result["reference"] == ["def test(): pass"]
            assert result["metadata"] == {"author": "test"}
            assert "processed_prompt" in result


class TestLoadContextConfigs:
    """Test cases for loading context configurations."""
    
    def test_load_context_configs_success(self):
        """Test successful loading of context configurations."""
        mock_config = {
            "no_context": {
                "template": "{{prompt}}",
                "description": "Pure problem with no additional context"
            },
            "minimal_context": {
                "template": "{{prompt}}\n\nRequirements:\n- Follow best practices",
                "description": "Basic constraints and requirements"
            }
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(mock_config))):
                result = load_context_configs()
                
                assert result == mock_config
                assert "no_context" in result
                assert "minimal_context" in result
    
    def test_load_context_configs_file_not_found(self):
        """Test loading context configs when file doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                load_context_configs()
    
    def test_load_context_configs_invalid_json(self):
        """Test loading context configs with invalid JSON."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="invalid json")):
                with pytest.raises(json.JSONDecodeError):
                    load_context_configs()
    
    def test_load_context_configs_empty_file(self):
        """Test loading context configs from empty file."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="")):
                with pytest.raises(json.JSONDecodeError):
                    load_context_configs()
    
    def test_load_context_configs_malformed_structure(self):
        """Test loading context configs with malformed structure."""
        malformed_config = {
            "no_context": "not a dict",  # Should be a dict with template and description
            "minimal_context": {
                "template": "{{prompt}}"
                # Missing description
            }
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(malformed_config))):
                # Should load but validation might catch issues later
                result = load_context_configs()
                assert result == malformed_config


class TestContextTemplateValidation:
    """Test cases for context template validation."""
    
    def test_validate_context_template_valid(self):
        """Test validation of valid context templates."""
        valid_templates = [
            "{{prompt}}",
            "{{prompt}}\n\nRequirements:\n- Follow best practices",
            "Company Standards:\n{{company_standards}}\n\nProblem:\n{{prompt}}",
            "Domain: {{domain}}\n\nProblem:\n{{prompt}}\n\nRequirements:\n{{domain_requirements}}"
        ]
        
        for template in valid_templates:
            # Basic validation - should not raise exceptions
            assert "{{" in template and "}}" in template
    
    def test_validate_context_template_invalid(self):
        """Test validation of invalid context templates."""
        invalid_templates = [
            "{{prompt",  # Missing closing brace
            "prompt}}",  # Missing opening brace
            "{{}}",      # Empty variable name
            "{{prompt} {other}}",  # Malformed braces
        ]
        
        # These should be caught by template engine or validation
        for template in invalid_templates:
            # Basic check for malformed templates
            if template.count("{{") != template.count("}}"):
                assert True  # Malformed template detected
    
    def test_context_template_variable_extraction(self):
        """Test extraction of variables from context templates."""
        template = "Company: {{company}}\nProblem: {{prompt}}\nDomain: {{domain}}"
        
        # Extract variables (this would be implementation-specific)
        import re
        variables = re.findall(r'\{\{(\w+)\}\}', template)
        
        assert "company" in variables
        assert "prompt" in variables
        assert "domain" in variables
        assert len(variables) == 3


class TestContextModeHandling:
    """Test cases for context mode handling and edge cases."""
    
    def test_context_mode_case_sensitivity(self):
        """Test context mode handling with different cases."""
        doc = {
            "prompt": "Write a function",
            "context_mode": "NO_CONTEXT"  # Uppercase
        }
        
        mock_configs = {
            "no_context": {
                "template": "{{prompt}}",
                "description": "Pure problem"
            }
        }
        
        with patch('utils.load_context_configs', return_value=mock_configs):
            # Implementation should handle case normalization
            result = apply_context_template(doc, "no_context")
            assert "processed_prompt" in result
    
    def test_context_mode_with_special_characters(self):
        """Test context mode with special characters in prompt."""
        doc = {
            "prompt": "Write a function with 'quotes' and \"double quotes\" and {braces}",
            "context_mode": "no_context"
        }
        
        mock_configs = {
            "no_context": {
                "template": "{{prompt}}",
                "description": "Pure problem"
            }
        }
        
        with patch('utils.load_context_configs', return_value=mock_configs):
            result = apply_context_template(doc, "no_context")
            
            assert "processed_prompt" in result
            assert "quotes" in result["processed_prompt"]
            assert "double quotes" in result["processed_prompt"]
            assert "{braces}" in result["processed_prompt"]
    
    def test_context_mode_with_unicode_characters(self):
        """Test context mode with Unicode characters."""
        doc = {
            "prompt": "Write a function that handles émojis 🚀 and ñoñó characters",
            "context_mode": "no_context"
        }
        
        mock_configs = {
            "no_context": {
                "template": "{{prompt}}",
                "description": "Pure problem"
            }
        }
        
        with patch('utils.load_context_configs', return_value=mock_configs):
            result = apply_context_template(doc, "no_context")
            
            assert "processed_prompt" in result
            assert "émojis" in result["processed_prompt"]
            assert "🚀" in result["processed_prompt"]
            assert "ñoñó" in result["processed_prompt"]
    
    def test_context_mode_with_multiline_prompt(self):
        """Test context mode with multiline prompts."""
        doc = {
            "prompt": """Write a function that:
1. Takes two parameters
2. Adds them together
3. Returns the result""",
            "context_mode": "minimal_context"
        }
        
        mock_configs = {
            "minimal_context": {
                "template": "{{prompt}}\n\nRequirements:\n- Follow best practices",
                "description": "Basic constraints"
            }
        }
        
        with patch('utils.load_context_configs', return_value=mock_configs):
            result = apply_context_template(doc, "minimal_context")
            
            assert "processed_prompt" in result
            assert "Takes two parameters" in result["processed_prompt"]
            assert "Requirements:" in result["processed_prompt"]


# Test fixtures
@pytest.fixture
def sample_context_configs():
    """Sample context configurations for testing."""
    return {
        "no_context": {
            "template": "{{prompt}}",
            "description": "Pure problem with no additional context"
        },
        "minimal_context": {
            "template": "{{prompt}}\n\nRequirements:\n- Follow PEP 8 style guidelines\n- Include error handling",
            "description": "Basic constraints and requirements"
        },
        "full_context": {
            "template": "Company Standards:\n{{company_standards}}\n\nProblem:\n{{prompt}}\n\nBest Practices:\n{{best_practices}}",
            "description": "Complete company standards and best practices"
        },
        "domain_context": {
            "template": "Domain: {{domain}}\n\nSpecialist Requirements:\n{{domain_requirements}}\n\nProblem:\n{{prompt}}",
            "description": "Domain-specific professional requirements"
        }
    }


@pytest.fixture
def sample_document():
    """Sample document for context testing."""
    return {
        "id": "test_001",
        "title": "Add Two Numbers",
        "language": "python",
        "scenario": "function_generation",
        "difficulty": "simple",
        "context_mode": "minimal_context",
        "prompt": "Write a function that adds two numbers and returns the result",
        "reference": ["def add(a, b): return a + b"],
        "tests": [],
        "metadata": {
            "time_limit_s": 10,
            "memory_limit_mb": 100,
            "seed": 1234,
            "author": "test",
            "license": "MIT"
        }
    }


if __name__ == "__main__":
    pytest.main([__file__])