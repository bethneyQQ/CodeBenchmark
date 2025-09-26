"""Unit tests for utils.py module."""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import datasets

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import (
    load_dataset, validate_problem_schema, process_docs, 
    doc_to_text, doc_to_target, apply_context_template,
    filter_by_metadata, load_context_configs
)


class TestLoadDataset:
    """Test cases for load_dataset function."""
    
    def test_load_dataset_file_not_found(self):
        """Test load_dataset raises FileNotFoundError when problems.jsonl doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                load_dataset()
    
    def test_load_dataset_empty_file(self):
        """Test load_dataset handles empty file gracefully."""
        mock_file_content = ""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=mock_file_content)):
                with pytest.raises(ValueError, match="No valid problems found"):
                    load_dataset()
    
    def test_load_dataset_valid_problems(self):
        """Test load_dataset with valid problems."""
        valid_problem = {
            "id": "test_001",
            "title": "Test Problem",
            "language": "python",
            "scenario": "code_completion",
            "difficulty": "simple",
            "context_mode": "no_context",
            "prompt": "Write a function",
            "reference": ["def test(): pass"],
            "tests": [],
            "metadata": {
                "time_limit_s": 10,
                "memory_limit_mb": 100,
                "seed": 1234,
                "author": "test",
                "license": "MIT"
            }
        }
        mock_file_content = json.dumps(valid_problem) + "\n"
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=mock_file_content)):
                with patch('utils.validate_problem_schema', return_value=True):
                    with patch('datasets.Dataset.from_list') as mock_dataset:
                        mock_dataset.return_value = MagicMock()
                        result = load_dataset()
                        mock_dataset.assert_called_once_with([valid_problem])
    
    def test_load_dataset_invalid_json(self):
        """Test load_dataset handles invalid JSON gracefully."""
        mock_file_content = "invalid json\n"
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=mock_file_content)):
                with pytest.raises(ValueError, match="No valid problems found"):
                    load_dataset()


class TestValidateProblemSchema:
    """Test cases for validate_problem_schema function."""
    
    def test_validate_problem_schema_valid(self):
        """Test validation with valid problem schema."""
        valid_problem = {
            "id": "test_001",
            "title": "Test Problem",
            "language": "python",
            "scenario": "code_completion",
            "difficulty": "simple",
            "context_mode": "no_context",
            "prompt": "Write a function",
            "reference": ["def test(): pass"],
            "tests": [],
            "metadata": {
                "time_limit_s": 10,
                "memory_limit_mb": 100,
                "seed": 1234,
                "author": "test",
                "license": "MIT"
            }
        }
        assert validate_problem_schema(valid_problem) == True
    
    def test_validate_problem_schema_missing_required_fields(self):
        """Test validation fails with missing required fields."""
        invalid_problem = {
            "id": "test_001",
            # Missing required fields
        }
        assert validate_problem_schema(invalid_problem) == False
    
    def test_validate_problem_schema_invalid_language(self):
        """Test validation fails with invalid language."""
        invalid_problem = {
            "id": "test_001",
            "title": "Test Problem",
            "language": "invalid_language",
            "scenario": "code_completion",
            "difficulty": "simple",
            "context_mode": "no_context",
            "prompt": "Write a function"
        }
        assert validate_problem_schema(invalid_problem) == False


class TestProcessDocs:
    """Test cases for process_docs function."""
    
    def test_process_docs_basic(self):
        """Test basic document processing."""
        doc = {
            "id": "test_001",
            "prompt": "Write a function",
            "context_mode": "no_context"
        }
        result = process_docs(doc)
        assert "processed_prompt" in result
        assert result["id"] == "test_001"
    
    def test_process_docs_with_context(self):
        """Test document processing with context application."""
        doc = {
            "id": "test_001",
            "prompt": "Write a function",
            "context_mode": "minimal_context"
        }
        with patch('utils.apply_context_template') as mock_apply:
            mock_apply.return_value = doc
            result = process_docs(doc)
            mock_apply.assert_called_once()


class TestDocToText:
    """Test cases for doc_to_text function."""
    
    def test_doc_to_text_basic(self):
        """Test basic text extraction from document."""
        doc = {
            "processed_prompt": "Write a function that adds two numbers"
        }
        result = doc_to_text(doc)
        assert result == "Write a function that adds two numbers"
    
    def test_doc_to_text_missing_prompt(self):
        """Test doc_to_text with missing processed_prompt."""
        doc = {"id": "test_001"}
        result = doc_to_text(doc)
        assert result == ""


class TestDocToTarget:
    """Test cases for doc_to_target function."""
    
    def test_doc_to_target_with_reference(self):
        """Test target extraction with reference implementation."""
        doc = {
            "reference": ["def add(a, b): return a + b"]
        }
        result = doc_to_target(doc)
        assert result == "def add(a, b): return a + b"
    
    def test_doc_to_target_multiple_references(self):
        """Test target extraction with multiple references."""
        doc = {
            "reference": ["def add(a, b): return a + b", "def subtract(a, b): return a - b"]
        }
        result = doc_to_target(doc)
        assert result == "def add(a, b): return a + b"
    
    def test_doc_to_target_no_reference(self):
        """Test target extraction without reference."""
        doc = {"id": "test_001"}
        result = doc_to_target(doc)
        assert result == ""


class TestApplyContextTemplate:
    """Test cases for apply_context_template function."""
    
    def test_apply_context_template_no_context(self):
        """Test context template application for no_context mode."""
        doc = {
            "prompt": "Write a function",
            "context_mode": "no_context"
        }
        with patch('utils.load_context_configs') as mock_load:
            mock_load.return_value = {
                "no_context": {
                    "template": "{{prompt}}",
                    "description": "Pure problem"
                }
            }
            result = apply_context_template(doc, "no_context")
            assert "processed_prompt" in result
    
    def test_apply_context_template_minimal_context(self):
        """Test context template application for minimal_context mode."""
        doc = {
            "prompt": "Write a function",
            "context_mode": "minimal_context"
        }
        with patch('utils.load_context_configs') as mock_load:
            mock_load.return_value = {
                "minimal_context": {
                    "template": "{{prompt}}\n\nRequirements:\n- Follow best practices",
                    "description": "Basic constraints"
                }
            }
            result = apply_context_template(doc, "minimal_context")
            assert "processed_prompt" in result
            assert "Requirements:" in result["processed_prompt"]


class TestFilterByMetadata:
    """Test cases for filter_by_metadata function."""
    
    def test_filter_by_metadata_scenario(self):
        """Test filtering by scenario."""
        mock_dataset = MagicMock()
        mock_dataset.filter.return_value = mock_dataset
        
        filters = {"scenario": "code_completion"}
        result = filter_by_metadata(mock_dataset, filters)
        
        mock_dataset.filter.assert_called_once()
    
    def test_filter_by_metadata_difficulty(self):
        """Test filtering by difficulty."""
        mock_dataset = MagicMock()
        mock_dataset.filter.return_value = mock_dataset
        
        filters = {"difficulty": "simple"}
        result = filter_by_metadata(mock_dataset, filters)
        
        mock_dataset.filter.assert_called_once()
    
    def test_filter_by_metadata_multiple_filters(self):
        """Test filtering by multiple criteria."""
        mock_dataset = MagicMock()
        mock_dataset.filter.return_value = mock_dataset
        
        filters = {"scenario": "code_completion", "difficulty": "simple", "language": "python"}
        result = filter_by_metadata(mock_dataset, filters)
        
        # Should be called multiple times for each filter
        assert mock_dataset.filter.call_count >= 1


class TestLoadContextConfigs:
    """Test cases for load_context_configs function."""
    
    def test_load_context_configs_success(self):
        """Test successful loading of context configurations."""
        mock_config = {
            "no_context": {
                "template": "{{prompt}}",
                "description": "Pure problem"
            }
        }
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(mock_config))):
                result = load_context_configs()
                assert result == mock_config
    
    def test_load_context_configs_file_not_found(self):
        """Test load_context_configs when file doesn't exist."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                load_context_configs()
    
    def test_load_context_configs_invalid_json(self):
        """Test load_context_configs with invalid JSON."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="invalid json")):
                with pytest.raises(json.JSONDecodeError):
                    load_context_configs()


# Test fixtures and mock data
@pytest.fixture
def sample_problem():
    """Sample problem for testing."""
    return {
        "id": "test_001",
        "title": "Test Problem",
        "language": "python",
        "scenario": "code_completion",
        "difficulty": "simple",
        "context_mode": "no_context",
        "prompt": "Write a function that adds two numbers",
        "reference": ["def add(a, b): return a + b"],
        "tests": [
            {
                "type": "unit",
                "file": "test_add.py",
                "cmd": "python -m pytest test_add.py"
            }
        ],
        "metadata": {
            "time_limit_s": 10,
            "memory_limit_mb": 100,
            "seed": 1234,
            "author": "test",
            "license": "MIT"
        }
    }


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    problems = [
        {
            "id": "test_001",
            "scenario": "code_completion",
            "difficulty": "simple",
            "language": "python"
        },
        {
            "id": "test_002", 
            "scenario": "bug_fix",
            "difficulty": "intermediate",
            "language": "javascript"
        }
    ]
    return datasets.Dataset.from_list(problems)


if __name__ == "__main__":
    pytest.main([__file__])