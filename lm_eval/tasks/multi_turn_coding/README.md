# Multi-Turn Coding Evaluation Framework 🚀

A comprehensive evaluation framework for testing multiple model backends on multi-turn software engineering workflows.

## 🆕 Universal Model Support (v2.0)

This framework now supports multiple model backends with a unified interface:

- **🤖 Claude Code SDK** - Enhanced coding with file system access
- **⚡ DeepSeek Models** - Via Alibaba Cloud DashScope API  
- **🧠 OpenAI Models** - GPT-3.5, GPT-4 series
- **💬 Anthropic Claude** - Direct API integration
- **🌐 DashScope Models** - Qwen and other Alibaba models
- **🔧 Custom Models** - Easy to extend for new backends

### Quick Model Selection Examples

```bash
# Claude Code (default, most capable for coding tasks)
lm_eval --model claude-code-local --model_args model=claude-3-sonnet-20240229,multi_turn=true --tasks multi_turn_coding_eval_universal

# DeepSeek (excellent for code generation)
lm_eval --model deepseek --model_args model=deepseek-v3.1 --tasks multi_turn_coding_eval_deepseek

# OpenAI GPT-4
lm_eval --model openai-completions --model_args model=gpt-4 --tasks multi_turn_coding_eval_universal

# Anthropic Claude  
lm_eval --model anthropic_llms --model_args model=claude-3-haiku-20240307 --tasks multi_turn_coding_eval_universal
```

## Overview

This evaluation framework tests models through a complete software development lifecycle in a single comprehensive prompt that covers 4 sequential phases:

1. **PRD Generation** - Product Requirements Document creation
2. **Technical Design** - System architecture and API specifications  
3. **Code Implementation** - Complete Python project with tests
4. **Quality Metrics** - Measurable evaluation criteria definition

The model receives a single prompt describing all phases and must complete them sequentially, with file system interactions for each deliverable.

### 🆕 New in v2.1: Enhanced Multi-Model Support
- **Enhanced simple_test.sh** - Now supports 5 different model backends
- **Cross-platform compatibility** - Added Windows PowerShell script (simple_test.ps1)
- **Flexible model selection** - Easy switching between Claude Code, DeepSeek, OpenAI, Anthropic
- **Comprehensive documentation** - Detailed usage guides and troubleshooting

### 🆕 New in v2.0: Easy Difficulty Level
- **5 new beginner-friendly problems** designed for learning and onboarding
- **Educational focus** with clear, simple requirements and expected outcomes
- **Perfect starting point** for testing the evaluation framework
- **Recommended first run**: `./simple_test.sh --difficulty easy`

## Features

- **45 Diverse Problems** - Easy (5), Simple (5), Medium (20), Complex (15) across multiple domains
- **Progressive Difficulty** - From beginner tutorials to enterprise-scale systems
- **File System Integration** - Real file creation and validation
- **Sequential Task Completion** - Single prompt covering all phases
- **Comprehensive Metrics** - 6 evaluation dimensions with file-based analysis
- **Safe Code Execution** - Sandboxed testing of generated code
- **Claude Code SDK** - Enhanced coding capabilities with tool access

### Problem Distribution
| Difficulty | Count | Description | Time Estimate |
|------------|-------|-------------|---------------|
| **Easy**   | 5     | Beginner tutorials, educational projects | 1-3 minutes |
| **Simple** | 5     | Basic single-service applications | 2-5 minutes |
| **Medium** | 20    | Multi-component systems with integrations | 5-15 minutes |
| **Complex** | 15   | Enterprise-scale, advanced requirements | 15-30 minutes |
| **Total**  | **45** | Complete range from learning to production | 1-30 minutes |

## Quick Start

### 🚀 Quick Start Checklist

```bash
# 1. Choose your model backend and install dependencies
cd lm_eval/tasks/multi_turn_coding
python check_dependencies.py

# For Claude Code SDK (recommended for coding tasks):
pip install claude-code-sdk
export ANTHROPIC_API_KEY="your-key"

# For DeepSeek models:  
export DASHSCOPE_API_KEY="your-key"

# For OpenAI models:
export OPENAI_API_KEY="your-key"

# 2. Install missing packages
pip install -r requirements.txt

# 3. Set API key
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"

# 4. Run setup
python setup_evaluation.py --all

# 5. Test with easy problem
./simple_test.sh --difficulty easy

# 6. Analyze results
python quick_results.py results/simple_test_easy
```

## 🔄 Multi-Model Testing (Enhanced simple_test.sh)

The `simple_test.sh` script now supports multiple model backends for easy comparison testing:

### Supported Model Backends

| Backend | Best For | Models Available |
|---------|----------|------------------|
| **claude-code** | File operations, coding tasks | claude-3-haiku-20240307, claude-3-sonnet-20240229, claude-3-opus-20240229 |
| **deepseek** | Cost-effective code generation | deepseek-v3.1, deepseek-v3, deepseek-r1 |
| **openai** | Stable, reliable performance | gpt-4-turbo, gpt-4, gpt-3.5-turbo |
| **anthropic** | Strong reasoning capabilities | claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307 |
| **universal** | Generic compatibility | Any model with proper configuration |

### Quick Multi-Model Examples

```bash
# Test with Claude Code SDK (default, recommended)
./simple_test.sh --difficulty easy

# Test with DeepSeek (cost-effective)
./simple_test.sh --model-backend deepseek --model-name deepseek-v3.1 --difficulty easy

# Test with OpenAI GPT-4 (reliable)
./simple_test.sh --model-backend openai --model-name gpt-4-turbo --difficulty easy

# Test with Anthropic Claude API (reasoning-focused)
./simple_test.sh --model-backend anthropic --model-name claude-3-sonnet-20240229 --difficulty easy

# Compare multiple models on same problem set
./simple_test.sh --model-backend claude-code --difficulty medium --limit 2
./simple_test.sh --model-backend deepseek --difficulty medium --limit 2
./simple_test.sh --model-backend openai --difficulty medium --limit 2
```

### Windows PowerShell Support

For Windows users, use `simple_test.ps1`:

```powershell
# Basic usage
.\simple_test.ps1 -Difficulty easy

# Test different models
.\simple_test.ps1 -ModelBackend deepseek -ModelName deepseek-v3.1 -Difficulty easy
.\simple_test.ps1 -ModelBackend openai -ModelName gpt-4-turbo -Difficulty easy
```

### Environment Setup by Backend

```bash
# Claude Code SDK & Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"

# DeepSeek
export DASHSCOPE_API_KEY="your-dashscope-key"

# OpenAI
export OPENAI_API_KEY="your-openai-key"
```

### Performance Comparison Recommendations

```bash
# Cost-optimized testing
./simple_test.sh --model-backend deepseek --model-name deepseek-v3.1 --limit 5

# Quality-focused testing  
./simple_test.sh --model-backend claude-code --model-name claude-3-sonnet-20240229 --limit 3

# Stability testing
./simple_test.sh --model-backend openai --model-name gpt-4-turbo --limit 10

# Reasoning capability testing
./simple_test.sh --model-backend anthropic --model-name claude-3-opus-20240229 --limit 2
```

For detailed multi-model usage instructions, see [MULTI_MODEL_USAGE.md](MULTI_MODEL_USAGE.md).

### 1. Prerequisites

- **Python 3.8+** (Python 3.12 recommended)
- **Anthropic API Key** (for Claude Code SDK)
- **10GB+ free disk space** (for generated artifacts)
- **Internet connection** (for package downloads and API calls)

### 2. Installation

#### Core Requirements
```bash
# Essential packages for evaluation
pip install claude-code-sdk datasets pytest

# Analysis and visualization packages
pip install seaborn matplotlib pandas numpy

# Additional ML/data packages (if needed)
pip install torch transformers scikit-learn

# Set up environment
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

#### Alternative Installation Methods

**Using conda (recommended for data science environments):**
```bash
# Create new environment (optional)
conda create -n multi-turn-eval python=3.12
conda activate multi-turn-eval

# Install packages via conda
conda install pandas numpy matplotlib seaborn pytest scikit-learn
pip install claude-code-sdk datasets transformers torch

# Set API key
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

**Using requirements.txt (recommended):**
```bash
# Navigate to the task directory
cd lm_eval/tasks/multi_turn_coding

# Install all dependencies at once
pip install -r requirements.txt

# Or install specific subsets
pip install -r requirements.txt --no-deps  # Skip dependency resolution
pip install $(grep -E "^(claude-code-sdk|datasets|pytest)" requirements.txt)  # Core only
```

**Requirements.txt contents:**
- **Core packages**: claude-code-sdk, datasets, pytest
- **Analysis packages**: seaborn, matplotlib, pandas, numpy  
- **ML packages**: torch, transformers, scikit-learn
- **Development tools**: black, flake8, mypy
- **Utilities**: tqdm, pathlib2, huggingface-hub

#### Package Details

| Package | Purpose | Required | Alternative |
|---------|---------|----------|-------------|
| `claude-code-sdk` | Core Claude Code integration | ✅ Required | None |
| `datasets` | Dataset loading and processing | ✅ Required | Manual JSON parsing |
| `pytest` | Code execution testing | ✅ Required | `unittest` |
| `seaborn` | Advanced visualizations | 🟡 Recommended | `matplotlib` only |
| `matplotlib` | Basic plotting and charts | 🟡 Recommended | No visualizations |
| `pandas` | Data analysis and manipulation | 🟡 Recommended | Manual data handling |
| `numpy` | Numerical computations | 🟡 Recommended | Built-in math |
| `torch` | ML model support (if needed) | 🟠 Optional | `tensorflow` |
| `transformers` | Hugging Face models (if needed) | 🟠 Optional | None |
| `scikit-learn` | ML utilities (if needed) | 🟠 Optional | Manual ML |

#### Troubleshooting Installation

**Common Issues:**

1. **Claude Code SDK not found:**
   ```bash
   pip install --upgrade claude-code-sdk
   ```

2. **Seaborn import error:**
   ```bash
   # For conda users
   conda install seaborn
   
   # For pip users
   pip install --upgrade seaborn matplotlib
   ```

3. **Datasets package issues:**
   ```bash
   pip install --upgrade datasets huggingface-hub
   ```

4. **PyTorch installation (platform-specific):**
   ```bash
   # CPU only
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   
   # GPU (CUDA 11.8)
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   
   # GPU (CUDA 12.1)
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   
   # macOS (Apple Silicon)
   pip install torch
   ```

5. **Permission errors:**
   ```bash
   pip install --user claude-code-sdk datasets pytest seaborn matplotlib
   ```

#### Verification

**Automated Dependency Check (Recommended):**
```bash
# Navigate to task directory
cd lm_eval/tasks/multi_turn_coding

# Run comprehensive dependency check
python check_dependencies.py

# Expected output:
# ✅ All required packages are installed!
# ⚠️  Some optional packages missing (install as needed)
```

**Manual Verification:**
```bash
# Test all imports
python -c "
import claude_code_sdk
import datasets
import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
print('✅ All core packages imported successfully!')

try:
    import seaborn as sns
    print('✅ Seaborn available - full visualization support')
except ImportError:
    print('⚠️  Seaborn not available - using matplotlib fallback')

try:
    import torch
    print('✅ PyTorch available - ML model support enabled')
except ImportError:
    print('ℹ️  PyTorch not available - ML features disabled')
"

# Test Claude Code SDK
python -c "
import claude_code_sdk
print('✅ Claude Code SDK version:', claude_code_sdk.__version__ if hasattr(claude_code_sdk, '__version__') else 'installed')
"
```

**Quick Installation Test:**
```bash
# Test the evaluation pipeline
cd lm_eval/tasks/multi_turn_coding
python setup_evaluation.py --validate
./simple_test.sh --help
```

### 3. Setup Evaluation Environment

```bash
# Run full setup
python setup_evaluation.py --all

# Or step by step
python setup_evaluation.py --validate  # Check environment
python setup_evaluation.py --setup     # Create directories
python setup_evaluation.py --config    # Create sample config
python setup_evaluation.py --test      # Run quick test
```

### 4. Run Evaluation

#### Quick Test (Recommended First)
```bash
# Simple test with 1 random problem (full context vs no context)
cd lm_eval/tasks/multi_turn_coding
./simple_test.sh

# Test with specific difficulty levels
./simple_test.sh --difficulty easy      # Super easy problems (5 available)
./simple_test.sh --difficulty simple    # Easy problems (5 available)
./simple_test.sh --difficulty medium    # Medium problems (20 available)
./simple_test.sh --difficulty complex   # Hard problems (15 available)

# Test multiple problems of same difficulty
./simple_test.sh --difficulty easy --limit 2
./simple_test.sh --difficulty simple --limit 2
./simple_test.sh --difficulty medium --limit 3

# Show help for all options
./simple_test.sh --help
```

#### Basic Evaluation
```bash
# Basic evaluation (5 problems, all contexts enabled)
lm_eval \
  --model claude-code-local \
  --model_args model=claude-3-haiku-20240307,multi_turn=true,debug=false \
  --tasks multi_turn_coding_eval_claude_code \
  --output_path results/multi_turn_results.json \
  --log_samples \
  --limit 5 \
  --batch_size 1

# Full evaluation (all 45 problems)
lm_eval \
  --model claude-code-local \
  --model_args model=claude-3-haiku-20240307,multi_turn=true \
  --tasks multi_turn_coding_eval_claude_code \
  --output_path results/full_evaluation.json \
  --log_samples \
  --batch_size 1
```

### 5. Context Configuration

Control which context information is provided to the model in each phase:

```bash
# All contexts enabled (default)
export ENABLE_PRD_CONTEXT=true ENABLE_DESIGN_CONTEXT=true ENABLE_CODE_CONTEXT=true ENABLE_QUALITY_CONTEXT=true
lm_eval \
  --model claude-code-local \
  --model_args model=claude-3-haiku-20240307,multi_turn=true \
  --tasks multi_turn_coding_eval_claude_code \
  --output_path results/full_context_results.json \
  --log_samples \
  --limit 5 \
  --batch_size 1

# No context information (baseline evaluation)
export ENABLE_PRD_CONTEXT=false ENABLE_DESIGN_CONTEXT=false ENABLE_CODE_CONTEXT=false ENABLE_QUALITY_CONTEXT=false
lm_eval \
  --model claude-code-local \
  --model_args model=claude-3-haiku-20240307,multi_turn=true \
  --tasks multi_turn_coding_eval_claude_code \
  --output_path results/no_context_results.json \
  --log_samples \
  --limit 5 \
  --batch_size 1

# Only PRD and design contexts (partial guidance)
export ENABLE_PRD_CONTEXT=true ENABLE_DESIGN_CONTEXT=true ENABLE_CODE_CONTEXT=false ENABLE_QUALITY_CONTEXT=false
lm_eval \
  --model claude-code-local \
  --model_args model=claude-3-haiku-20240307,multi_turn=true \
  --tasks multi_turn_coding_eval_claude_code \
  --output_path results/partial_context_results.json \
  --log_samples \
  --limit 5 \
  --batch_size 1

# Only code context (implementation guidance only)
export ENABLE_PRD_CONTEXT=false ENABLE_DESIGN_CONTEXT=false ENABLE_CODE_CONTEXT=true ENABLE_QUALITY_CONTEXT=false
lm_eval \
  --model claude-code-local \
  --model_args model=claude-3-haiku-20240307,multi_turn=true \
  --tasks multi_turn_coding_eval_claude_code \
  --output_path results/code_context_only_results.json \
  --log_samples \
  --limit 5 \
  --batch_size 1
```

### 6. Debug Mode

```bash
# Enable debug mode to see detailed responses
lm_eval \
  --model claude-code-local \
  --model_args model=claude-3-haiku-20240307,multi_turn=true,debug=true \
  --tasks multi_turn_coding_eval_claude_code \
  --output_path results/debug_results.json \
  --log_samples \
  --limit 1 \
  --batch_size 1
```

## Problem Structure

Each problem includes:

```json
{
  "problem_id": "simple_001",
  "complexity": "simple|medium|complex", 
  "domain": "web_app|data_processing|api|ml_pipeline|devops_tool",
  "problem_description": "Create a task management web application...",
  "prd_context": "Company-specific PRD requirements (optional)",
  "design_context": "Technical constraints and standards (optional)", 
  "code_context": "Coding standards and frameworks (optional)",
  "quality_context": "Quality gates and compliance (optional)"
}
```

## Evaluation Phases

### Phase 1: PRD Generation
- **Input**: Problem description + PRD context
- **Output**: `./output/{problem_id}/prd.md`
- **Evaluation**: Completeness, clarity, user stories, acceptance criteria

### Phase 2: Technical Design  
- **Input**: PRD file + design context
- **Output**: `./output/{problem_id}/design.md`
- **Evaluation**: Architecture coherence, API design, data models

### Phase 3: Code Implementation
- **Input**: Design file + code context  
- **Output**: `./output/{problem_id}/src/` (complete project)
- **Evaluation**: Syntax validity, execution success, project structure

### Phase 4: Quality Metrics
- **Input**: All previous artifacts + quality context
- **Output**: `./output/{problem_id}/quality_metrics.md`
- **Evaluation**: Metric definition quality, measurability

## Metrics

### 1. File Existence Check (0-1)
Verifies all required files and directories were created:
- PRD markdown file
- Design document  
- Source code directory
- Quality metrics file

### 2. PRD Quality (0-1)
Evaluates PRD content for:
- Problem statement clarity
- User stories completeness
- Acceptance criteria definition
- Functional requirements
- Non-functional requirements

### 3. Design Coherence (0-1)
Assesses technical design for:
- System architecture clarity
- API specification completeness
- Data model definition
- Security considerations

### 4. Code Execution Test (0-1)
Tests generated code:
- Syntax validity (70% weight)
- Test execution success (30% weight)
- Error handling
- Performance considerations

### 5. Project Structure Validation (0-1)
Validates Python project structure:
- `requirements.txt` (0.2)
- `__init__.py` (0.1)
- `main.py` (0.2)
- `README.md` (0.1)
- Test files (0.2)
- `setup.py` (0.1)
- Python files present (0.1)

### 6. Integration Test (0-1)
Tests consistency across artifacts:
- PRD → Design keyword overlap (0.3)
- Design → Code concept mapping (0.4)
- Implementation completeness (0.3)

## Configuration

### Model Arguments

```bash
--model_args model=claude-3-haiku-20240307,multi_turn=true,debug=false,permission_mode=bypassPermissions,cwd=./output,allowed_tools="['Bash','Python','FileEditor']"
```

**Model Parameters:**
- `model`: Claude model to use
- `multi_turn`: Enable conversation state (required for this task)
- `debug`: Print detailed responses
- `cwd`: Working directory for file operations
- `allowed_tools`: List of tools Claude can use
- `permission_mode`: File operation permissions (default: "bypassPermissions")

### Context Configuration (Environment Variables)

```bash
# Set context configuration via environment variables
export ENABLE_PRD_CONTEXT=true
export ENABLE_DESIGN_CONTEXT=true
export ENABLE_CODE_CONTEXT=true
export ENABLE_QUALITY_CONTEXT=true
```

**Context Parameters:**
- `ENABLE_PRD_CONTEXT`: Include company-specific PRD requirements (default: true)
- `ENABLE_DESIGN_CONTEXT`: Include technical constraints and standards (default: true)
- `ENABLE_CODE_CONTEXT`: Include coding standards and frameworks (default: true)
- `ENABLE_QUALITY_CONTEXT`: Include quality gates and compliance requirements (default: true)

### Sample Configuration File

```json
{
  "model": "claude-3-haiku-20240307",
  "debug": false,
  "allowed_tools": ["Bash", "Python", "FileEditor"],
  "cwd": "./output", 
  "max_problems": 5,
  "timeout_per_phase": 300,
  "cleanup_between_runs": true,
  "context_config": {
    "enable_prd_context": true,
    "enable_design_context": true,
    "enable_code_context": true,
    "enable_quality_context": true
  }
}
```

## Output Structure

```
./output/
├── simple_001/
│   ├── prd.md
│   ├── design.md
│   ├── quality_metrics.md
│   └── src/
│       ├── main.py
│       ├── requirements.txt
│       ├── tests/
│       └── README.md
├── simple_002/
│   └── ...
└── results/
    ├── multi_turn_results.json
    └── samples_multi_turn_coding_eval_claude_code.jsonl
```

## Project Files

### Testing Scripts
- **`simple_test.sh`** - Multi-model testing script for Linux/macOS
- **`simple_test.ps1`** - Multi-model testing script for Windows PowerShell
- **`quick_results.py`** - Results analysis script
- **`analyze_my_results.py`** - Custom results analysis script

### Documentation
- **`README.md`** - Main documentation (this file)
- **`MULTI_MODEL_USAGE.md`** - Detailed multi-model usage guide
- **`MODEL_CONFIGURATIONS.md`** - Model configuration reference
- **`UNIVERSAL_MODELS_GUIDE.md`** - Universal model support guide

### Configuration Files
- **`multi_turn_coding.yaml`** - Claude Code SDK configuration
- **`multi_turn_coding_deepseek.yaml`** - DeepSeek model configuration
- **`multi_turn_coding_openai.yaml`** - OpenAI model configuration
- **`multi_turn_coding_universal.yaml`** - Universal model configuration

## Domains and Complexity

### Domains (9 problems each)
- **Web Applications**: Full-stack web development
- **Data Processing**: ETL, analytics, streaming
- **APIs**: REST, GraphQL, microservices  
- **ML Pipelines**: Training, serving, MLOps
- **DevOps Tools**: Infrastructure, monitoring, CI/CD

### Complexity Levels
- **Easy (5 problems)**: Beginner-friendly, educational projects
  - Hello World web page
  - Text file word counter
  - Basic calculator API
  - Number guessing game
  - Simple file backup script
  
- **Simple (5 problems)**: Single service, basic features
  - Task management web app
  - CSV data validator
  - REST API for inventory
  - Basic ML regression pipeline
  - Log analyzer tool
  
- **Medium (20 problems)**: Multi-component systems, integrations
  - Collaborative project management platform
  - Distributed fraud detection system
  - API gateway with monitoring
  - MLOps pipeline with A/B testing
  - CI/CD orchestrator
  
- **Complex (15 problems)**: Enterprise-scale, advanced requirements
  - Complete ERP system
  - Autonomous vehicle sensor fusion
  - High-frequency trading platform
  - AI drug discovery platform
  - Quantum computing platform

### Difficulty Selection
```bash
# Test specific difficulty levels
./simple_test.sh --difficulty easy      # 5 problems available (beginner)
./simple_test.sh --difficulty simple    # 5 problems available (basic)
./simple_test.sh --difficulty medium    # 20 problems available (intermediate)
./simple_test.sh --difficulty complex   # 15 problems available (advanced)

# Use with metadata for manual evaluation
lm_eval --model claude-code-local --model_args model=claude-3-haiku-20240307,multi_turn=true,permission_mode=bypassPermissions --tasks multi_turn_coding_eval_claude_code --metadata '{"difficulty_filter":"simple"}' --limit 5 --output_path results/simple_only.json
```

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```bash
   export ANTHROPIC_API_KEY="your_key_here"
   # Add to ~/.bashrc or ~/.zshrc for persistence
   echo 'export ANTHROPIC_API_KEY="your_key_here"' >> ~/.bashrc
   ```

2. **Missing Dependencies**
   ```bash
   # Install all required packages
   pip install claude-code-sdk datasets pytest seaborn matplotlib pandas numpy
   
   # Or use conda
   conda install pandas numpy matplotlib seaborn pytest scikit-learn
   pip install claude-code-sdk datasets transformers torch
   ```

3. **Import Errors**
   ```bash
   # Test specific imports
   python -c "import seaborn; print('✅ seaborn OK')"
   python -c "import datasets; print('✅ datasets OK')"
   python -c "import claude_code_sdk; print('✅ claude-code-sdk OK')"
   
   # If imports fail, reinstall
   pip install --upgrade --force-reinstall seaborn datasets claude-code-sdk
   ```

4. **Environment Conflicts**
   ```bash
   # Check Python environment
   which python
   python --version
   pip list | grep -E "(seaborn|datasets|claude)"
   
   # Create clean environment
   conda create -n multi-turn-eval python=3.12
   conda activate multi-turn-eval
   pip install claude-code-sdk datasets pytest seaborn matplotlib pandas
   ```

5. **Permission Errors**
   ```bash
   chmod +x setup_evaluation.py simple_test.sh
   
   # If pip permission issues
   pip install --user claude-code-sdk datasets pytest seaborn matplotlib
   ```

6. **Output Directory Issues**
   ```bash
   python setup_evaluation.py --cleanup --force
   python setup_evaluation.py --setup
   
   # Manual cleanup
   rm -rf ./output/* ./results/*
   mkdir -p ./output ./results
   ```

7. **Analysis Script Errors**
   ```bash
   # Test analysis dependencies
   python -c "
   try:
       import seaborn, matplotlib, pandas
       print('✅ Analysis packages OK')
   except ImportError as e:
       print(f'❌ Missing: {e}')
       print('Install with: pip install seaborn matplotlib pandas')
   "
   
   # Use fallback analysis
   python quick_results.py results/simple_test_easy
   ```

8. **Claude Code SDK Issues**
   ```bash
   # Check SDK version and status
   python -c "
   import claude_code_sdk
   print('SDK imported successfully')
   try:
       print('Version:', claude_code_sdk.__version__)
   except AttributeError:
       print('Version info not available')
   "
   
   # Reinstall if needed
   pip uninstall claude-code-sdk
   pip install claude-code-sdk
   ```

### Debug Commands

```bash
# Validate environment
python setup_evaluation.py --validate

# Quick test (recommended first step)
cd lm_eval/tasks/multi_turn_coding
./simple_test.sh --difficulty easy    # Start with super easy problems

# Test single problem with debug
lm_eval \
  --model claude-code-local \
  --model_args model=claude-3-haiku-20240307,multi_turn=true,debug=true \
  --tasks multi_turn_coding_eval_claude_code \
  --limit 1 \
  --log_samples

# Check generated files
ls -la ./output/*/
```

## Performance Expectations

### Timing (per problem)
- **Easy**: 1-3 minutes
- **Simple**: 2-5 minutes
- **Medium**: 5-15 minutes  
- **Complex**: 15-30 minutes

### Success Rates (Claude Sonnet 4)
| Difficulty | File Creation | Syntax Validity | Code Execution | Integration |
|------------|---------------|-----------------|----------------|-------------|
| Easy       | >98%          | >95%            | >90%           | >85%        |
| Simple     | >95%          | >90%            | >80%           | >75%        |
| Medium     | >90%          | >85%            | >75%           | >70%        |
| Complex    | >85%          | >80%            | >70%           | >65%        |

## Context Configuration Scenarios

The evaluation framework supports flexible context configuration to test different scenarios:

### Scenario 1: Full Context (Default)
**Use Case**: Evaluate model performance with complete company guidance
```bash
lm_eval \
  --model claude-code-local \
  --model_args model=claude-3-haiku-20240307,multi_turn=true \
  --tasks multi_turn_coding_eval_claude_code \
  --task_args enable_prd_context=true,enable_design_context=true,enable_code_context=true,enable_quality_context=true \
  --output_path results/full_context.json \
  --limit 10
```

### Scenario 2: No Context (Baseline)
**Use Case**: Test model's inherent software engineering capabilities without guidance
```bash
lm_eval \
  --model claude-code-local \
  --model_args model=claude-3-haiku-20240307,multi_turn=true \
  --tasks multi_turn_coding_eval_claude_code \
  --task_args enable_prd_context=false,enable_design_context=false,enable_code_context=false,enable_quality_context=false \
  --output_path results/baseline.json \
  --limit 10
```

### Scenario 3: Requirements Only
**Use Case**: Test impact of business context on technical decisions
```bash
lm_eval \
  --model claude-code-local \
  --model_args model=claude-3-haiku-20240307,multi_turn=true \
  --tasks multi_turn_coding_eval_claude_code \
  --task_args enable_prd_context=true,enable_design_context=false,enable_code_context=false,enable_quality_context=false \
  --output_path results/prd_only.json \
  --limit 10
```

### Scenario 4: Technical Constraints Only
**Use Case**: Evaluate adherence to technical standards without business context
```bash
lm_eval \
  --model claude-code-local \
  --model_args model=claude-3-haiku-20240307,multi_turn=true \
  --tasks multi_turn_coding_eval_claude_code \
  --task_args enable_prd_context=false,enable_design_context=true,enable_code_context=true,enable_quality_context=true \
  --output_path results/technical_only.json \
  --limit 10
```

### Scenario 5: Progressive Context
**Use Case**: Test how context affects each phase individually
```bash
# Phase 1: PRD with context
lm_eval \
  --model claude-code-local \
  --model_args model=claude-3-haiku-20240307,multi_turn=true \
  --tasks multi_turn_coding_eval_claude_code \
  --task_args enable_prd_context=true,enable_design_context=false,enable_code_context=false,enable_quality_context=false \
  --output_path results/progressive_prd.json \
  --limit 5

# Phase 2: PRD + Design with context
lm_eval \
  --model claude-code-local \
  --model_args model=claude-3-haiku-20240307,multi_turn=true \
  --tasks multi_turn_coding_eval_claude_code \
  --task_args enable_prd_context=true,enable_design_context=true,enable_code_context=false,enable_quality_context=false \
  --output_path results/progressive_design.json \
  --limit 5
```

### Context Impact Analysis

Compare results across different context configurations:

```bash
# Run all context scenarios automatically
cd lm_eval/tasks/multi_turn_coding
./run_context_comparison.sh

# Analyze and visualize results
python analyze_context_impact.py \
  --results_dir results/context_comparison \
  --output_format both

# Analyze context impact across difficulty levels
./run_difficulty_analysis.sh

# Or manually specify directories
python analyze_context_impact.py \
  --difficulty_dirs results/simple_test_easy results/simple_test_medium results/simple_test_complex

# View generated reports
ls results/
# - context_impact_analysis.png (bar charts)
# - context_heatmap.png (performance heatmap)  
# - context_impact_report.md (detailed analysis with embedded images)
# - difficulty_context_analysis.png (difficulty comparison charts)
# - difficulty_context_heatmap.png (difficulty impact heatmap)
# - difficulty_context_analysis_report.md (comprehensive difficulty analysis)
```

**Analysis Output:**
- **Performance Comparison**: Side-by-side metrics for all scenarios
- **Improvement Analysis**: Percentage improvements over baseline
- **Visual Charts**: Bar charts and heatmaps showing context impact
- **Detailed Report**: Markdown report with insights and recommendations
- **Difficulty Analysis**: Cross-difficulty context impact comparison

### Difficulty-Based Context Analysis

Analyze how context effectiveness varies across problem difficulty levels:

#### Option 1: Individual Difficulty Testing (Quick)
```bash
# Test individual difficulties (1 problem each)
./simple_test.sh --difficulty easy
./simple_test.sh --difficulty simple  
./simple_test.sh --difficulty medium
./simple_test.sh --difficulty complex

# Analyze existing results
./run_difficulty_analysis.sh
```

#### Option 2: Comprehensive Difficulty Comparison (Recommended)
```bash
# Run systematic comparison (2 problems per difficulty, 2 * 4 * 2 = 16 total)
./run_difficulty_comparison.sh

# Analyze results with detailed insights
python analyze_difficulty_comparison.py

# Generate visualizations and comprehensive report
python analyze_context_impact.py \
  --difficulty_dirs results/difficulty_comparison/easy results/difficulty_comparison/simple \
                   results/difficulty_comparison/medium results/difficulty_comparison/complex
```

**Difficulty Analysis Features:**
- **Systematic Evaluation**: 4 problems per difficulty level (16 total)
- **Full vs No Context**: Direct comparison of context impact
- **Cross-Difficulty Trends**: Understand how context necessity scales with complexity
- **Metric Sensitivity**: Identify which metrics benefit most from context at each level
- **Performance Patterns**: Discover consistent trends across difficulty spectrum
- **Visual Analysis**: Heatmaps and charts showing difficulty-context relationships
- **Actionable Insights**: Data-driven recommendations for context usage

**Expected Patterns:**
- **Easy Problems**: Minimal context impact (model performs well without guidance)
- **Medium Problems**: Moderate context benefit (context helps with complexity)
- **Complex Problems**: High context impact (context essential for good performance)

### Recommended Evaluation Workflow

#### 🚀 Quick Start (5 minutes)
```bash
# 1. Test single easy problem
./simple_test.sh --difficulty easy

# 2. View results
python quick_results.py results/simple_test_easy
```

#### 📊 Standard Evaluation (30 minutes)
```bash
# 1. Test each difficulty level
./simple_test.sh --difficulty easy --limit 2
./simple_test.sh --difficulty simple --limit 2
./simple_test.sh --difficulty medium --limit 2
./simple_test.sh --difficulty complex --limit 2

# 2. Analyze difficulty trends
./run_difficulty_analysis.sh
```

#### 🔬 Comprehensive Analysis (2-3 hours)
```bash
# 1. Run systematic comparison
./run_difficulty_comparison.sh

# 2. Analyze results
python analyze_difficulty_comparison.py

# 3. Generate visualizations
python analyze_context_impact.py \
  --difficulty_dirs results/difficulty_comparison/easy \
                   results/difficulty_comparison/simple \
                   results/difficulty_comparison/medium \
                   results/difficulty_comparison/complex

# 4. View comprehensive report
open results/difficulty_context_analysis_report.md
```

## Advanced Usage

### Custom Problem Sets

Create custom `problems.jsonl`:

```json
[
  {
    "problem_id": "custom_001",
    "complexity": "medium",
    "domain": "web_app", 
    "problem_description": "Your custom problem...",
    "prd_context": null,
    "design_context": "Use FastAPI framework...",
    "code_context": "Include comprehensive tests...",
    "quality_context": "Code coverage >90%..."
  }
]
```

### Custom Metrics

Add to `custom_metrics.py`:

```python
def custom_security_check(references: List[str], predictions: List[List[str]]) -> Dict[str, float]:
    """Custom security validation metric."""
    # Your implementation
    return {'custom_security_check': score}
```

### Batch Processing

```bash
# Process multiple model configurations
for model in claude-3-haiku-20240307 claude-3-5-sonnet-20241022; do
  lm_eval \
    --model claude-code-local \
    --model_args model=$model,multi_turn=true \
    --tasks multi_turn_coding_eval_claude_code \
    --output_path results/${model}_results.json \
    --limit 10
done

# Process multiple context configurations
contexts=(
  "enable_prd_context=true,enable_design_context=true,enable_code_context=true,enable_quality_context=true"
  "enable_prd_context=false,enable_design_context=false,enable_code_context=false,enable_quality_context=false"
  "enable_prd_context=true,enable_design_context=false,enable_code_context=false,enable_quality_context=false"
  "enable_prd_context=false,enable_design_context=true,enable_code_context=true,enable_quality_context=true"
)

context_names=("full_context" "no_context" "prd_only" "technical_only")

for i in "${!contexts[@]}"; do
  lm_eval \
    --model claude-code-local \
    --model_args model=claude-3-haiku-20240307,multi_turn=true \
    --tasks multi_turn_coding_eval_claude_code \
    --task_args "${contexts[$i]}" \
    --output_path "results/${context_names[$i]}_results.json" \
    --limit 10
done
```

## Contributing

1. **Add New Problems**: Extend `problems.jsonl` with diverse scenarios
2. **Improve Metrics**: Enhance evaluation functions in `custom_metrics.py`
3. **Add Domains**: Create new problem categories
4. **Optimize Performance**: Improve file I/O and execution safety

## License

This evaluation framework is part of the LM Evaluation Harness project and follows the same licensing terms.

## Quick Reference

### Context Configuration Options

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `ENABLE_PRD_CONTEXT` | Include company-specific PRD requirements | `true` |
| `ENABLE_DESIGN_CONTEXT` | Include technical constraints and standards | `true` |
| `ENABLE_CODE_CONTEXT` | Include coding standards and frameworks | `true` |
| `ENABLE_QUALITY_CONTEXT` | Include quality gates and compliance | `true` |

### Common Command Patterns

```bash
# Full evaluation with all contexts
export ENABLE_PRD_CONTEXT=true ENABLE_DESIGN_CONTEXT=true ENABLE_CODE_CONTEXT=true ENABLE_QUALITY_CONTEXT=true
lm_eval --model claude-code-local --model_args model=claude-3-haiku-20240307,multi_turn=true,permission_mode=bypassPermissions --tasks multi_turn_coding_eval_claude_code --output_path results/full.json --limit 10

# Baseline evaluation (no contexts)
export ENABLE_PRD_CONTEXT=false ENABLE_DESIGN_CONTEXT=false ENABLE_CODE_CONTEXT=false ENABLE_QUALITY_CONTEXT=false
lm_eval --model claude-code-local --model_args model=claude-3-haiku-20240307,multi_turn=true,permission_mode=bypassPermissions --tasks multi_turn_coding_eval_claude_code --output_path results/baseline.json --limit 10

# Debug mode with partial context
export ENABLE_PRD_CONTEXT=true ENABLE_DESIGN_CONTEXT=false ENABLE_CODE_CONTEXT=true ENABLE_QUALITY_CONTEXT=false
lm_eval --model claude-code-local --model_args model=claude-3-haiku-20240307,multi_turn=true,debug=true,permission_mode=bypassPermissions --tasks multi_turn_coding_eval_claude_code --output_path results/debug.json --limit 1
```

### Helper Scripts

| Script | Purpose |
|--------|---------|
| `setup_evaluation.py --all` | Complete environment setup |
| `simple_test.sh` | Quick test with 1 problem (full vs no context) |
| `run_context_comparison.sh` | Run all context scenarios |
| `analyze_context_impact.py` | Generate analysis reports with embedded visualizations |
| `run_difficulty_analysis.sh` | Analyze context impact across difficulty levels |
| `run_difficulty_comparison.sh` | Systematic evaluation across all difficulty levels |
| `analyze_difficulty_comparison.py` | Quick analysis of difficulty comparison results |

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run validation: `python setup_evaluation.py --validate`
3. Enable debug mode for detailed output
4. Review generated files in `./output/` directories
5. Use context comparison tools to understand performance differences