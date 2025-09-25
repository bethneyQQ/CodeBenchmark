# Comprehensive Coding Evaluation Task

A unified single-turn coding evaluation framework that combines simple and complex coding scenarios with multi-language and multi-model support.

## Overview

The `comprehensive_coding` task provides a comprehensive evaluation of coding capabilities across multiple dimensions:

### Task Categories

1. **Simple Coding Tasks** (inspired by python_coding)
   - Code completion
   - Bug fixing
   - Code translation
   - Documentation generation
   - Function implementation

2. **Complex Coding Tasks** (inspired by multi_turn_generic)
   - Full project design and implementation
   - Multi-component system architecture
   - End-to-end solution development

### Key Features

✅ **Multi-Language Support** - Python, JavaScript, Java, C++, Go  
✅ **Multi-Model Backend** - Claude, OpenAI, DeepSeek, Anthropic, Universal  
✅ **Context-Aware Evaluation** - With/without context scenarios  
✅ **Industry-Standard Metrics** - CodeBLEU, ROUGE, Pass@K, Execution Success  
✅ **Comparative Analysis** - Built-in model comparison tools  

## Supported Languages

| Language | Simple Tasks | Complex Tasks | Execution Testing |
|----------|--------------|---------------|-------------------|
| Python | ✅ | ✅ | ✅ |
| JavaScript | ✅ | ✅ | ✅ |
| Java | ✅ | ✅ | ✅ |
| C++ | ✅ | ✅ | ❌ |
| Go | ✅ | ✅ | ✅ |

## Task Structure

### Simple Tasks (5 categories × 5 languages = 25 tasks)
- **Code Completion**: Fill in missing code parts
- **Bug Fixing**: Identify and fix code errors
- **Code Translation**: Convert between programming languages
- **Documentation**: Generate comments and docstrings
- **Function Implementation**: Create functions from specifications

### Complex Tasks (3 categories × 5 languages = 15 tasks)
- **System Design**: Design and implement complete systems
- **API Development**: Create full REST APIs with documentation
- **Algorithm Implementation**: Implement complex algorithms with optimization

## Metrics

### Code Quality Metrics
- **CodeBLEU**: Industry-standard code similarity metric
- **ROUGE-L**: Longest common subsequence for documentation
- **Edit Distance**: Minimal changes needed for correctness
- **Syntax Validity**: Syntactic correctness of generated code

### Functional Metrics
- **Pass@K**: Functional correctness (K=1,5,10)
- **Execution Success**: Code runs without errors
- **Test Coverage**: Percentage of test cases passed
- **Performance Score**: Runtime efficiency evaluation

### Context Adherence Metrics
- **Context Utilization**: How well context is used
- **Style Compliance**: Adherence to coding standards
- **Security Awareness**: Security best practices followed

## Usage

### Basic Evaluation
```bash
# Evaluate all tasks with default model
lm_eval --model hf --model_args pretrained=model-name --tasks comprehensive_coding_suite

# Evaluate specific language
lm_eval --model hf --model_args pretrained=model-name --tasks comprehensive_coding_python

# Evaluate with context comparison
lm_eval --model hf --model_args pretrained=model-name --tasks comprehensive_coding_context_comparison
```

### Multi-Model Comparison
```bash
# Claude Code SDK
lm_eval --model claude-code-local --model_args model=claude-3-sonnet-20240229 --tasks comprehensive_coding_suite --output_path results/claude_comprehensive.json

# OpenAI
lm_eval --model openai-completions --model_args model=gpt-4-turbo --tasks comprehensive_coding_suite --output_path results/openai_comprehensive.json

# DeepSeek
lm_eval --model deepseek --model_args model=deepseek-v3.1 --tasks comprehensive_coding_suite --output_path results/deepseek_comprehensive.json
```

### Results Analysis
```bash
cd lm_eval/tasks/comprehensive_coding
python analyze_comprehensive_results.py results/claude_comprehensive.json results/openai_comprehensive.json results/deepseek_comprehensive.json
```

## Dataset Structure

Each problem includes:
- **Problem description** in multiple languages
- **Context information** (optional)
- **Expected outputs** with test cases
- **Evaluation criteria** specific to the task
- **Difficulty level** (Easy, Medium, Hard)

## Context Scenarios

### With Context
- Company coding standards
- Existing codebase patterns
- Specific framework requirements
- Performance constraints

### Without Context
- Minimal problem description
- No additional guidance
- Tests pure coding ability
- Baseline performance measurement

## Installation

```bash
cd lm_eval/tasks/comprehensive_coding
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run simple evaluation
lm_eval --model hf --model_args pretrained=your-model --tasks comprehensive_coding_python_simple --limit 5

# 3. Analyze results
python analyze_comprehensive_results.py results/
```