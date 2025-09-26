# Examples

This directory contains minimal working examples for smoke testing and validation of the single_turn_scenarios task.

## Files

- `minimal_evaluation.py` - Minimal example for running a single scenario
- `smoke_test_example.py` - Quick smoke test for validation
- `metadata_filtering_example.py` - Example of metadata filtering usage
- `dashscope_example.py` - Complete DashScope (Qwen models) usage example

## Usage

These examples can be used for:
- Quick validation that the task is working correctly
- Understanding how to use the task programmatically
- Testing different configuration options
- Smoke testing during development

## Running Examples

```bash
# Run minimal evaluation example
python lm_eval/tasks/single_turn_scenarios/examples/minimal_evaluation.py

# Run smoke test
python lm_eval/tasks/single_turn_scenarios/examples/smoke_test_example.py

# Test metadata filtering
python lm_eval/tasks/single_turn_scenarios/examples/metadata_filtering_example.py

# Run DashScope example (requires DASHSCOPE_API_KEY)
export DASHSCOPE_API_KEY="your-dashscope-api-key"
python lm_eval/tasks/single_turn_scenarios/examples/dashscope_example.py
```

## Model-Specific Examples

### DashScope (Qwen Models)

The `dashscope_example.py` provides a comprehensive example of using Alibaba Cloud's DashScope service:

- **Setup Verification**: Checks API key and connectivity
- **Basic Evaluation**: Simple code completion task
- **Model Comparison**: Compares different Qwen models (qwen-coder-plus, qwen-coder, qwen-turbo)
- **Advanced Evaluation**: Complex algorithm implementation with metadata filtering

```bash
# Prerequisites
export DASHSCOPE_API_KEY="your-dashscope-api-key"
pip install requests  # for API connectivity testing

# Run the example
python lm_eval/tasks/single_turn_scenarios/examples/dashscope_example.py
```

Features demonstrated:
- API key validation and connectivity testing
- Multiple Qwen model variants
- Metadata filtering for specific scenarios
- Result analysis and comparison
- Error handling and troubleshooting

## Requirements

- lm_eval installed and configured
- At least one model backend available (OpenAI, Anthropic, DashScope, etc.)
- Docker installed for sandbox execution (optional for basic testing)
- API keys for your chosen model provider:
  - `OPENAI_API_KEY` for OpenAI models
  - `ANTHROPIC_API_KEY` for Anthropic Claude models
  - `DASHSCOPE_API_KEY` for Alibaba Cloud DashScope (Qwen models)

## API Key Setup

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# DashScope (Alibaba Cloud)
export DASHSCOPE_API_KEY="your-dashscope-api-key"

# Or create a .env file
cat > .env << EOF
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
DASHSCOPE_API_KEY=your-dashscope-api-key
EOF
```