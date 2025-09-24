# DashScope Model Integration Guide

This guide explains how to use the DashScope API models with the lm-evaluation-harness.

## Overview

The DashScope model backend enables you to evaluate Alibaba Cloud's Qwen models through the DashScope API. This implementation supports text generation tasks but does not support loglikelihood computations (as the DashScope API doesn't provide logprobs).

## Installation

First, install the DashScope SDK:

```bash
pip install dashscope
```

## Authentication

You need a DashScope API key to use this model. You can obtain one from [Alibaba Cloud DashScope Console](https://dashscope.console.aliyun.com/).

Set your API key in one of two ways:

### Option 1: Environment Variable (Recommended)
```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

### Option 2: Command Line Parameter
```bash
python -m lm_eval --model dashscope --model_args api_key=your-api-key-here,model=qwen-turbo
```

## Usage

### Basic Usage

```bash
python -m lm_eval \
  --model dashscope \
  --model_args model=qwen-turbo \
  --tasks python_code_completion \
  --limit 5
```

### Available Models

The DashScope backend supports various Qwen models:

- `qwen-turbo` - Fast, cost-effective model
- `qwen-plus` - Balanced performance and speed  
- `qwen-max` - Most capable model
- `qwen-max-longcontext` - Extended context length
- Other models as available in the DashScope API

### Model Arguments

You can customize the model behavior with these arguments:

```bash
python -m lm_eval \
  --model dashscope \
  --model_args model=qwen-plus,max_tokens=1024,temperature=0.7,batch_size=4 \
  --tasks your_task
```

Available model arguments:

- `model` (str): DashScope model name (default: "qwen-turbo")
- `api_key` (str): API key (optional if set via environment)
- `max_tokens` (int): Maximum tokens to generate (default: 1024)
- `temperature` (float): Sampling temperature (default: 0.0)
- `batch_size` (int): Batch size for requests (default: 1)

## Supported Tasks

The DashScope model backend supports:

✅ **Generation tasks** (`generate_until`):
- Code completion
- Text generation
- Function generation
- Any task requiring text generation

❌ **Likelihood tasks** (not supported):
- Multiple choice questions requiring logprobs
- Perplexity computation
- Tasks requiring `loglikelihood` or `loglikelihood_rolling`

## Examples

### Code Completion Task

```bash
# Using environment variable for API key
export DASHSCOPE_API_KEY="your-key"
python -m lm_eval \
  --model dashscope \
  --model_args model=qwen-turbo \
  --tasks python_code_completion \
  --output_path results/dashscope_code.json \
  --limit 10
```

### Custom Dataset

```bash
python -m lm_eval \
  --model dashscope \
  --model_args model=qwen-plus,temperature=0.3 \
  --tasks python_code_completion \
  --metadata '{"dataset_path": "my_custom_problems.jsonl"}' \
  --limit 5
```

### Multiple Tasks

```bash
python -m lm_eval \
  --model dashscope \
  --model_args model=qwen-max,max_tokens=512 \
  --tasks python_code_completion,python_function_generation \
  --output_path results/dashscope_eval.json
```

## Performance Considerations

1. **Batch Size**: DashScope processes requests sequentially. Increase `batch_size` for better throughput.

2. **Rate Limits**: The DashScope API has rate limits. The implementation includes automatic retry with exponential backoff.

3. **Cost**: Different models have different pricing. Check the [DashScope pricing page](https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-metering-and-billing) for details.

## Error Handling

The implementation handles common API errors:

- **Rate limit errors**: Automatic retry with exponential backoff
- **API connection errors**: Retry with backoff
- **Authentication errors**: Clear error messages
- **Invalid model names**: DashScope API error messages

## Limitations

1. **No logprobs support**: Tasks requiring loglikelihood computation are not supported
2. **No tokenizer access**: Token counting is approximated
3. **Sequential processing**: Requests are processed one by one (no true batching)

## Troubleshooting

### Common Issues

**Error: "attempted to use 'dashscope' LM type, but package `dashscope` is not installed"**
- Solution: Install the DashScope package: `pip install dashscope`

**Error: "DashScope API key must be provided"**
- Solution: Set the API key via environment variable or command line argument

**Error: "DashScope API error: InvalidApiKey"**
- Solution: Check that your API key is valid and has sufficient permissions

**Error: "Model not found"**
- Solution: Verify the model name is correct and available in your region

### Debug Mode

Enable debug logging to see detailed API interactions:

```bash
export LOGLEVEL=DEBUG
python -m lm_eval --model dashscope --model_args model=qwen-turbo --tasks your_task
```

## Integration with CodeBenchmark

The DashScope model works seamlessly with CodeBenchmark's Python coding tasks:

```bash
# Test with custom dataset
python -m lm_eval \
  --model dashscope \
  --model_args model=qwen-turbo,api_key=your-key \
  --tasks python_code_completion \
  --metadata '{"dataset_path": "my_custom_data.jsonl"}' \
  --output_path results/dashscope_test.json \
  --log_samples \
  --limit 3
```

This enables you to evaluate Qwen models on your custom coding tasks and compare performance across different model backends.