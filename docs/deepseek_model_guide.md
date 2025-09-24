# DeepSeek Model Integration Guide

This guide explains how to use DeepSeek API models with the lm-evaluation-harness.

## Overview

The DeepSeek model backend enables you to evaluate DeepSeek's models through their OpenAI-compatible API. This implementation supports text generation tasks but does not support loglikelihood computations (as the DeepSeek API doesn't provide logprobs).

## Installation

First, install the OpenAI SDK (DeepSeek uses OpenAI-compatible API):

```bash
pip install openai
```

## Authentication

You need a DeepSeek API key to use this model. You can obtain one from [DeepSeek Platform](https://platform.deepseek.com/).

Set your API key in one of two ways:

### Option 1: Environment Variable (Recommended)
```bash
export DEEPSEEK_API_KEY="your-api-key-here"
```

### Option 2: Command Line Parameter
```bash
python -m lm_eval --model deepseek --model_args api_key=your-api-key-here,model=deepseek-chat
```

## Usage

### Basic Usage

```bash
python -m lm_eval \
  --model deepseek \
  --model_args model=deepseek-chat \
  --tasks python_code_completion \
  --limit 5
```

### Available Models

The DeepSeek backend supports various DeepSeek models:

- `deepseek-chat` - General conversation and reasoning model
- `deepseek-coder` - Specialized code generation and understanding model
- Other models as available in the DeepSeek API

### Model Arguments

You can customize the model behavior with these arguments:

```bash
python -m lm_eval \
  --model deepseek \
  --model_args model=deepseek-coder,max_tokens=1024,temperature=0.7,batch_size=4 \
  --tasks your_task
```

Available model arguments:

- `model` (str): DeepSeek model name (default: "deepseek-chat")
- `api_key` (str): API key (optional if set via environment)
- `max_tokens` (int): Maximum tokens to generate (default: 1024)
- `temperature` (float): Sampling temperature (default: 0.0)
- `batch_size` (int): Batch size for requests (default: 1)

## Supported Tasks

The DeepSeek model backend supports:

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
export DEEPSEEK_API_KEY="your-key"
python -m lm_eval \
  --model deepseek \
  --model_args model=deepseek-coder \
  --tasks python_code_completion \
  --output_path results/deepseek_code.json \
  --log_samples
```

### Function Generation Task

```bash
python -m lm_eval \
  --model deepseek \
  --model_args model=deepseek-coder,temperature=0.1 \
  --tasks python_function_generation \
  --limit 10
```

### Custom Dataset with DeepSeek

```bash
python -m lm_eval \
  --model deepseek \
  --model_args model=deepseek-chat,max_tokens=512 \
  --tasks python_code_completion \
  --metadata '{"dataset_path": "my_custom_problems.jsonl"}' \
  --output_path results/deepseek_custom.json
```

## Error Handling

The DeepSeek model includes comprehensive error handling:

- **API Key Validation**: Clear error messages if API key is missing
- **Retry Logic**: Automatic retry with exponential backoff on transient failures
- **Rate Limit Handling**: Respects API rate limits with proper backoff
- **Graceful Degradation**: Returns empty strings on unrecoverable errors

## Performance Considerations

- **Batch Size**: Start with `batch_size=1` and increase based on your rate limits
- **Rate Limits**: DeepSeek has rate limits - monitor your usage and adjust batch sizes accordingly
- **Context Length**: DeepSeek models support long contexts (up to 32K tokens)

## Troubleshooting

### API Key Issues
```bash
Error: DeepSeek API key must be provided either through 'api_key' parameter or 'DEEPSEEK_API_KEY' environment variable
```
**Solution**: Set your API key using `export DEEPSEEK_API_KEY="your-key"`

### Rate Limit Errors  
```bash
Error calling DeepSeek API: Rate limit exceeded
```
**Solution**: Reduce `batch_size` or add delays between requests

### Model Not Found
```bash
Error: Model 'invalid-model' not found
```
**Solution**: Use valid model names like `deepseek-chat` or `deepseek-coder`

## Integration with CodeBenchmark

DeepSeek integrates seamlessly with CodeBenchmark's Python coding tasks:

```bash
# Code completion with custom datasets
python -m lm_eval \
  --model deepseek \
  --model_args model=deepseek-coder \
  --tasks python_code_completion \
  --metadata '{"dataset_path": "coding_problems.jsonl"}' \
  --limit 20

# Multiple coding tasks
python -m lm_eval \
  --model deepseek \
  --model_args model=deepseek-coder,temperature=0.0 \
  --tasks python_code_completion,python_function_generation,python_code_repair \
  --output_path results/deepseek_coding_eval.json
```

## API Rate Limits

DeepSeek has different rate limits based on your plan:

- **Free Tier**: Limited requests per minute
- **Paid Tier**: Higher rate limits

Monitor your usage and adjust batch sizes accordingly.

## Best Practices

1. **Use appropriate models**: 
   - `deepseek-coder` for coding tasks
   - `deepseek-chat` for general tasks

2. **Set temperature appropriately**:
   - `temperature=0.0` for deterministic code generation
   - `temperature=0.1-0.3` for some creativity while maintaining accuracy

3. **Batch processing**: Start with small batches and increase based on your rate limits

4. **Error handling**: Always use `--log_samples` for debugging and monitoring outputs

## Comparison with Other Models

| Feature | DeepSeek | DashScope (Qwen) | Claude |
|---------|----------|------------------|--------|
| Code Generation | ✅ Excellent | ✅ Good | ✅ Excellent |
| API Cost | 💰 Low | 💰 Medium | 💰 High |
| Rate Limits | ⚡ Moderate | ⚡ Good | ⚡ Conservative |
| Context Length | 📄 32K | 📄 32K+ | 📄 200K |
| Specialized Code Model | ✅ deepseek-coder | ❌ General only | ❌ General only |