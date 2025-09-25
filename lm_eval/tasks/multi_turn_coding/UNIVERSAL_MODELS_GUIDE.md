# Universal Model Support Guide

This guide explains how to use different model backends with the multi-turn coding evaluation framework.

## Supported Models

### 1. Claude Code SDK (Recommended for Coding)

**Best for:** File system operations, complex coding tasks, iterative development

```bash
# Install
pip install claude-code-sdk
export ANTHROPIC_API_KEY="your-anthropic-key"

# Run evaluation
lm_eval --model claude-code-local \
        --model_args model=claude-3-sonnet-20240229,multi_turn=true \
        --tasks multi_turn_coding_eval_claude_code \
        --limit 1 \
        --output_path results/claude_results.json

# Available Claude Code models:
# - claude-3-haiku-20240307 (fast, cost-effective)
# - claude-3-sonnet-20240229 (balanced performance)
# - claude-3-opus-20240229 (highest capability)
```

### 2. DeepSeek Models

**Best for:** Code generation, algorithmic problems, cost-effective solutions

```bash
# Setup
export DASHSCOPE_API_KEY="your-dashscope-key"

# Run evaluation  
lm_eval --model deepseek \
        --model_args model=deepseek-v3.1 \
        --tasks multi_turn_coding_eval_deepseek \
        --limit 1 \
        --output_path results/deepseek_results.json

# Available DeepSeek models:
# - deepseek-v3.1 (latest, most capable)
# - deepseek-v3 (previous version)
# - deepseek-r1 (reasoning focused)
```

### 3. OpenAI Models

**Best for:** Well-established performance, broad compatibility

```bash
# Setup
export OPENAI_API_KEY="your-openai-key"

# Run evaluation
lm_eval --model openai-completions \
        --model_args model=gpt-4-turbo \
        --tasks multi_turn_coding_eval_universal \
        --limit 1 \
        --output_path results/openai_results.json

# Available OpenAI models:
# - gpt-4-turbo (latest GPT-4)
# - gpt-4 (standard GPT-4)
# - gpt-3.5-turbo (cost-effective)
```

### 4. Anthropic Claude API

**Best for:** Reasoning tasks, safety-conscious applications

```bash
# Setup
export ANTHROPIC_API_KEY="your-anthropic-key"

# Run evaluation
lm_eval --model anthropic_llms \
        --model_args model=claude-3-sonnet-20240229 \
        --tasks multi_turn_coding_eval_universal \
        --limit 1 \
        --output_path results/anthropic_results.json

# Available Anthropic models:
# - claude-3-opus-20240229 (most capable)
# - claude-3-sonnet-20240229 (balanced)
# - claude-3-haiku-20240307 (fastest)
```

### 5. DashScope Models (Alibaba Cloud)

**Best for:** Chinese language tasks, Qwen model family

```bash
# Setup
export DASHSCOPE_API_KEY="your-dashscope-key"

# Run evaluation
lm_eval --model dashscope \
        --model_args model=qwen-turbo \
        --tasks multi_turn_coding_eval_universal \
        --limit 1 \
        --output_path results/dashscope_results.json

# Available DashScope models:
# - qwen-turbo (fast and efficient)
# - qwen-plus (balanced performance)
# - qwen-max (highest capability)
```

## Model Comparison

| Model Backend | File Operations | Code Quality | Speed | Cost | Best Use Case |
|---------------|-----------------|--------------|--------|------|---------------|
| Claude Code SDK | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Complex software projects |
| DeepSeek | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Cost-effective coding |
| OpenAI GPT-4 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | General purpose |
| Anthropic Claude | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Safety-critical apps |
| DashScope | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Chinese language tasks |

## Configuration Examples

### Custom Generation Parameters

```bash
# High creativity for design tasks
lm_eval --model deepseek \
        --model_args model=deepseek-v3.1,temperature=0.7,max_tokens=4000 \
        --tasks multi_turn_coding_eval_deepseek

# Deterministic for reproducible results
lm_eval --model claude-code-local \
        --model_args model=claude-3-haiku-20240307,temperature=0.0,multi_turn=true \
        --tasks multi_turn_coding_eval_claude_code
```

### Batch Processing Multiple Models

```bash
#!/bin/bash
# compare_models.sh

models=(
    "claude-code-local:model=claude-3-haiku-20240307,multi_turn=true:multi_turn_coding"
    "deepseek:model=deepseek-v3.1:multi_turn_coding_eval_deepseek" 
    "openai-completions:model=gpt-4-turbo:multi_turn_coding_eval_openai"
)

for model_config in "${models[@]}"; do
    IFS=':' read -r model args task <<< "$model_config"
    echo "Running evaluation with $model..."
    
    lm_eval --model "$model" \
            --model_args "$args" \
            --tasks "$task" \
            --limit 3 \
            --output_path "results/${model}_results.json" \
            --log_samples
done
```

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```bash
   # Check environment variables
   echo $ANTHROPIC_API_KEY
   echo $DASHSCOPE_API_KEY
   echo $OPENAI_API_KEY
   ```

2. **Model Not Available**
   ```bash
   # List available models
   lm_eval --tasks list | grep multi_turn_coding
   ```

3. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install claude-code-sdk  # For Claude Code
   pip install openai          # For OpenAI
   pip install anthropic       # For Anthropic
   ```

### Performance Optimization

1. **Use Appropriate Batch Sizes**
   ```bash
   # For models with rate limits
   lm_eval --model deepseek --batch_size 1
   
   # For faster models  
   lm_eval --model claude-code-local --batch_size 5
   ```

2. **Optimize Token Usage**
   ```bash
   # Reduce max tokens for simpler tasks
   lm_eval --model openai-completions \
           --model_args model=gpt-3.5-turbo,max_tokens=2000
   ```

## Extending Support

### Adding a New Model Backend

1. **Update `model_adapter.py`**:
   ```python
   def _setup_your_model(self):
       """Setup your custom model."""
       from your_model_package import YourModel
       self.model = YourModel(**self.model_args)
   ```

2. **Add to the setup method**:
   ```python
   elif self.model_name == "your-model":
       self._setup_your_model()
   ```

3. **Create a task configuration**:
   ```yaml
   # multi_turn_coding_your_model.yaml
   task: multi_turn_coding_eval_your_model
   # ... rest of configuration
   ```

### Testing New Models

```bash
# Test with minimal configuration
python -c "
from model_adapter import create_model_adapter
adapter = create_model_adapter('your-model', {'model': 'your-model-id'})
print(adapter.get_model_info())
"
```

## Best Practices

1. **Start Small**: Use `--limit 1` for initial testing
2. **Choose Appropriate Tasks**: Use model-specific task files when available
3. **Monitor Resources**: Some models require significant memory or API quotas
4. **Compare Results**: Run the same tasks across different models for comparison
5. **Save Results**: Always use `--output_path` to save detailed results

## Support

For issues with specific model backends:
- **Claude Code**: Check Claude Code SDK documentation
- **DeepSeek**: Verify DashScope API access and quotas
- **OpenAI**: Ensure API key has sufficient credits
- **Anthropic**: Check API tier and rate limits

For framework issues, check the main README.md or create an issue in the repository.