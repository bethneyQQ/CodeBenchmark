# Multi-Model Usage Guide

## Overview

The `simple_test.sh` and `simple_test.ps1` scripts now support multiple model backends, allowing you to easily test different AI models on multi-turn coding tasks.

## Supported Model Backends

### 1. Claude Code SDK (Recommended)
- **Backend Name**: `claude-code`
- **Advantages**: Best file system operation support, optimized for coding tasks
- **Model Options**: 
  - `claude-3-haiku-20240307` (fast, cost-effective)
  - `claude-3-sonnet-20240229` (balanced performance)
  - `claude-3-opus-20240229` (highest capability)

### 2. DeepSeek
- **Backend Name**: `deepseek`
- **Advantages**: Cost-effective, strong code generation capabilities
- **Model Options**:
  - `deepseek-v3.1` (latest version)
  - `deepseek-v3` (stable version)
  - `deepseek-r1` (reasoning-focused)

### 3. OpenAI
- **Backend Name**: `openai`
- **Advantages**: Mature and stable, broad compatibility
- **Model Options**:
  - `gpt-4-turbo` (latest GPT-4)
  - `gpt-4` (standard GPT-4)
  - `gpt-3.5-turbo` (cost-effective)

### 4. Anthropic Claude API
- **Backend Name**: `anthropic`
- **Advantages**: Strong reasoning capabilities, high safety
- **Model Options**:
  - `claude-3-opus-20240229` (most capable)
  - `claude-3-sonnet-20240229` (balanced)
  - `claude-3-haiku-20240307` (fastest)

### 5. Universal (Generic Configuration)
- **Backend Name**: `universal`
- **Advantages**: Compatible with any model, safe configuration
- **Use Case**: Testing new models or as fallback option

## Usage

### Linux/macOS (Bash)

```bash
# Basic usage - default Claude Code
./simple_test.sh

# Specify difficulty
./simple_test.sh --difficulty easy

# Use DeepSeek
./simple_test.sh --model-backend deepseek --model-name deepseek-v3.1

# Use OpenAI GPT-4
./simple_test.sh --model-backend openai --model-name gpt-4-turbo

# Use Anthropic Claude API
./simple_test.sh --model-backend anthropic --model-name claude-3-sonnet-20240229

# Test multiple problems
./simple_test.sh --difficulty medium --limit 3 --model-backend deepseek

# Debug mode
./simple_test.sh --debug --model-backend universal
```

### Windows (PowerShell)

```powershell
# Basic usage - default Claude Code
.\simple_test.ps1

# Specify difficulty
.\simple_test.ps1 -Difficulty easy

# Use DeepSeek
.\simple_test.ps1 -ModelBackend deepseek -ModelName deepseek-v3.1

# Use OpenAI GPT-4
.\simple_test.ps1 -ModelBackend openai -ModelName gpt-4-turbo

# Use Anthropic Claude API
.\simple_test.ps1 -ModelBackend anthropic -ModelName claude-3-sonnet-20240229

# Test multiple problems
.\simple_test.ps1 -Difficulty medium -Limit 3 -ModelBackend deepseek

# Debug mode
.\simple_test.ps1 -Debug -ModelBackend universal
```

## Environment Variables

Set the appropriate API keys based on the model backend you're using:

### Claude Code SDK
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### DeepSeek
```bash
export DASHSCOPE_API_KEY="your-dashscope-key"
```

### OpenAI
```bash
export OPENAI_API_KEY="your-openai-key"
```

### Anthropic Claude API
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Performance Comparison Recommendations

### Cost Priority
```bash
# DeepSeek - lowest cost
./simple_test.sh --model-backend deepseek --model-name deepseek-v3.1 --limit 5
```

### Quality Priority
```bash
# Claude Code SDK - best coding capabilities
./simple_test.sh --model-backend claude-code --model-name claude-3-sonnet-20240229 --limit 3
```

### Stability Priority
```bash
# OpenAI - most stable API
./simple_test.sh --model-backend openai --model-name gpt-4-turbo --limit 10
```

### Reasoning Priority
```bash
# Anthropic Claude - strongest reasoning
./simple_test.sh --model-backend anthropic --model-name claude-3-opus-20240229 --limit 2
```

## Results Analysis

All model results can be analyzed using the same tools:

```bash
# Use our custom analysis script
python analyze_my_results.py results/simple_test_easy

# Or use the original quick results script (requires specific file format)
python quick_results.py results/simple_test_easy
```

## Troubleshooting

### 1. Unsupported Model Backend
```
❌ Unsupported model backend: xxx
```
**Solution**: Check backend name spelling. Supported backends: `claude-code`, `deepseek`, `openai`, `anthropic`, `universal`

### 2. API Key Not Set
```
Error: API key not found
```
**Solution**: Set the corresponding environment variable (see Environment Variables section above)

### 3. Missing Dependencies
```
ModuleNotFoundError: No module named 'xxx'
```
**Solution**: Install missing packages
```bash
pip install numpy datasets claude-code-sdk
```

### 4. Permission Issues (Windows)
```
Execution of scripts is disabled on this system
```
**Solution**: 
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Advanced Usage

### Batch Testing Multiple Models
```bash
# Create batch testing script
for backend in claude-code deepseek openai; do
    echo "Testing $backend..."
    ./simple_test.sh --model-backend $backend --difficulty easy --limit 1
done
```

### Custom Output Directories
Scripts automatically create output directories based on difficulty:
- `results/simple_test` (default)
- `results/simple_test_easy` (when difficulty is specified)

### Debug Mode Details
Using `--debug` or `-Debug` parameter shows:
- Detailed problem information
- Context settings
- Full command line
- Execution summary

This enhanced version allows you to easily test and compare different models on multi-turn coding tasks!