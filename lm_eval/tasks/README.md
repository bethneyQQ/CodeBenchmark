# Tasks

A list of supported tasks and task groupings can be viewed with `lm-eval --tasks list`.

For more information, including a full list of task names and their precise meanings or sources, follow the links
provided to the individual README.md files for each subfolder.

## Recent Updates

- **Multi-Turn Evaluation Framework**: Added comprehensive multi-turn evaluation systems including `multi_turn_generic` for three-phase iterative problem-solving and `multi_turn_scenarios` for flexible conversational evaluation across 8+ scenario types.
- **Python Coding Suite**: Introduced comprehensive Python coding evaluation with 5 task categories, context-aware evaluation, and detailed analysis tools.
- **Multi-Model Support**: Enhanced simple_test.sh with support for Claude Code, DeepSeek, OpenAI, Anthropic, and universal model backends.

## Available Tasks

This repository contains the following evaluation tasks (only tasks with actual implementations are listed):

| Task Family | Description | Language(s) | Usage | Results Analysis |
|-------------|-------------|-------------|-------|------------------|
| [multi_turn_coding](multi_turn_coding/README.md) | Multi-turn coding evaluation testing complete software development lifecycle through PRD → Design → Implementation → Quality phases with file system integration and multiple model backend support. | Python | `lm_eval --model claude-code-local --model_args model=claude-3-haiku-20240307,multi_turn=true --tasks multi_turn_coding_eval_claude_code --limit 1 --output_path results/multi_turn_results.json --log_samples` | `python quick_results.py results/` or `python analyze_context_impact.py` |
| [multi_turn_generic](multi_turn_generic/README.md) | Three-phase multi-turn evaluation system testing iterative problem-solving through Analysis → Design → Implementation phases with comprehensive scoring metrics. | English | `lm_eval --model hf --model_args pretrained=model-name --tasks multi_turn_generic --limit 5 --output_path results/generic_results.json --log_samples --apply_chat_template` | `python analyze_results.py results/` with phase-specific analysis |
| [multi_turn_scenarios](multi_turn_scenarios/README.md) | Flexible multi-turn conversation evaluation framework supporting 8+ scenario types including code review, problem solving, teaching dialogue, and conversational scenarios with full chat template support. | English | `lm_eval --model hf --model_args pretrained=model-name --tasks multi_turn_scenarios.code_review_3_turn --limit 3 --output_path results/scenarios_results.json --log_samples --apply_chat_template` | `python analyze_results.py results/ --recommendations` with scenario-specific insights |
| [python_coding](python_coding/README.md) | Comprehensive Python coding evaluation suite with 5 task categories: code completion, repair, translation, docstring generation, and function generation. Includes context-aware evaluation and comparison tools. | Python | `lm_eval --model hf --model_args pretrained=model-name --tasks python_coding_suite --limit 10 --output_path results/python_coding_results.json --log_samples` | `python analyze_context_impact.py results/` |

## Detailed Task Descriptions

### 1. Multi-Turn Coding (`multi_turn_coding`)

**Purpose**: Evaluates models on complete software development lifecycle through sequential phases.

**Phases**:
1. **PRD Generation** - Create Product Requirements Document
2. **Technical Design** - System architecture and API specifications  
3. **Code Implementation** - Complete Python project with tests
4. **Quality Metrics** - Define measurable evaluation criteria

**Key Features**:
- 45 diverse problems across 5 difficulty levels (Easy, Simple, Medium, Complex)
- Real file system integration with artifact generation
- Multi-model backend support (Claude Code, DeepSeek, OpenAI, Anthropic)
- Cross-platform testing scripts (Linux/macOS and Windows)

**Usage Examples**:
```bash
# Quick test with easy problems (uses simple_test.sh)
./simple_test.sh --difficulty easy

# Test with different model backends
./simple_test.sh --model-backend deepseek --model-name deepseek-v3.1
./simple_test.sh --model-backend openai --model-name gpt-4-turbo

# Full evaluation with Claude Code SDK
lm_eval --model claude-code-local \
    --model_args model=claude-3-sonnet-20240229,multi_turn=true \
    --tasks multi_turn_coding_eval_claude_code \
    --limit 5 \
    --output_path results/multi_turn_claude_results.json \
    --log_samples
```

**Results Analysis**:
```bash
# Quick results overview (run from multi_turn_coding directory)
cd lm_eval/tasks/multi_turn_coding
python quick_results.py results/

# Single model analysis
python compare_models.py results/deepseek_comparison*.json --verbose

# Multi-model comparison (when multiple result files exist)
python compare_models.py results/*_comparison*.json

# Context impact analysis (requires specific file naming)
python analyze_context_impact.py --results_dir results/

# Difficulty comparison analysis
python analyze_difficulty_comparison.py
```

### 2. Multi-Turn Generic (`multi_turn_generic`)

**Purpose**: Tests iterative problem-solving capabilities through structured phases.

**Phases**:
1. **Problem Analysis** - Understanding and breaking down problems
2. **Solution Design** - Creating structured solution approaches
3. **Implementation** - Writing complete, working code

**Key Features**:
- Sequential context-dependent interactions
- Comprehensive scoring across all phases
- Chat template support for instruction-tuned models
- Built-in consistency checking between phases

**Usage Examples**:
```bash
# Basic evaluation
lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks multi_turn_generic \
    --limit 5 \
    --output_path results/generic_evaluation.json \
    --log_samples \
    --apply_chat_template

# With custom parameters
lm_eval --model hf \
    --model_args pretrained=your-model-name,temperature=0.1,max_length=2048 \
    --tasks multi_turn_generic \
    --limit 10 \
    --output_path results/generic_custom.json \
    --log_samples \
    --apply_chat_template
```

**Results Analysis**:
```bash
# Comprehensive phase analysis
python analyze_results.py results/

# Detailed analysis with raw data
python analyze_results.py results/ --verbose
```
- **Built-in Metrics**: `multi_turn_score`, `phase_consistency`, `solution_quality`
- **Phase-Specific Analysis**: Performance breakdown by Analysis, Design, Implementation phases
- **Performance Insights**: Best/worst performing areas with improvement recommendations
- **Automatic Evaluation**: Phase-to-phase consistency checking

### 3. Multi-Turn Scenarios (`multi_turn_scenarios`)

**Purpose**: Flexible framework for diverse multi-turn interaction patterns.

**Scenario Types**:
- **Code Review** - Collaborative code improvement workflows
- **Iterative Problem Solving** - Progressive solution refinement  
- **Teaching Dialogue** - Instructional conversations with assessment
- **Conversational** - Natural multi-turn conversations
- **Workflow** - Structured task completion processes
- **Collaborative** - Multi-party interaction scenarios
- **Debug Sessions** - Interactive debugging workflows
- **Design Iterations** - Iterative design and feedback cycles

**Key Features**:
- Extensible scenario system with registry
- Full chat template integration (ChatML, Llama, Alpaca, etc.)
- Scenario-specific evaluation metrics
- Dynamic scenario discovery

**Usage Examples**:
```bash
# Code review scenario
lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks multi_turn_scenarios.code_review_3_turn \
    --limit 3 \
    --output_path results/code_review_results.json \
    --log_samples \
    --apply_chat_template

# Iterative problem solving
lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks multi_turn_scenarios.iterative_problem_solving \
    --limit 5 \
    --output_path results/problem_solving_results.json \
    --log_samples \
    --apply_chat_template

# Teaching dialogue
lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks multi_turn_scenarios.teaching_dialogue \
    --limit 3 \
    --output_path results/teaching_results.json \
    --log_samples \
    --apply_chat_template
```

**Results Analysis**:
```bash
# Scenario performance analysis
python analyze_results.py results/

# With improvement recommendations
python analyze_results.py results/ --recommendations

# Detailed analysis with raw data
python analyze_results.py results/ --verbose
```
- **Scenario-Specific Metrics** - Tailored evaluation for each interaction type
- **Performance Ranking** - Scenarios ranked by average performance
- **Distribution Analysis** - Performance distribution across scenario types
- **Improvement Recommendations** - Targeted suggestions for weak areas

### 4. Python Coding (`python_coding`)

**Purpose**: Comprehensive evaluation of Python coding capabilities across multiple dimensions.

**Task Categories**:
1. **Code Completion** (`python_code_completion`) - Complete partial code snippets
2. **Code Repair** (`python_code_repair`) - Fix buggy code to make it work
3. **Code Translation** (`python_code_translation`) - Translate code to Python
4. **Docstring Generation** (`python_docstring_generation`) - Generate documentation
5. **Function Generation** (`python_function_generation`) - Create functions from descriptions

**Key Features**:
- 25 problems (5 per category) with real-world scenarios
- Context-aware evaluation with company-specific requirements
- Multiple evaluation metrics per task type
- Aggregate scoring across all categories

**Usage Examples**:
```bash
# Run complete suite
lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks python_coding_suite \
    --limit 25 \
    --output_path results/python_suite_results.json \
    --log_samples

# Run individual tasks
lm_eval --model hf \
    --model_args pretrained=your-model-name \
    --tasks python_code_completion \
    --limit 5 \
    --output_path results/code_completion_results.json \
    --log_samples

# Context comparison evaluation
./run_context_comparison.sh
```

**Results Analysis**:
```bash
# Single model analysis (run from python_coding directory)
cd lm_eval/tasks/python_coding
python analyze_context_impact.py --results_dir results/

# Compare full vs no context performance
./run_full_vs_no_context.sh

# Multi-model comparison
python compare_models.py results/*_results*.json
```

**Evaluation Metrics by Task**:
- **Code Completion**: exact_match, BLEU (128 tokens)
- **Code Repair**: exact_match, edit_distance (256 tokens)  
- **Code Translation**: exact_match, BLEU, CodeBLEU (512 tokens)
- **Docstring Generation**: exact_match, BLEU, ROUGE, METEOR (256 tokens)
- **Function Generation**: exact_match, pass@k (512 tokens)

## General Usage Guidelines

### Quick Start for Any Task

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API Keys** (if using API-based models):
   ```bash
   export ANTHROPIC_API_KEY="your-key"  # For Claude models
   export OPENAI_API_KEY="your-key"     # For OpenAI models
   export DASHSCOPE_API_KEY="your-key"  # For DeepSeek models
   ```

3. **Run Evaluation**:
   ```bash
   # For local models
   lm_eval --model hf --model_args pretrained=your-model --tasks task_name
   
   # For API models with chat templates
   lm_eval --model hf --model_args pretrained=your-model --tasks task_name --apply_chat_template
   ```

4. **Analyze Results**:
   ```bash
   # Each task has its own analysis script
   python analyze_results.py results/
   ```

### Common Parameters

- `--limit N` - Limit number of problems to evaluate
- `--output_path path` - Specify output file location
- `--log_samples` - Save detailed sample outputs
- `--apply_chat_template` - Use chat templates for instruction-tuned models
- `--batch_size N` - Set batch size for evaluation

### Results Location

Results are typically saved to:
- **JSON Results**: `results/task_name_timestamp.json`
- **Sample Outputs**: `samples_task_name_timestamp.jsonl`
- **Generated Artifacts**: `output/` directory (for coding tasks)

### Troubleshooting

1. **Missing Dependencies**: Run `pip install -r requirements.txt` in the task directory
2. **API Key Issues**: Ensure environment variables are set correctly
3. **Memory Issues**: Reduce `--batch_size` or `--limit`
4. **Chat Template Issues**: Verify model supports chat templates or remove `--apply_chat_template`

For task-specific issues, refer to the individual README.md files in each task directory.

## Troubleshooting Common Issues

### 1. Results Analysis Problems

**Issue**: `analyze_context_impact.py` reports "No results found to analyze"
```bash
⚠️  Missing Full Context: ['full_context_test*.json', 'full_context_results*.json']
❌ No results found to analyze
```

**Solution**: Use the flexible `compare_models.py` script instead:
```bash
cd lm_eval/tasks/multi_turn_coding
python compare_models.py results/*.json --verbose
```

**Issue**: Most metrics show 0.000 values
**Cause**: Model failed to complete tasks properly (API issues, configuration problems, etc.)
**Solution**: 
- Check API keys are set correctly
- Verify model has sufficient context length
- Try with a simpler difficulty level: `--difficulty easy`
- Check the sample output files for error messages

### 2. Model Backend Issues

**Issue**: DeepSeek evaluation fails or produces poor results
**Solution**:
```bash
# Ensure API key is set
export DASHSCOPE_API_KEY="your-key"

# Try with explicit model configuration
lm_eval --model deepseek \
    --model_args model=deepseek-v3.1,temperature=0.0 \
    --tasks multi_turn_coding_eval_deepseek \
    --limit 1 \
    --output_path results/deepseek_test.json \
    --log_samples
```

**Issue**: DashScope/Qwen evaluation fails with API key error
**Solution**:
```bash
# Ensure API key is set
export DASHSCOPE_API_KEY="your-key"

# Use universal task for DashScope models
lm_eval --model dashscope \
    --model_args model=qwen-plus \
    --tasks multi_turn_coding_eval_universal \
    --limit 1 \
    --output_path results/qwen_test.json \
    --log_samples
```

**Issue**: Claude Code SDK not working
**Solution**:
```bash
# Install and configure Claude Code SDK
pip install claude-code-sdk
export ANTHROPIC_API_KEY="your-key"

# Test with simple configuration
lm_eval --model claude-code-local \
    --model_args model=claude-3-haiku-20240307,multi_turn=true \
    --tasks multi_turn_coding_eval_claude_code \
    --limit 1 \
    --output_path results/claude_test.json \
    --log_samples
```

### 3. File Path Issues

**Issue**: Results files not found in expected location
**Solution**: Results are saved in task-specific directories:
- Multi-turn coding: `lm_eval/tasks/multi_turn_coding/results/`
- Python coding: `lm_eval/tasks/python_coding/results/`
- Generic/Scenarios: Usually in workspace root `results/`

**Issue**: Analysis scripts can't find files
**Solution**: Always run analysis scripts from the correct directory:
```bash
# For multi-turn coding analysis
cd lm_eval/tasks/multi_turn_coding
python compare_models.py results/*.json

# For python coding analysis  
cd lm_eval/tasks/python_coding
python analyze_context_impact.py --results_dir results/
```

### 4. Missing Dependencies

**Issue**: `ModuleNotFoundError: No module named 'pandas'`
**Solution**:
```bash
pip install pandas matplotlib seaborn numpy
```

**Issue**: `ModuleNotFoundError: No module named 'claude_code_sdk'`
**Solution**:
```bash
pip install claude-code-sdk
```

### 5. Performance Issues

**Issue**: Evaluation takes too long or times out
**Solution**:
- Reduce the number of problems: `--limit 1`
- Use faster models: Claude Haiku instead of Opus
- Use simpler difficulty: `--difficulty easy`
- Check network connectivity for API-based models

## Model Backend Usage Examples

### Claude Code SDK (Recommended for Coding Tasks)

**Setup**:
```bash
pip install claude-code-sdk
export ANTHROPIC_API_KEY="your-anthropic-key"
```

**Usage Examples**:
```bash
# Multi-turn coding with Claude Code SDK
lm_eval --model claude-code-local \
    --model_args model=claude-3-haiku-20240307,multi_turn=true,debug=false \
    --tasks multi_turn_coding_eval_claude_code \
    --limit 3 \
    --output_path results/claude_code_results.json \
    --log_samples

# With Sonnet for better performance
lm_eval --model claude-code-local \
    --model_args model=claude-3-sonnet-20240229,multi_turn=true \
    --tasks multi_turn_coding_eval_claude_code \
    --limit 5 \
    --output_path results/claude_sonnet_results.json \
    --log_samples
```

### Anthropic Claude API

**Setup**:
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
```

**Usage Examples**:
```bash
# Multi-turn generic with Claude API
lm_eval --model anthropic_llms \
    --model_args model=claude-3-haiku-20240307 \
    --tasks multi_turn_generic \
    --limit 5 \
    --output_path results/claude_api_generic.json \
    --log_samples \
    --apply_chat_template

# Multi-turn scenarios with Claude Sonnet
lm_eval --model anthropic_llms \
    --model_args model=claude-3-sonnet-20240229 \
    --tasks multi_turn_scenarios.code_review_3_turn \
    --limit 3 \
    --output_path results/claude_scenarios.json \
    --log_samples \
    --apply_chat_template

# Python coding with Claude Opus
lm_eval --model anthropic_llms \
    --model_args model=claude-3-opus-20240229 \
    --tasks python_coding_suite \
    --limit 10 \
    --output_path results/claude_python_coding.json \
    --log_samples
```

### DeepSeek Models

**Setup**:
```bash
export DASHSCOPE_API_KEY="your-dashscope-key"
```

**Usage Examples**:
```bash
# Multi-turn coding with DeepSeek
lm_eval --model deepseek \
    --model_args model=deepseek-v3.1 \
    --tasks multi_turn_coding_eval_deepseek \
    --limit 5 \
    --output_path results/deepseek_coding.json \
    --log_samples

# Python coding with DeepSeek (cost-effective)
lm_eval --model deepseek \
    --model_args model=deepseek-v3.1 \
    --tasks python_coding_suite \
    --limit 25 \
    --output_path results/deepseek_python.json \
    --log_samples

# Multi-turn generic with DeepSeek R1 (reasoning-focused)
lm_eval --model deepseek \
    --model_args model=deepseek-r1 \
    --tasks multi_turn_generic \
    --limit 10 \
    --output_path results/deepseek_r1_generic.json \
    --log_samples \
    --apply_chat_template
```

### DashScope Models (Alibaba Cloud)

**Setup**:
```bash
export DASHSCOPE_API_KEY="your-dashscope-key"
```

**Usage Examples**:
```bash
# Multi-turn scenarios with Qwen (use universal task for DashScope)
lm_eval --model dashscope \
    --model_args model=qwen-turbo \
    --tasks multi_turn_coding_eval_universal \
    --limit 5 \
    --output_path results/qwen_scenarios.json \
    --log_samples

# Python coding with Qwen Max
lm_eval --model dashscope \
    --model_args model=qwen-max \
    --tasks python_coding_suite \
    --limit 15 \
    --output_path results/qwen_python.json \
    --log_samples

# Multi-turn generic with Qwen Plus
lm_eval --model dashscope \
    --model_args model=qwen-plus \
    --tasks multi_turn_generic \
    --limit 8 \
    --output_path results/qwen_generic.json \
    --log_samples \
    --apply_chat_template
```

### OpenAI Models

**Setup**:
```bash
export OPENAI_API_KEY="your-openai-key"
```

**Usage Examples**:
```bash
# Multi-turn coding with GPT-4 Turbo
lm_eval --model openai-completions \
    --model_args model=gpt-4-turbo \
    --tasks multi_turn_coding_eval_openai \
    --limit 3 \
    --output_path results/gpt4_coding.json \
    --log_samples

# Multi-turn coding with Qwen Plus (DashScope)
lm_eval --model dashscope \
    --model_args model=qwen-plus \
    --tasks multi_turn_coding_eval_universal \
    --limit 3 \
    --output_path results/qwen_coding.json \
    --log_samples

# Python coding with GPT-4
lm_eval --model openai-completions \
    --model_args model=gpt-4 \
    --tasks python_coding_suite \
    --limit 20 \
    --output_path results/gpt4_python.json \
    --log_samples

# Multi-turn scenarios with GPT-3.5 Turbo (cost-effective)
lm_eval --model openai-completions \
    --model_args model=gpt-3.5-turbo \
    --tasks multi_turn_scenarios.teaching_dialogue \
    --limit 5 \
    --output_path results/gpt35_scenarios.json \
    --log_samples \
    --apply_chat_template

# Multi-turn generic with GPT-4 Turbo
lm_eval --model openai-completions \
    --model_args model=gpt-4-turbo \
    --tasks multi_turn_generic \
    --limit 10 \
    --output_path results/gpt4_generic.json \
    --log_samples \
    --apply_chat_template
```

### Model Selection Guidelines

| Use Case | Recommended Model | Reason |
|----------|------------------|---------|
| **Coding Tasks** | Claude Code SDK | Best file system integration and coding capabilities |
| **Cost-Effective Coding** | DeepSeek v3.1 | Excellent code generation at low cost |
| **Reasoning Tasks** | Claude Opus or DeepSeek R1 | Strong reasoning and problem-solving abilities |
| **Stable API** | OpenAI GPT-4 Turbo | Most reliable API with consistent performance |
| **Chinese Language** | Qwen models via DashScope | Native Chinese language support |
| **Batch Processing** | DeepSeek or GPT-3.5 Turbo | Cost-effective for large-scale evaluation |

### Performance Comparison Commands

```bash
# Compare different models on same task
lm_eval --model claude-code-local --model_args model=claude-3-haiku-20240307,multi_turn=true --tasks multi_turn_coding_eval_claude_code --limit 1 --output_path results/claude_comparison.json --log_samples

lm_eval --model deepseek --model_args model=deepseek-v3.1 --tasks multi_turn_coding_eval_deepseek --limit 1 --output_path results/deepseek_comparison.json --log_samples

lm_eval --model dashscope  --model_args model=qwen-plus --tasks multi_turn_coding_eval_openai --limit 1 --output_path results/qwen_comparison.json --log_samples

# Then analyze and compare results (run from multi_turn_coding directory)
cd lm_eval/tasks/multi_turn_coding
python compare_models.py ../../../results/claude_comparison*.json ../../../results/qwen_comparison*.json results/deepseek_comparison*.json
```
  