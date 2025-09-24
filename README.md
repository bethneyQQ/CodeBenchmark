# Complete Guide: Python Code Quality Evaluation



## 🧪 **Step 1: Verify Task Works (Optional)**

Run the simple test to confirm everything is working:

```bash
python test_task_simple.py
```

## 📖 **Quick Usage Guide**

### Basic Usage with Default Dataset
```bash
# Use python -m lm_eval (recommended)
python -m lm_eval --model claude-local --model_args model=claude-3-5-haiku-20241022 --tasks python_code_completion --limit 1

# With output file
python -m lm_eval --model claude-local --model_args model=claude-3-5-haiku-20241022 --tasks python_code_completion --output_path results/test.json --log_samples --limit 2
```

### Using Custom Dataset (PowerShell)
```powershell
# Custom dataset in task directory
python -m lm_eval --model claude-local --model_args model=claude-3-5-haiku-20241022 --tasks python_code_completion --metadata '{\"dataset_path\": \"my_custom_data.jsonl\"}' --limit 1

# Absolute path to custom dataset
python -m lm_eval --model claude-local --tasks python_code_completion --metadata '{\"dataset_path\": \"C:/path/to/custom_problems.jsonl\"}' --limit 1

# Multiple dataset paths (uses first one)
python -m lm_eval --model claude-local --tasks python_code_completion --metadata '{\"dataset_paths\": [\"dataset1.jsonl\", \"dataset2.jsonl\"]}' --limit 1
```

### Using Custom Dataset (Bash/Unix)
```bash
# Custom dataset in task directory
python -m lm_eval --model claude-local --model_args model=claude-3-5-haiku-20241022 --tasks python_code_completion --metadata '{"dataset_path": "my_custom_data.jsonl"}' --limit 1

# Absolute path to custom dataset
python -m lm_eval --model claude-local --tasks python_code_completion --metadata '{"dataset_path": "/path/to/custom_problems.jsonl"}' --limit 1
```

## 📄 **Custom Dataset Format**

### Mixed Dataset (Recommended)
Your custom dataset can contain multiple task types in a single JSONL file. Each task will automatically filter for its category:

```json
{"category": "code_completion", "incomplete_code": "def factorial(n):\n    # Complete this function", "expected_completion": "return 1 if n <= 1 else n * factorial(n-1)", "context": "Mathematical functions"}
{"category": "code_repair", "buggy_code": "def divide(a, b):\n    return a / b", "error_description": "No zero division handling", "fixed_code": "def divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b"}
{"category": "function_generation", "function_description": "Create a function to check if a string is a palindrome", "expected_function": "def is_palindrome(s):\n    return s.lower() == s.lower()[::-1]"}
```

### Task-Specific Fields Required

**Code Completion Tasks** (`python_code_completion`, `python_code_completion_*`):
- `category`: "code_completion"  
- `incomplete_code`: Partial function code
- `expected_completion`: Expected completion

**Code Repair Tasks** (`python_code_repair`, `python_code_repair_*`):
- `category`: "code_repair"
- `buggy_code`: Broken code to fix
- `error_description`: Description of the issue  
- `fixed_code`: Corrected code

**Function Generation Tasks** (`python_function_generation`):
- `category`: "function_generation"
- `function_description`: What the function should do
- `expected_function`: Complete function implementation

**Docstring Generation Tasks** (`python_docstring_generation`):
- `category`: "docstring_generation" 
- `function_code`: Function without docstring
- `expected_docstring`: Expected documentation

**Code Translation Tasks** (`python_code_translation`):
- `category`: "code_translation"
- `source_language`: Original language (e.g., "JavaScript")
- `source_code`: Code in original language
- `target_code`: Python equivalent

## 🎯 **Available Tasks**

- `python_code_completion` - Complete partial functions
- `python_code_repair` - Fix broken code
- `python_code_completion_minimal_context` - Minimal context completion
- `python_code_completion_no_context` - No context completion
- `python_code_repair_no_context` - Repair without context
- `python_code_translation` - Code translation tasks
- `python_docstring_generation` - Generate documentation
- `python_function_generation` - Generate complete functions

## 🤖 **Supported Model Backends**

- **Claude (Anthropic)**: `claude-local` - Direct API access
- **Claude Code CLI**: `anthropic-chat` - CLI-based access  
- **DashScope (Qwen)**: `dashscope` - Alibaba Cloud's Qwen models
- **DeepSeek**: `deepseek` - DeepSeek's specialized coding models
- **Local Models**: `local-chat-completions`, `local-completions` - LM Studio/OpenAI-compatible APIs
- **Other Models**: Any model supported by lm-evaluation-harness

## 🚀 **Step 2: Run with Your Local Model**

Run with your local Qwen model, managed by LM_Studio:

### Option A: Chat Completions API
```bash
python -m lm_eval \
  --model local-chat-completions \
  --model_args base_url=http://localhost:1234/v1/chat/completions,model=qwen/qwen3-1.7b \
  --tasks python_code_completion \
  --output_path output_code_results/qwen_code_results.json \
  --log_samples \
  --limit 2 \
  --batch_size 1 \
  --apply_chat_template
```

### Option B: Completions API
```bash
python -m lm_eval \
  --model local-completions \
  --model_args base_url=http://localhost:1234/v1/completions,model=qwen/qwen3-1.7b \
  --tasks python_code_completion \
  --output_path output_code_results/qwen_code_results.json \
  --log_samples \
  --limit 2 \
  --batch_size 1
```

## 🔬 **Step 3: Test with Hosted Model (Recommended)**

Test with a remote model to verify results:

### Option A: Claude (Anthropic) - Recommended for Code
```bash
export ANTHROPIC_API_KEY="your_key_here"

python -m lm_eval \
  --model claude-local \
  --model_args model=claude-3-5-haiku-20241022 \
  --tasks python_code_completion \
  --output_path test_results/claude_test.json \
  --log_samples \
  --limit 2 \
  --batch_size 1
```

### Option B: DeepSeek - Specialized Code Model
```bash
export DEEPSEEK_API_KEY="your_key_here"

python -m lm_eval \
  --model deepseek \
  --model_args model=deepseek-coder \
  --tasks python_code_completion \
  --output_path test_results/deepseek_test.json \
  --log_samples \
  --limit 2 \
  --batch_size 1
```

### Option B: Claude with Custom Dataset  
```bash
export ANTHROPIC_API_KEY="your_key_here"

python -m lm_eval \
  --model claude-local \
  --model_args model=claude-3-5-haiku-20241022 \
  --tasks python_code_completion \
  --metadata '{"dataset_path": "my_custom_data.jsonl"}' \
  --output_path test_results/claude_custom_test.json \
  --log_samples \
  --limit 2 \
  --batch_size 1
```

### Option C: DashScope (Qwen Models)
```bash
export DASHSCOPE_API_KEY="your_api_key_here"

python -m lm_eval \
  --model dashscope \
  --model_args model=qwen-turbo \
  --tasks python_code_completion \
  --output_path test_results/dashscope_test.json \
  --log_samples \
  --limit 2
```

## 🔧 **Configurable Dataset Paths**

You can specify custom dataset paths for the evaluation:

```bash
# Use a custom dataset file
lm_eval --model anthropic-chat --tasks python_code_completion --metadata '{"dataset_path": "path/to/custom_problems.jsonl"}'

# Use multiple dataset files
lm_eval --model anthropic-chat --tasks python_code_completion --metadata '{"dataset_paths": ["path1.jsonl", "path2.jsonl"]}'
```

### Dataset Format
Your custom dataset should be a JSONL file with this structure:
```json
{
    "problem_id": "001",
    "problem_description": "Implement a function to calculate factorial",
    "function_signature": "def factorial(n):",
    "test_cases": ["factorial(5) == 120", "factorial(0) == 1"],
    "difficulty": "easy",
    "tags": ["math", "recursion"]
}
```

##  **Understanding Results**

### Output Files
```
output_code_results/
├── qwen_code_quality_results.json           # Main metrics
├── samples_python_code_quality.jsonl       # Individual responses
└── configs/
    └── python_code_quality.yaml            # Task config
```

### Metrics Explained
- **Code Quality Score** (0-1): Composite score combining functionality and style
- **Functional Correctness** (0-1): Percentage of test cases that pass
- **Syntax Validity** (0-1): Percentage of syntactically valid code
- **Code Style Score** (0-1): Adherence to Python best practices

### View Results
```bash
# Main metrics summary
cat output_code_results/qwen_code_quality_results.json

# Individual model responses with scores
cat output_code_results/samples_python_code_quality.jsonl

# Extract just the metric values
grep -E "(code_quality_score|functional_correctness|syntax_validity|code_style_score)" output_code_results/qwen_code_quality_results.json
```

## **Test Problems (First 2 with --limit 2)**

1. **Factorial Function**
   - Calculate n! with proper error handling
   - Tests: `factorial(0) == 1`, `factorial(5) == 120`, `factorial(1) == 1`

2. **Palindrome Checker**
   - Detect palindromes ignoring case and spaces
   - Tests: `is_palindrome('racecar') == True`, etc.

## 🔧 **Troubleshooting**

### Common Issues
- **"lm_eval command not found"**: Use `python -m lm_eval` instead of `lm_eval`
- **"task not found"**: Run from repo root directory and check task names
- **API key errors**: Ensure `ANTHROPIC_API_KEY` environment variable is set
- **JSON metadata errors**: Use proper escaping for PowerShell: `'{\"key\": \"value\"}'`
- **Dataset not found**: Ensure dataset file is in the task directory or use absolute path
- **Connection errors**: Check local server is running (for local models)

### Quick Fixes
```bash
# Check tasks are registered
python -m lm_eval --tasks list | grep python_

# Test with dummy model
python -m lm_eval --model dummy --tasks python_code_completion --limit 1 --predict_only --output_path dummy_test

# Test custom dataset loading
python -m lm_eval --model dummy --tasks python_code_completion --metadata '{"dataset_path": "my_custom_data.jsonl"}' --limit 1 --predict_only

# Run component tests
python test_task_simple.py

# Verify API key is set
echo $env:ANTHROPIC_API_KEY  # PowerShell
echo $ANTHROPIC_API_KEY      # Bash
```

### PowerShell vs Bash JSON Formatting
```powershell
# PowerShell (Windows) - Escape quotes
--metadata '{\"dataset_path\": \"file.jsonl\"}'

# Bash/Unix - Use single quotes around JSON
--metadata '{"dataset_path": "file.jsonl"}'
```

