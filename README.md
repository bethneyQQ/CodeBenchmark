# 🎯 Complete Guide: Python Code Quality Evaluation

## ✅ **Task Status: Fully Tested and Working**

All tests pass! The task is ready for evaluation.

## 🧪 **Step 1: Verify Task Works (Optional)**

Run the simple test to confirm everything is working:

```bash
python test_task_simple.py
```

## 🚀 **Step 2: Run with Your Local Model**

Run with your local Qwen model, managed by LM_Studio:

### Option A: Chat Completions API
```bash
lm_eval \
  --model local-chat-completions \
  --model_args base_url=http://localhost:1234/v1/chat/completions,model=qwen/qwen3-1.7b \
  --tasks python_code_quality \
  --output_path output_code_results/qwen_code_quality_results.json \
  --log_samples \
  --limit 2 \
  --batch_size 1 \
  --apply_chat_template
```

### Option B: Completions API
```bash
lm_eval \
  --model local-completions \
  --model_args base_url=http://localhost:1234/v1/completions,model=qwen/qwen3-1.7b \
  --tasks python_code_quality \
  --output_path output_code_results/qwen_code_quality_results.json \
  --log_samples \
  --limit 2 \
  --batch_size 1
```

## 🔬 **Step 2: Test with Hosted Model (Recommended)**

Test with a remoted model to verify results:

### Option A: Claude (Anthropic) - Recommended for Code
```bash
export ANTHROPIC_API_KEY="your_key_here"

lm_eval \
  --model anthropic-chat \
  --model_args model=claude-3-haiku-20240307 \
  --tasks python_code_quality \
  --output_path test_results/claude_test.json \
  --log_samples \
  --limit 2 \
  --batch_size 1
```

### Option B: Claude Sonnet (More Powerful)
```bash
export ANTHROPIC_API_KEY="your_key_here"

lm_eval \
  --model anthropic-chat \
  --model_args model=claude-3-haiku-20240307 \
  --tasks python_code_quality \
  --output_path test_results/claude_test1.json \
  --log_samples \
  --limit 1 \
  --batch_size 1
```

## 📊 **Understanding Results**

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

## 🎯 **Test Problems (First 2 with --limit 2)**

1. **Factorial Function**
   - Calculate n! with proper error handling
   - Tests: `factorial(0) == 1`, `factorial(5) == 120`, `factorial(1) == 1`

2. **Palindrome Checker**
   - Detect palindromes ignoring case and spaces
   - Tests: `is_palindrome('racecar') == True`, etc.

## 🔧 **Troubleshooting**

### Common Issues
- **"task not found"**: Run from repo root directory
- **API key errors**: Ensure environment variables are set
- **Connection errors**: Check local server is running
- **Chat template errors**: Add `--apply_chat_template` flag for chat APIs

### Quick Fixes
```bash
# Check task is registered
lm_eval --tasks list | grep python_code_quality

# Test with dummy model
lm_eval --model dummy --tasks python_code_quality --limit 1 --predict_only --output_path dummy_test

# Run component tests
python test_task_simple.py
```

## 🎉 **You're Ready!**

1. ✅ Task is fully tested and working
2. ✅ Multiple API options available
3. ✅ Comprehensive metrics implemented
4. ✅ Safe code execution environment
5. ✅ 20 diverse Python problems in dataset

Choose your preferred command and start evaluating! 🚀
