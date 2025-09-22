#!/bin/bash

# Python Coding Context Comparison Script - Full vs No Context Only
# This script runs evaluations comparing Full Context vs No Context only

set -e

# Parse command line arguments
DEBUG_MODE=false
HELP=false
LIMIT=1  # Default limit

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$HELP" = "true" ]; then
    echo "🐍 Python Coding Context Comparison (Full vs No Context Only)"
    echo "============================================================"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --debug         Enable debug mode with detailed logging"
    echo "  --limit N       Number of problems to evaluate (default: 1)"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run with default settings (1 problem)"
    echo "  $0 --debug           # Run with debug logging enabled"
    echo "  $0 --limit 5         # Run with 5 problems"
    echo "  $0 --debug --limit 5 # Debug mode with 5 problems"
    echo ""
    exit 0
fi

echo "🐍 Python Coding Context Comparison (Full vs No Context Only)"
echo "============================================================="

# Configuration
OUTPUT_DIR="results/full_vs_no_context"
MODEL_TYPE="claude-local"
MODEL_NAME="claude-3-5-haiku-20241022"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "📁 Output directory: $OUTPUT_DIR"
echo "🤖 Model: $MODEL_TYPE ($MODEL_NAME)"
echo "📊 Sample limit: $LIMIT"
echo "🐛 Debug mode: $DEBUG_MODE"
echo ""

# Set debug environment variable for the evaluation
if [ "$DEBUG_MODE" = "true" ]; then
    export PYTHON_CODING_DEBUG=true
    echo "🔍 Debug logging enabled - detailed problem and response information will be shown"
    echo ""
else
    # Ensure debug mode is disabled if not requested
    unset PYTHON_CODING_DEBUG
fi

# Test model availability
echo "🧪 Setting up environment..."

# Find the lm-evaluation-harness root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LM_EVAL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "   Script directory: $SCRIPT_DIR"
echo "   LM-Eval root: $LM_EVAL_ROOT"

# Check if we found the correct directory
if [ ! -d "$LM_EVAL_ROOT/lm_eval" ]; then
    echo "❌ Could not find lm_eval directory at: $LM_EVAL_ROOT/lm_eval"
    echo "   Please make sure this script is in the correct location within the lm-evaluation-harness repository"
    exit 1
fi

echo "✅ Found lm_eval directory at: $LM_EVAL_ROOT/lm_eval"
echo "✅ Running from task directory: $(pwd)"
echo ""

# Function to run evaluation
run_evaluation() {
    local task_name=$1
    local output_file=$2
    local description=$3
    
    echo "🔄 Running: $description"
    echo "   Task: $task_name"
    echo "   Output: $output_file"
    
    # Build model_args based on debug mode
    if [ "$DEBUG_MODE" = "true" ]; then
        MODEL_ARGS="model=$MODEL_NAME,debug=true"
        echo "   🐛 Debug mode: Enabled (model will print responses)"
    else
        MODEL_ARGS="model=$MODEL_NAME"
        echo "   🐛 Debug mode: Disabled"
    fi
    
    # Create output directory for this specific file
    mkdir -p "$(dirname "$output_file")"
    
    echo "   🚀 Command: lm_eval --model $MODEL_TYPE --model_args \"$MODEL_ARGS\" --tasks $task_name --output_path $output_file --limit $LIMIT"
    
    lm_eval \
        --model "$MODEL_TYPE" \
        --model_args "$MODEL_ARGS" \
        --tasks "$task_name" \
        --output_path "$output_file" \
        --limit "$LIMIT" \
        --log_samples \
        --verbosity DEBUG
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # lm_eval adds timestamps to filenames, so look for files that start with our expected name
        local output_dir="$(dirname "$output_file")"
        local base_name="$(basename "$output_file" .json)"
        local actual_file=$(find "$output_dir" -name "${base_name}*.json" -type f | head -1)
        
        if [ -n "$actual_file" ] && [ -f "$actual_file" ]; then
            echo "   ✅ Success - Results saved to $actual_file"
            echo "   📊 File size: $(ls -lh "$actual_file" | awk '{print $5}')"
        else
            echo "   ⚠️  Command succeeded but output file not found"
            echo "   🔍 Expected pattern: ${output_dir}/${base_name}*.json"
            echo "   🔍 Files in output directory:"
            ls -la "$output_dir" 2>/dev/null || echo "      Directory not found"
            return 1
        fi
    else
        echo "   ❌ Failed with exit code: $exit_code"
        echo "   🔍 Check the error messages above for details"
        return 1
    fi
    echo ""
}

# Run evaluations for different context configurations
echo "🚀 Starting Full vs No Context Evaluations"
echo ""

# Define all task categories
TASK_CATEGORIES=("code_completion" "code_repair" "function_generation" "docstring_generation" "code_translation")

for category in "${TASK_CATEGORIES[@]}"; do
    # Capitalize first letter manually for compatibility
    category_title=$(echo "$category" | sed 's/_/ /g' | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) tolower(substr($i,2))}1')
    echo "🔧 $category_title"
    
    # Full Context Evaluation
    echo "   Running with Full Context..."
    export PYTHON_CODING_ENABLE_CONTEXT=true
    export PYTHON_CODING_CONTEXT_MODE=full
    run_evaluation "python_${category}" \
        "$OUTPUT_DIR/${category}_full_context.json" \
        "$category_title (Full Context)"
    
    # No Context Evaluation  
    echo "   Running with No Context..."
    export PYTHON_CODING_ENABLE_CONTEXT=false
    export PYTHON_CODING_CONTEXT_MODE=none
    run_evaluation "python_${category}" \
        "$OUTPUT_DIR/${category}_no_context.json" \
        "$category_title (No Context)"
    
    echo ""
done

# Clean up environment variables
unset PYTHON_CODING_ENABLE_CONTEXT
unset PYTHON_CODING_CONTEXT_MODE

echo ""
echo "🎉 Full vs No Context Comparison Complete!"
echo ""
echo "📋 Results Summary:"
echo "   📁 Results directory: $OUTPUT_DIR"
echo ""
echo "🔍 Files Generated:"
if [ -d "$OUTPUT_DIR" ]; then
    for file in "$OUTPUT_DIR"/*.json; do
        if [ -f "$file" ]; then
            echo "   - $(basename "$file")"
        fi
    done
    
    # Show file count and total size
    file_count=$(find "$OUTPUT_DIR" -name "*.json" -type f | wc -l)
    total_size=$(find "$OUTPUT_DIR" -name "*.json" -type f -exec ls -l {} \; | awk '{sum += $5} END {print sum}')
    echo "   📊 Total: $file_count files, $total_size bytes"
else
    echo "   ❌ Output directory not found: $OUTPUT_DIR"
fi
echo ""
echo "📊 Running Context Impact Analysis..."
python analyze_context_impact.py --results_dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "💡 Analysis Complete! Check these files:"
    echo "   📊 Report: $OUTPUT_DIR/analysis/context_impact_report.md (includes embedded charts)"
    echo "   📈 Charts: $OUTPUT_DIR/analysis/context_impact_charts.png"
else
    echo "⚠️  Analysis failed, but results are still available in $OUTPUT_DIR"
fi

echo ""
echo "💡 Usage Tips:"
echo "   - Review the analysis report for detailed insights"
echo "   - Use visualizations to identify context impact patterns"
echo "   - Compare exact match and BLEU scores across contexts"
echo "   - Use --debug flag for detailed problem and response logging"
echo ""
echo "🚀 Next Steps:"
echo "   - Run with larger sample sizes for production analysis"
echo "   - Test with different models to validate findings"
echo "   - Implement context-specific optimizations based on results"