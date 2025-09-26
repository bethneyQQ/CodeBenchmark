#!/bin/bash

# LM Evaluation Harness - Environment Setup Script
# This script sets up the complete evaluation environment with dependency checks and API key validation

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Header
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  LM Evaluation Harness Setup Script   ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running on Windows (Git Bash/WSL)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    log_info "Detected Windows environment (Git Bash/MSYS)"
    IS_WINDOWS=true
else
    log_info "Detected Unix-like environment"
    IS_WINDOWS=false
fi

# Function to check command availability
check_command() {
    if command -v "$1" &> /dev/null; then
        log_success "$1 is installed"
        return 0
    else
        log_error "$1 is not installed"
        return 1
    fi
}

# Function to check Python version
check_python_version() {
    if command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 8 ]]; then
            log_success "Python $PYTHON_VERSION is compatible"
            return 0
        else
            log_error "Python $PYTHON_VERSION is too old. Python 3.8+ required"
            return 1
        fi
    else
        log_error "Python is not installed"
        return 1
    fi
}

# Function to check Node.js version
check_nodejs_version() {
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version | sed 's/v//')
        NODE_MAJOR=$(echo $NODE_VERSION | cut -d'.' -f1)
        
        if [[ $NODE_MAJOR -ge 18 ]]; then
            log_success "Node.js $NODE_VERSION is compatible"
            return 0
        else
            log_warning "Node.js $NODE_VERSION is old. Node.js 18+ recommended for Claude Code SDK"
            return 1
        fi
    else
        log_warning "Node.js is not installed. Required for Claude Code SDK"
        return 1
    fi
}

# Function to check and create virtual environment
setup_virtual_environment() {
    log_info "Setting up Python virtual environment..."
    
    if [[ ! -d ".venv" ]]; then
        log_info "Creating virtual environment..."
        python -m venv .venv
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    if [[ $IS_WINDOWS == true ]]; then
        source .venv/Scripts/activate
    else
        source .venv/bin/activate
    fi
    
    log_success "Virtual environment activated"
}

# Function to install Python dependencies
install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Upgrade pip first
    log_info "Upgrading pip..."
    python -m pip install --upgrade pip
    
    # Install core dependencies
    log_info "Installing core dependencies..."
    pip install torch transformers datasets evaluate
    pip install sacrebleu rouge-score 
    pip install pandas numpy matplotlib seaborn
    pip install accelerate peft more_itertools
    pip install pytablewriter
    
    log_success "Core dependencies installed"
    
    # Install model-specific dependencies
    log_info "Installing model-specific dependencies..."
    
    # Claude Code SDK
    log_info "Installing Claude Code SDK..."
    pip install claude-code-sdk || log_warning "Failed to install claude-code-sdk"
    
    # DashScope for DeepSeek/Qwen
    log_info "Installing DashScope..."
    pip install dashscope || log_warning "Failed to install dashscope"
    
    # OpenAI
    log_info "Installing OpenAI..."
    pip install openai || log_warning "Failed to install openai"
    
    log_success "Model-specific dependencies installation completed"
}

# Function to check API keys
check_api_keys() {
    log_info "Checking API key configuration..."
    
    local keys_configured=0
    local total_keys=3
    
    # Check Anthropic API Key
    if [[ -n "${ANTHROPIC_API_KEY}" ]]; then
        log_success "ANTHROPIC_API_KEY is configured"
        ((keys_configured++))
    else
        log_warning "ANTHROPIC_API_KEY is not set"
        echo "  To configure: export ANTHROPIC_API_KEY=\"your-anthropic-key\""
    fi
    
    # Check DashScope API Key
    if [[ -n "${DASHSCOPE_API_KEY}" ]]; then
        log_success "DASHSCOPE_API_KEY is configured"
        ((keys_configured++))
    else
        log_warning "DASHSCOPE_API_KEY is not set"
        echo "  To configure: export DASHSCOPE_API_KEY=\"your-dashscope-key\""
    fi
    
    # Check OpenAI API Key
    if [[ -n "${OPENAI_API_KEY}" ]]; then
        log_success "OPENAI_API_KEY is configured"
        ((keys_configured++))
    else
        log_warning "OPENAI_API_KEY is not set"
        echo "  To configure: export OPENAI_API_KEY=\"your-openai-key\""
    fi
    
    echo ""
    log_info "API Keys configured: $keys_configured/$total_keys"
    
    if [[ $keys_configured -eq 0 ]]; then
        log_error "No API keys configured. You need at least one API key to run evaluations."
        echo ""
        echo "Available model backends and their required API keys:"
        echo "  • Claude Code SDK & Anthropic Claude: ANTHROPIC_API_KEY"
        echo "  • DeepSeek & Qwen models: DASHSCOPE_API_KEY"
        echo "  • OpenAI GPT models: OPENAI_API_KEY"
        echo ""
        echo "To set API keys permanently, add them to your shell profile:"
        echo "  echo 'export ANTHROPIC_API_KEY=\"your-key\"' >> ~/.bashrc"
        echo "  echo 'export DASHSCOPE_API_KEY=\"your-key\"' >> ~/.bashrc"
        echo "  echo 'export OPENAI_API_KEY=\"your-key\"' >> ~/.bashrc"
        return 1
    elif [[ $keys_configured -lt $total_keys ]]; then
        log_warning "Some API keys are missing. You can still use the configured models."
        return 0
    else
        log_success "All API keys are configured!"
        return 0
    fi
}

# Function to test model connections
test_model_connections() {
    log_info "Testing model connections..."
    
    # Test Claude Code SDK
    if [[ -n "${ANTHROPIC_API_KEY}" ]]; then
        log_info "Testing Claude Code SDK connection..."
        python -c "
try:
    import claude_code_sdk
    print('✅ Claude Code SDK import successful')
except ImportError as e:
    print(f'❌ Claude Code SDK import failed: {e}')
except Exception as e:
    print(f'⚠️  Claude Code SDK available but connection test failed: {e}')
" 2>/dev/null || log_warning "Claude Code SDK test failed"
    fi
    
    # Test DashScope
    if [[ -n "${DASHSCOPE_API_KEY}" ]]; then
        log_info "Testing DashScope connection..."
        python -c "
try:
    import dashscope
    dashscope.api_key = '$DASHSCOPE_API_KEY'
    print('✅ DashScope configuration successful')
except ImportError as e:
    print(f'❌ DashScope import failed: {e}')
except Exception as e:
    print(f'⚠️  DashScope available but configuration test failed: {e}')
" 2>/dev/null || log_warning "DashScope test failed"
    fi
    
    # Test OpenAI
    if [[ -n "${OPENAI_API_KEY}" ]]; then
        log_info "Testing OpenAI connection..."
        python -c "
try:
    import openai
    openai.api_key = '$OPENAI_API_KEY'
    print('✅ OpenAI configuration successful')
except ImportError as e:
    print(f'❌ OpenAI import failed: {e}')
except Exception as e:
    print(f'⚠️  OpenAI available but configuration test failed: {e}')
" 2>/dev/null || log_warning "OpenAI test failed"
    fi
}

# Function to run a simple evaluation test
run_simple_test() {
    log_info "Running simple evaluation test..."
    
    if [[ -f "test_task_simple.py" ]]; then
        log_info "Running test_task_simple.py..."
        python test_task_simple.py || log_warning "Simple test failed, but this might be expected"
    else
        log_warning "test_task_simple.py not found, skipping simple test"
    fi
}

# Function to create directories
create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p results
    mkdir -p output
    
    log_success "Directories created"
}

# Function to display usage examples
show_usage_examples() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}           Usage Examples               ${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    echo "🚀 Quick Start Commands:"
    echo ""
    
    if [[ -n "${ANTHROPIC_API_KEY}" ]]; then
        echo "# Test with Claude Code SDK:"
        echo "python -m lm_eval --model claude-code-local \\"
        echo "    --model_args model=claude-3-haiku-20240307,multi_turn=true \\"
        echo "    --tasks multi_turn_coding_eval_claude_code \\"
        echo "    --limit 1 --output_path results/claude_test.json --log_samples"
        echo ""
    fi
    
    if [[ -n "${DASHSCOPE_API_KEY}" ]]; then
        echo "# Test with DeepSeek:"
        echo "python -m lm_eval --model deepseek \\"
        echo "    --model_args model=deepseek-v3.1 \\"
        echo "    --tasks multi_turn_coding_eval_deepseek \\"
        echo "    --limit 1 --output_path results/deepseek_test.json --log_samples"
        echo ""
        
        echo "# Test with Qwen:"
        echo "python -m lm_eval --model dashscope \\"
        echo "    --model_args model=qwen-plus \\"
        echo "    --tasks multi_turn_coding_eval_universal \\"
        echo "    --limit 1 --output_path results/qwen_test.json --log_samples"
        echo ""
    fi
    
    if [[ -n "${OPENAI_API_KEY}" ]]; then
        echo "# Test with OpenAI:"
        echo "python -m lm_eval --model openai-completions \\"
        echo "    --model_args model=gpt-4-turbo \\"
        echo "    --tasks multi_turn_coding_eval_openai \\"
        echo "    --limit 1 --output_path results/openai_test.json --log_samples"
        echo ""
    fi
    
    echo "📊 Analysis Commands:"
    echo ""
    echo "# Compare multiple models:"
    echo "cd lm_eval/tasks/multi_turn_coding"
    echo "python compare_models.py ../../../results/*.json --verbose"
    echo ""
    echo "# Analyze context impact:"
    echo "python analyze_context_impact.py --results_dir ../../../results/"
    echo ""
}

# Main execution
main() {
    log_info "Starting environment setup..."
    echo ""
    
    # Step 1: Check prerequisites
    log_info "Step 1: Checking prerequisites..."
    local prereq_failed=false
    
    check_command "git" || prereq_failed=true
    check_python_version || prereq_failed=true
    check_nodejs_version || true  # Node.js is optional, just warn
    
    if [[ $prereq_failed == true ]]; then
        log_error "Prerequisites check failed. Please install missing components."
        exit 1
    fi
    
    echo ""
    
    # Step 2: Set up virtual environment
    log_info "Step 2: Setting up virtual environment..."
    setup_virtual_environment
    echo ""
    
    # Step 3: Install dependencies
    log_info "Step 3: Installing dependencies..."
    install_python_dependencies
    echo ""
    
    # Step 4: Create directories
    log_info "Step 4: Creating directories..."
    create_directories
    echo ""
    
    # Step 5: Check API keys
    log_info "Step 5: Checking API key configuration..."
    check_api_keys
    echo ""
    
    # Step 6: Test model connections
    log_info "Step 6: Testing model connections..."
    test_model_connections
    echo ""
    
    # Step 7: Run simple test
    log_info "Step 7: Running simple test..."
    run_simple_test
    echo ""
    
    # Final summary
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}         Setup Complete! 🎉            ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    
    log_success "Environment setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    if [[ $IS_WINDOWS == true ]]; then
        echo "   source .venv/Scripts/activate"
    else
        echo "   source .venv/bin/activate"
    fi
    echo "2. Configure any missing API keys"
    echo "3. Run your first evaluation"
    echo ""
    
    show_usage_examples
}

# Run main function
main "$@"