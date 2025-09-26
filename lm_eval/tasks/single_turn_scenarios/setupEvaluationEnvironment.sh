#!/bin/bash

# Single Turn Scenarios - Environment Setup Script
# This script sets up the evaluation environment with all required dependencies

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

# Global variables
SETUP_LOG="setup_$(date +%Y%m%d_%H%M%S).log"
MISSING_DEPS=()
OPTIONAL_DEPS=()
API_KEYS_MISSING=()

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get version of a command
get_version() {
    local cmd="$1"
    local version_flag="$2"
    if command_exists "$cmd"; then
        $cmd $version_flag 2>/dev/null | head -n 1 || echo "Unknown version"
    else
        echo "Not installed"
    fi
}

# Function to check Python version
check_python() {
    log_info "Checking Python installation..."
    
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            log_success "Python $PYTHON_VERSION found (>= 3.8 required)"
            echo "python3,$PYTHON_VERSION,required,installed" >> "$SETUP_LOG"
        else
            log_error "Python $PYTHON_VERSION found, but >= 3.8 required"
            MISSING_DEPS+=("python3>=3.8")
            echo "python3,$PYTHON_VERSION,required,version_too_old" >> "$SETUP_LOG"
        fi
    else
        log_error "Python 3 not found"
        MISSING_DEPS+=("python3")
        echo "python3,not_found,required,missing" >> "$SETUP_LOG"
    fi
}

# Function to check Node.js
check_node() {
    log_info "Checking Node.js installation..."
    
    if command_exists node; then
        NODE_VERSION=$(get_version node "--version" | sed 's/v//')
        NODE_MAJOR=$(echo $NODE_VERSION | cut -d'.' -f1)
        
        if [ "$NODE_MAJOR" -ge 16 ]; then
            log_success "Node.js $NODE_VERSION found (>= 16 required)"
            echo "node,$NODE_VERSION,required,installed" >> "$SETUP_LOG"
        else
            log_error "Node.js $NODE_VERSION found, but >= 16 required"
            MISSING_DEPS+=("node>=16")
            echo "node,$NODE_VERSION,required,version_too_old" >> "$SETUP_LOG"
        fi
    else
        log_error "Node.js not found"
        MISSING_DEPS+=("node")
        echo "node,not_found,required,missing" >> "$SETUP_LOG"
    fi
    
    # Check npm
    if command_exists npm; then
        NPM_VERSION=$(get_version npm "--version")
        log_success "npm $NPM_VERSION found"
        echo "npm,$NPM_VERSION,required,installed" >> "$SETUP_LOG"
    else
        log_warning "npm not found (usually comes with Node.js)"
        OPTIONAL_DEPS+=("npm")
        echo "npm,not_found,required,missing" >> "$SETUP_LOG"
    fi
}

# Function to check Java
check_java() {
    log_info "Checking Java installation..."
    
    if command_exists java; then
        JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2)
        JAVA_MAJOR=$(echo $JAVA_VERSION | cut -d'.' -f1)
        
        if [ "$JAVA_MAJOR" -ge 11 ]; then
            log_success "Java $JAVA_VERSION found (>= 11 required)"
            echo "java,$JAVA_VERSION,required,installed" >> "$SETUP_LOG"
        else
            log_error "Java $JAVA_VERSION found, but >= 11 required"
            MISSING_DEPS+=("java>=11")
            echo "java,$JAVA_VERSION,required,version_too_old" >> "$SETUP_LOG"
        fi
    else
        log_error "Java not found"
        MISSING_DEPS+=("java")
        echo "java,not_found,required,missing" >> "$SETUP_LOG"
    fi
    
    # Check javac
    if command_exists javac; then
        JAVAC_VERSION=$(javac -version 2>&1 | cut -d' ' -f2)
        log_success "javac $JAVAC_VERSION found"
        echo "javac,$JAVAC_VERSION,required,installed" >> "$SETUP_LOG"
    else
        log_warning "javac not found (Java compiler)"
        MISSING_DEPS+=("javac")
        echo "javac,not_found,required,missing" >> "$SETUP_LOG"
    fi
}

# Function to check C/C++ compilers
check_cpp_compilers() {
    log_info "Checking C/C++ compilers..."
    
    local found_compiler=false
    
    # Check GCC
    if command_exists gcc; then
        GCC_VERSION=$(get_version gcc "--version" | head -n 1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
        log_success "GCC $GCC_VERSION found"
        echo "gcc,$GCC_VERSION,required,installed" >> "$SETUP_LOG"
        found_compiler=true
    else
        log_warning "GCC not found"
        echo "gcc,not_found,required,missing" >> "$SETUP_LOG"
    fi
    
    # Check Clang
    if command_exists clang; then
        CLANG_VERSION=$(get_version clang "--version" | head -n 1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
        log_success "Clang $CLANG_VERSION found"
        echo "clang,$CLANG_VERSION,required,installed" >> "$SETUP_LOG"
        found_compiler=true
    else
        log_warning "Clang not found"
        echo "clang,not_found,required,missing" >> "$SETUP_LOG"
    fi
    
    if [ "$found_compiler" = false ]; then
        log_error "No C/C++ compiler found (need gcc or clang)"
        MISSING_DEPS+=("gcc_or_clang")
    fi
    
    # Check g++
    if command_exists g++; then
        GPP_VERSION=$(get_version g++ "--version" | head -n 1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -n 1)
        log_success "g++ $GPP_VERSION found"
        echo "g++,$GPP_VERSION,required,installed" >> "$SETUP_LOG"
    else
        log_warning "g++ not found"
        echo "g++,not_found,required,missing" >> "$SETUP_LOG"
    fi
}

# Function to check Go
check_go() {
    log_info "Checking Go installation..."
    
    if command_exists go; then
        GO_VERSION=$(get_version go "version" | grep -o 'go[0-9]\+\.[0-9]\+\.[0-9]\+' | sed 's/go//')
        GO_MAJOR=$(echo $GO_VERSION | cut -d'.' -f1)
        GO_MINOR=$(echo $GO_VERSION | cut -d'.' -f2)
        
        if [ "$GO_MAJOR" -eq 1 ] && [ "$GO_MINOR" -ge 18 ]; then
            log_success "Go $GO_VERSION found (>= 1.18 required)"
            echo "go,$GO_VERSION,optional,installed" >> "$SETUP_LOG"
        else
            log_warning "Go $GO_VERSION found, but >= 1.18 recommended"
            OPTIONAL_DEPS+=("go>=1.18")
            echo "go,$GO_VERSION,optional,version_too_old" >> "$SETUP_LOG"
        fi
    else
        log_warning "Go not found (optional for Go language support)"
        OPTIONAL_DEPS+=("go")
        echo "go,not_found,optional,missing" >> "$SETUP_LOG"
    fi
}

# Function to check Rust
check_rust() {
    log_info "Checking Rust installation..."
    
    if command_exists rustc; then
        RUST_VERSION=$(get_version rustc "--version" | cut -d' ' -f2)
        log_success "Rust $RUST_VERSION found"
        echo "rustc,$RUST_VERSION,optional,installed" >> "$SETUP_LOG"
        
        # Check Cargo
        if command_exists cargo; then
            CARGO_VERSION=$(get_version cargo "--version" | cut -d' ' -f2)
            log_success "Cargo $CARGO_VERSION found"
            echo "cargo,$CARGO_VERSION,optional,installed" >> "$SETUP_LOG"
        else
            log_warning "Cargo not found (usually comes with Rust)"
            OPTIONAL_DEPS+=("cargo")
            echo "cargo,not_found,optional,missing" >> "$SETUP_LOG"
        fi
    else
        log_warning "Rust not found (optional for Rust language support)"
        OPTIONAL_DEPS+=("rust")
        echo "rust,not_found,optional,missing" >> "$SETUP_LOG"
    fi
}

# Function to check Docker
check_docker() {
    log_info "Checking Docker installation..."
    
    if command_exists docker; then
        DOCKER_VERSION=$(get_version docker "--version" | cut -d' ' -f3 | sed 's/,//')
        log_success "Docker $DOCKER_VERSION found"
        echo "docker,$DOCKER_VERSION,required,installed" >> "$SETUP_LOG"
        
        # Check if Docker daemon is running
        if docker info >/dev/null 2>&1; then
            log_success "Docker daemon is running"
            echo "docker_daemon,running,required,active" >> "$SETUP_LOG"
        else
            log_error "Docker daemon is not running"
            MISSING_DEPS+=("docker_daemon_running")
            echo "docker_daemon,not_running,required,inactive" >> "$SETUP_LOG"
        fi
    else
        log_error "Docker not found (required for sandbox execution)"
        MISSING_DEPS+=("docker")
        echo "docker,not_found,required,missing" >> "$SETUP_LOG"
    fi
}

# Function to install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    # Check if pip is available
    if ! command_exists pip3 && ! command_exists pip; then
        log_error "pip not found. Please install pip first."
        return 1
    fi
    
    local pip_cmd="pip3"
    if ! command_exists pip3; then
        pip_cmd="pip"
    fi
    
    # Install core dependencies
    log_info "Installing core lm-eval dependencies..."
    $pip_cmd install --upgrade pip
    $pip_cmd install lm-eval[api]
    
    # Install additional dependencies for single_turn_scenarios
    log_info "Installing single_turn_scenarios specific dependencies..."
    $pip_cmd install docker
    $pip_cmd install matplotlib seaborn
    $pip_cmd install pandas numpy
    $pip_cmd install pytest
    $pip_cmd install pyyaml
    $pip_cmd install jsonschema
    $pip_cmd install nltk
    $pip_cmd install rouge-score
    $pip_cmd install codebleu
    
    log_success "Python dependencies installed"
    echo "python_deps,installed,required,success" >> "$SETUP_LOG"
}

# Function to check API keys
check_api_keys() {
    log_info "Checking API key configuration..."
    
    local env_file=".env"
    local env_template=".env.template"
    
    # Create .env.template if it doesn't exist
    if [ ! -f "$env_template" ]; then
        log_info "Creating .env.template file..."
        cat > "$env_template" << 'EOF'
# API Keys for Model Evaluation
# Copy this file to .env and fill in your API keys

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key  
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# DeepSeek API Key
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Optional: Other model provider keys
# COHERE_API_KEY=your_cohere_api_key_here
# HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Evaluation Configuration
# EVAL_BATCH_SIZE=4
# EVAL_TIMEOUT=60
EOF
        log_success "Created .env.template file"
    fi
    
    # Check if .env file exists
    if [ ! -f "$env_file" ]; then
        log_warning ".env file not found. Please copy .env.template to .env and configure your API keys."
        API_KEYS_MISSING+=("env_file")
        echo "env_file,not_found,required,missing" >> "$SETUP_LOG"
        return
    fi
    
    # Source the .env file
    set -a  # automatically export all variables
    source "$env_file" 2>/dev/null || true
    set +a
    
    # Check individual API keys
    local keys_found=0
    
    if [ -n "$OPENAI_API_KEY" ] && [ "$OPENAI_API_KEY" != "your_openai_api_key_here" ]; then
        log_success "OpenAI API key configured"
        echo "openai_api_key,configured,optional,found" >> "$SETUP_LOG"
        ((keys_found++))
    else
        log_warning "OpenAI API key not configured"
        API_KEYS_MISSING+=("OPENAI_API_KEY")
        echo "openai_api_key,not_configured,optional,missing" >> "$SETUP_LOG"
    fi
    
    if [ -n "$ANTHROPIC_API_KEY" ] && [ "$ANTHROPIC_API_KEY" != "your_anthropic_api_key_here" ]; then
        log_success "Anthropic API key configured"
        echo "anthropic_api_key,configured,optional,found" >> "$SETUP_LOG"
        ((keys_found++))
    else
        log_warning "Anthropic API key not configured"
        API_KEYS_MISSING+=("ANTHROPIC_API_KEY")
        echo "anthropic_api_key,not_configured,optional,missing" >> "$SETUP_LOG"
    fi
    
    if [ -n "$DEEPSEEK_API_KEY" ] && [ "$DEEPSEEK_API_KEY" != "your_deepseek_api_key_here" ]; then
        log_success "DeepSeek API key configured"
        echo "deepseek_api_key,configured,optional,found" >> "$SETUP_LOG"
        ((keys_found++))
    else
        log_warning "DeepSeek API key not configured"
        API_KEYS_MISSING+=("DEEPSEEK_API_KEY")
        echo "deepseek_api_key,not_configured,optional,missing" >> "$SETUP_LOG"
    fi
    
    if [ $keys_found -eq 0 ]; then
        log_error "No API keys configured. You need at least one API key to run evaluations."
        echo "api_keys,none_configured,required,critical" >> "$SETUP_LOG"
    else
        log_success "$keys_found API key(s) configured"
        echo "api_keys,$keys_found,optional,partial" >> "$SETUP_LOG"
    fi
}

# Function to generate environment report
generate_report() {
    log_info "Generating environment diagnostic report..."
    
    local report_file="environment_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# Environment Diagnostic Report

Generated on: $(date)
System: $(uname -a)

## Dependency Status

### Required Dependencies
EOF
    
    # Add dependency status to report
    while IFS=',' read -r component version type status; do
        if [ "$type" = "required" ]; then
            case $status in
                "installed")
                    echo "- ✅ $component: $version" >> "$report_file"
                    ;;
                "missing"|"not_found")
                    echo "- ❌ $component: Not installed" >> "$report_file"
                    ;;
                "version_too_old")
                    echo "- ⚠️ $component: $version (version too old)" >> "$report_file"
                    ;;
                *)
                    echo "- ❓ $component: $status" >> "$report_file"
                    ;;
            esac
        fi
    done < "$SETUP_LOG"
    
    echo "" >> "$report_file"
    echo "### Optional Dependencies" >> "$report_file"
    
    while IFS=',' read -r component version type status; do
        if [ "$type" = "optional" ]; then
            case $status in
                "installed")
                    echo "- ✅ $component: $version" >> "$report_file"
                    ;;
                "missing"|"not_found")
                    echo "- ⚪ $component: Not installed (optional)" >> "$report_file"
                    ;;
                "version_too_old")
                    echo "- ⚠️ $component: $version (older version)" >> "$report_file"
                    ;;
                *)
                    echo "- ❓ $component: $status" >> "$report_file"
                    ;;
            esac
        fi
    done < "$SETUP_LOG"
    
    # Add installation instructions for missing dependencies
    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        echo "" >> "$report_file"
        echo "## Missing Required Dependencies" >> "$report_file"
        echo "" >> "$report_file"
        echo "The following required dependencies are missing:" >> "$report_file"
        echo "" >> "$report_file"
        
        for dep in "${MISSING_DEPS[@]}"; do
            echo "- $dep" >> "$report_file"
        done
        
        echo "" >> "$report_file"
        echo "### Installation Instructions" >> "$report_file"
        echo "" >> "$report_file"
        
        # Add OS-specific installation instructions
        if command_exists apt-get; then
            echo "#### Ubuntu/Debian:" >> "$report_file"
            echo '```bash' >> "$report_file"
            echo "sudo apt-get update" >> "$report_file"
            for dep in "${MISSING_DEPS[@]}"; do
                case $dep in
                    "python3"*) echo "sudo apt-get install python3 python3-pip" >> "$report_file" ;;
                    "node"*) echo "curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -" >> "$report_file"
                            echo "sudo apt-get install -y nodejs" >> "$report_file" ;;
                    "java"*) echo "sudo apt-get install openjdk-11-jdk" >> "$report_file" ;;
                    "gcc_or_clang") echo "sudo apt-get install build-essential" >> "$report_file" ;;
                    "docker"*) echo "sudo apt-get install docker.io" >> "$report_file"
                              echo "sudo systemctl start docker" >> "$report_file"
                              echo "sudo usermod -aG docker \$USER" >> "$report_file" ;;
                esac
            done
            echo '```' >> "$report_file"
            echo "" >> "$report_file"
        fi
        
        if command_exists brew; then
            echo "#### macOS (Homebrew):" >> "$report_file"
            echo '```bash' >> "$report_file"
            for dep in "${MISSING_DEPS[@]}"; do
                case $dep in
                    "python3"*) echo "brew install python" >> "$report_file" ;;
                    "node"*) echo "brew install node" >> "$report_file" ;;
                    "java"*) echo "brew install openjdk@11" >> "$report_file" ;;
                    "gcc_or_clang") echo "xcode-select --install" >> "$report_file" ;;
                    "docker"*) echo "brew install --cask docker" >> "$report_file" ;;
                esac
            done
            echo '```' >> "$report_file"
            echo "" >> "$report_file"
        fi
    fi
    
    # Add API key configuration section
    if [ ${#API_KEYS_MISSING[@]} -gt 0 ]; then
        echo "" >> "$report_file"
        echo "## API Key Configuration" >> "$report_file"
        echo "" >> "$report_file"
        echo "Configure your API keys in the .env file:" >> "$report_file"
        echo "" >> "$report_file"
        echo '```bash' >> "$report_file"
        echo "cp .env.template .env" >> "$report_file"
        echo "# Edit .env file with your actual API keys" >> "$report_file"
        echo '```' >> "$report_file"
    fi
    
    log_success "Environment report generated: $report_file"
    echo "report,$report_file,info,generated" >> "$SETUP_LOG"
}

# Function to display summary
display_summary() {
    echo ""
    echo "=========================================="
    echo "         SETUP SUMMARY"
    echo "=========================================="
    
    if [ ${#MISSING_DEPS[@]} -eq 0 ]; then
        log_success "All required dependencies are installed!"
    else
        log_error "Missing ${#MISSING_DEPS[@]} required dependencies:"
        for dep in "${MISSING_DEPS[@]}"; do
            echo "  - $dep"
        done
    fi
    
    if [ ${#OPTIONAL_DEPS[@]} -gt 0 ]; then
        log_warning "Missing ${#OPTIONAL_DEPS[@]} optional dependencies:"
        for dep in "${OPTIONAL_DEPS[@]}"; do
            echo "  - $dep"
        done
    fi
    
    if [ ${#API_KEYS_MISSING[@]} -gt 0 ]; then
        log_warning "API keys need configuration:"
        for key in "${API_KEYS_MISSING[@]}"; do
            echo "  - $key"
        done
    fi
    
    echo ""
    echo "Setup log: $SETUP_LOG"
    echo "Environment report: environment_report_*.md"
    echo ""
    
    if [ ${#MISSING_DEPS[@]} -eq 0 ]; then
        log_success "Environment setup complete! You can now run evaluations."
        return 0
    else
        log_error "Please install missing dependencies before running evaluations."
        return 1
    fi
}

# Main setup function
main() {
    echo "=========================================="
    echo "  Single Turn Scenarios - Environment Setup"
    echo "=========================================="
    echo ""
    
    log_info "Starting environment setup..."
    log_info "Setup log: $SETUP_LOG"
    echo ""
    
    # Initialize log file
    echo "component,version,type,status" > "$SETUP_LOG"
    
    # Check all dependencies
    check_python
    check_node
    check_java
    check_cpp_compilers
    check_go
    check_rust
    check_docker
    
    echo ""
    
    # Install Python dependencies if requested
    if [ "$1" = "--install-deps" ] || [ "$1" = "-i" ]; then
        install_python_deps
        echo ""
    fi
    
    # Check API keys
    check_api_keys
    
    echo ""
    
    # Generate report
    generate_report
    
    # Display summary
    display_summary
}

# Help function
show_help() {
    echo "Single Turn Scenarios - Environment Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -i, --install-deps  Install Python dependencies automatically"
    echo ""
    echo "This script checks for required dependencies and generates a diagnostic report."
    echo "Required dependencies: Python 3.8+, Node.js 16+, Java 11+, C/C++ compiler, Docker"
    echo "Optional dependencies: Go 1.18+, Rust"
    echo ""
}

# Parse command line arguments
case "$1" in
    -h|--help)
        show_help
        exit 0
        ;;
    -i|--install-deps)
        main "$1"
        ;;
    "")
        main
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac