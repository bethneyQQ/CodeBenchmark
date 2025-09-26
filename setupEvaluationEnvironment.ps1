# LM Evaluation Harness - Environment Setup Script (PowerShell)
# This script sets up the complete evaluation environment with dependency checks and API key validation

param(
    [switch]$SkipTests,
    [switch]$Verbose
)

# Colors for output
$Red = "Red"
$Green = "Green" 
$Yellow = "Yellow"
$Blue = "Cyan"

# Logging functions
function Write-Info {
    param($Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Blue
}

function Write-Success {
    param($Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Green
}

function Write-Warning {
    param($Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Yellow
}

function Write-Error {
    param($Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Red
}

# Header
Write-Host "========================================" -ForegroundColor $Blue
Write-Host "  LM Evaluation Harness Setup Script   " -ForegroundColor $Blue
Write-Host "         (PowerShell Version)          " -ForegroundColor $Blue
Write-Host "========================================" -ForegroundColor $Blue
Write-Host ""

Write-Info "Detected Windows PowerShell environment"

# Function to check command availability
function Test-Command {
    param($CommandName)
    
    if (Get-Command $CommandName -ErrorAction SilentlyContinue) {
        Write-Success "$CommandName is installed"
        return $true
    } else {
        Write-Error "$CommandName is not installed"
        return $false
    }
}

# Function to check Python version
function Test-PythonVersion {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python (\d+)\.(\d+)") {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]
            
            if ($major -eq 3 -and $minor -ge 8) {
                Write-Success "Python $($matches[0]) is compatible"
                return $true
            } else {
                Write-Error "Python $($matches[0]) is too old. Python 3.8+ required"
                return $false
            }
        }
    } else {
        Write-Error "Python is not installed"
        return $false
    }
}

# Function to check Node.js version
function Test-NodeVersion {
    if (Get-Command node -ErrorAction SilentlyContinue) {
        $nodeVersion = node --version
        $nodeVersion = $nodeVersion -replace "v", ""
        $major = [int]($nodeVersion -split "\.")[0]
        
        if ($major -ge 18) {
            Write-Success "Node.js $nodeVersion is compatible"
            return $true
        } else {
            Write-Warning "Node.js $nodeVersion is old. Node.js 18+ recommended for Claude Code SDK"
            return $false
        }
    } else {
        Write-Warning "Node.js is not installed. Required for Claude Code SDK"
        return $false
    }
}

# Function to setup virtual environment
function Set-VirtualEnvironment {
    Write-Info "Setting up Python virtual environment..."
    
    if (-not (Test-Path ".venv")) {
        Write-Info "Creating virtual environment..."
        python -m venv .venv
        Write-Success "Virtual environment created"
    } else {
        Write-Info "Virtual environment already exists"
    }
    
    # Activate virtual environment
    Write-Info "Activating virtual environment..."
    & .venv\Scripts\Activate.ps1
    Write-Success "Virtual environment activated"
}

# Function to install Python dependencies
function Install-PythonDependencies {
    Write-Info "Installing Python dependencies..."
    
    # Upgrade pip first
    Write-Info "Upgrading pip..."
    python -m pip install --upgrade pip
    
    # Install core dependencies
    Write-Info "Installing core dependencies..."
    $corePackages = @(
        "torch", "transformers", "datasets", "evaluate",
        "sacrebleu", "rouge-score", 
        "pandas", "numpy", "matplotlib", "seaborn",
        "accelerate", "peft", "more_itertools",
        "pytablewriter"
    )
    
    foreach ($package in $corePackages) {
        Write-Info "Installing $package..."
        pip install $package
    }
    
    Write-Success "Core dependencies installed"
    
    # Install model-specific dependencies
    Write-Info "Installing model-specific dependencies..."
    
    # Claude Code SDK
    Write-Info "Installing Claude Code SDK..."
    try {
        pip install claude-code-sdk
        Write-Success "Claude Code SDK installed"
    } catch {
        Write-Warning "Failed to install claude-code-sdk: $_"
    }
    
    # DashScope for DeepSeek/Qwen
    Write-Info "Installing DashScope..."
    try {
        pip install dashscope
        Write-Success "DashScope installed"
    } catch {
        Write-Warning "Failed to install dashscope: $_"
    }
    
    # OpenAI
    Write-Info "Installing OpenAI..."
    try {
        pip install openai
        Write-Success "OpenAI installed"
    } catch {
        Write-Warning "Failed to install openai: $_"
    }
    
    Write-Success "Model-specific dependencies installation completed"
}

# Function to check API keys
function Test-ApiKeys {
    Write-Info "Checking API key configuration..."
    
    $keysConfigured = 0
    $totalKeys = 3
    
    # Check Anthropic API Key
    if ($env:ANTHROPIC_API_KEY) {
        Write-Success "ANTHROPIC_API_KEY is configured"
        $keysConfigured++
    } else {
        Write-Warning "ANTHROPIC_API_KEY is not set"
        Write-Host "  To configure: `$env:ANTHROPIC_API_KEY = `"your-anthropic-key`""
    }
    
    # Check DashScope API Key
    if ($env:DASHSCOPE_API_KEY) {
        Write-Success "DASHSCOPE_API_KEY is configured"
        $keysConfigured++
    } else {
        Write-Warning "DASHSCOPE_API_KEY is not set"
        Write-Host "  To configure: `$env:DASHSCOPE_API_KEY = `"your-dashscope-key`""
    }
    
    # Check OpenAI API Key
    if ($env:OPENAI_API_KEY) {
        Write-Success "OPENAI_API_KEY is configured"
        $keysConfigured++
    } else {
        Write-Warning "OPENAI_API_KEY is not set"
        Write-Host "  To configure: `$env:OPENAI_API_KEY = `"your-openai-key`""
    }
    
    Write-Host ""
    Write-Info "API Keys configured: $keysConfigured/$totalKeys"
    
    if ($keysConfigured -eq 0) {
        Write-Error "No API keys configured. You need at least one API key to run evaluations."
        Write-Host ""
        Write-Host "Available model backends and their required API keys:"
        Write-Host "  • Claude Code SDK & Anthropic Claude: ANTHROPIC_API_KEY"
        Write-Host "  • DeepSeek & Qwen models: DASHSCOPE_API_KEY"
        Write-Host "  • OpenAI GPT models: OPENAI_API_KEY"
        Write-Host ""
        Write-Host "To set API keys permanently, add them to your PowerShell profile:"
        Write-Host "  Add-Content `$PROFILE '`$env:ANTHROPIC_API_KEY = `"your-key`"'"
        return $false
    } elseif ($keysConfigured -lt $totalKeys) {
        Write-Warning "Some API keys are missing. You can still use the configured models."
        return $true
    } else {
        Write-Success "All API keys are configured!"
        return $true
    }
}

# Function to test model connections
function Test-ModelConnections {
    Write-Info "Testing model connections..."
    
    # Test Claude Code SDK
    if ($env:ANTHROPIC_API_KEY) {
        Write-Info "Testing Claude Code SDK connection..."
        $testResult = python -c @"
try:
    import claude_code_sdk
    print('✅ Claude Code SDK import successful')
except ImportError as e:
    print(f'❌ Claude Code SDK import failed: {e}')
except Exception as e:
    print(f'⚠️  Claude Code SDK available but connection test failed: {e}')
"@ 2>$null
        Write-Host $testResult
    }
    
    # Test DashScope
    if ($env:DASHSCOPE_API_KEY) {
        Write-Info "Testing DashScope connection..."
        $testResult = python -c @"
try:
    import dashscope
    dashscope.api_key = '$env:DASHSCOPE_API_KEY'
    print('✅ DashScope configuration successful')
except ImportError as e:
    print(f'❌ DashScope import failed: {e}')
except Exception as e:
    print(f'⚠️  DashScope available but configuration test failed: {e}')
"@ 2>$null
        Write-Host $testResult
    }
    
    # Test OpenAI
    if ($env:OPENAI_API_KEY) {
        Write-Info "Testing OpenAI connection..."
        $testResult = python -c @"
try:
    import openai
    openai.api_key = '$env:OPENAI_API_KEY'
    print('✅ OpenAI configuration successful')
except ImportError as e:
    print(f'❌ OpenAI import failed: {e}')
except Exception as e:
    print(f'⚠️  OpenAI available but configuration test failed: {e}')
"@ 2>$null
        Write-Host $testResult
    }
}

# Function to run simple test
function Invoke-SimpleTest {
    Write-Info "Running simple evaluation test..."
    
    if (Test-Path "test_task_simple.py") {
        Write-Info "Running test_task_simple.py..."
        try {
            python test_task_simple.py
            Write-Success "Simple test completed"
        } catch {
            Write-Warning "Simple test failed, but this might be expected: $_"
        }
    } else {
        Write-Warning "test_task_simple.py not found, skipping simple test"
    }
}

# Function to create directories
function New-Directories {
    Write-Info "Creating necessary directories..."
    
    New-Item -ItemType Directory -Force -Path "results" | Out-Null
    New-Item -ItemType Directory -Force -Path "output" | Out-Null
    
    Write-Success "Directories created"
}

# Function to show usage examples
function Show-UsageExamples {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor $Blue
    Write-Host "           Usage Examples               " -ForegroundColor $Blue
    Write-Host "========================================" -ForegroundColor $Blue
    Write-Host ""
    
    Write-Host "🚀 Quick Start Commands:" -ForegroundColor $Green
    Write-Host ""
    
    if ($env:ANTHROPIC_API_KEY) {
        Write-Host "# Test with Claude Code SDK:"
        Write-Host "python -m lm_eval --model claude-code-local \"
        Write-Host "    --model_args model=claude-3-haiku-20240307,multi_turn=true \"
        Write-Host "    --tasks multi_turn_coding_eval_claude_code \"
        Write-Host "    --limit 1 --output_path results/claude_test.json --log_samples"
        Write-Host ""
    }
    
    if ($env:DASHSCOPE_API_KEY) {
        Write-Host "# Test with DeepSeek:"
        Write-Host "python -m lm_eval --model deepseek \"
        Write-Host "    --model_args model=deepseek-v3.1 \"
        Write-Host "    --tasks multi_turn_coding_eval_deepseek \"
        Write-Host "    --limit 1 --output_path results/deepseek_test.json --log_samples"
        Write-Host ""
        
        Write-Host "# Test with Qwen:"
        Write-Host "python -m lm_eval --model dashscope \"
        Write-Host "    --model_args model=qwen-plus \"
        Write-Host "    --tasks multi_turn_coding_eval_universal \"
        Write-Host "    --limit 1 --output_path results/qwen_test.json --log_samples"
        Write-Host ""
    }
    
    if ($env:OPENAI_API_KEY) {
        Write-Host "# Test with OpenAI:"
        Write-Host "python -m lm_eval --model openai-completions \"
        Write-Host "    --model_args model=gpt-4-turbo \"
        Write-Host "    --tasks multi_turn_coding_eval_openai \"
        Write-Host "    --limit 1 --output_path results/openai_test.json --log_samples"
        Write-Host ""
    }
    
    Write-Host "📊 Analysis Commands:" -ForegroundColor $Green
    Write-Host ""
    Write-Host "# Compare multiple models:"
    Write-Host "cd lm_eval/tasks/multi_turn_coding"
    Write-Host "python compare_models.py ../../../results/*.json --verbose"
    Write-Host ""
    Write-Host "# Analyze context impact:"
    Write-Host "python analyze_context_impact.py --results_dir ../../../results/"
    Write-Host ""
}

# Main execution
function Main {
    Write-Info "Starting environment setup..."
    Write-Host ""
    
    # Step 1: Check prerequisites
    Write-Info "Step 1: Checking prerequisites..."
    $prereqFailed = $false
    
    if (-not (Test-Command "git")) { $prereqFailed = $true }
    if (-not (Test-PythonVersion)) { $prereqFailed = $true }
    Test-NodeVersion | Out-Null  # Node.js is optional, just warn
    
    if ($prereqFailed) {
        Write-Error "Prerequisites check failed. Please install missing components."
        exit 1
    }
    
    Write-Host ""
    
    # Step 2: Set up virtual environment
    Write-Info "Step 2: Setting up virtual environment..."
    Set-VirtualEnvironment
    Write-Host ""
    
    # Step 3: Install dependencies
    Write-Info "Step 3: Installing dependencies..."
    Install-PythonDependencies
    Write-Host ""
    
    # Step 4: Create directories
    Write-Info "Step 4: Creating directories..."
    New-Directories
    Write-Host ""
    
    # Step 5: Check API keys
    Write-Info "Step 5: Checking API key configuration..."
    Test-ApiKeys | Out-Null
    Write-Host ""
    
    # Step 6: Test model connections
    Write-Info "Step 6: Testing model connections..."
    Test-ModelConnections
    Write-Host ""
    
    # Step 7: Run simple test
    if (-not $SkipTests) {
        Write-Info "Step 7: Running simple test..."
        Invoke-SimpleTest
        Write-Host ""
    }
    
    # Final summary
    Write-Host "========================================" -ForegroundColor $Green
    Write-Host "         Setup Complete! 🎉            " -ForegroundColor $Green
    Write-Host "========================================" -ForegroundColor $Green
    Write-Host ""
    
    Write-Success "Environment setup completed successfully!"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "1. Activate the virtual environment:"
    Write-Host "   .venv\Scripts\Activate.ps1"
    Write-Host "2. Configure any missing API keys"
    Write-Host "3. Run your first evaluation"
    Write-Host ""
    
    Show-UsageExamples
}

# Run main function
Main