# Single Turn Scenarios - Environment Setup Script (PowerShell)
# This script sets up the evaluation environment with all required dependencies

param(
    [switch]$InstallDeps,
    [switch]$Help
)

# Global variables
$Script:SetupLog = "setup_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
$Script:MissingDeps = @()
$Script:OptionalDeps = @()
$Script:ApiKeysMissing = @()

# Logging functions
function Write-Info {
    param($Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param($Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param($Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param($Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to check if command exists
function Test-Command {
    param($CommandName)
    return $null -ne (Get-Command $CommandName -ErrorAction SilentlyContinue)
}

# Function to get version of a command
function Get-CommandVersion {
    param($Command, $VersionFlag)
    try {
        if (Test-Command $Command) {
            $output = & $Command $VersionFlag 2>$null
            return $output[0]
        }
        return "Not installed"
    }
    catch {
        return "Unknown version"
    }
}

# Function to check Python version
function Test-Python {
    Write-Info "Checking Python installation..."
    
    if (Test-Command "python") {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python (\d+)\.(\d+)\.(\d+)") {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]
            $version = "$major.$minor.$($matches[3])"
            
            if ($major -eq 3 -and $minor -ge 8) {
                Write-Success "Python $version found (>= 3.8 required)"
                Add-Content -Path $Script:SetupLog -Value "python,$version,required,installed"
            }
            else {
                Write-Error "Python $version found, but >= 3.8 required"
                $Script:MissingDeps += "python>=3.8"
                Add-Content -Path $Script:SetupLog -Value "python,$version,required,version_too_old"
            }
        }
    }
    elseif (Test-Command "python3") {
        $pythonVersion = python3 --version 2>&1
        if ($pythonVersion -match "Python (\d+)\.(\d+)\.(\d+)") {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]
            $version = "$major.$minor.$($matches[3])"
            
            if ($major -eq 3 -and $minor -ge 8) {
                Write-Success "Python $version found (>= 3.8 required)"
                Add-Content -Path $Script:SetupLog -Value "python3,$version,required,installed"
            }
            else {
                Write-Error "Python $version found, but >= 3.8 required"
                $Script:MissingDeps += "python3>=3.8"
                Add-Content -Path $Script:SetupLog -Value "python3,$version,required,version_too_old"
            }
        }
    }
    else {
        Write-Error "Python not found"
        $Script:MissingDeps += "python"
        Add-Content -Path $Script:SetupLog -Value "python,not_found,required,missing"
    }
}

# Function to check Node.js
function Test-Node {
    Write-Info "Checking Node.js installation..."
    
    if (Test-Command "node") {
        $nodeVersion = node --version 2>$null
        if ($nodeVersion -match "v(\d+)\.(\d+)\.(\d+)") {
            $major = [int]$matches[1]
            $version = "$major.$($matches[2]).$($matches[3])"
            
            if ($major -ge 16) {
                Write-Success "Node.js $version found (>= 16 required)"
                Add-Content -Path $Script:SetupLog -Value "node,$version,required,installed"
            }
            else {
                Write-Error "Node.js $version found, but >= 16 required"
                $Script:MissingDeps += "node>=16"
                Add-Content -Path $Script:SetupLog -Value "node,$version,required,version_too_old"
            }
        }
    }
    else {
        Write-Error "Node.js not found"
        $Script:MissingDeps += "node"
        Add-Content -Path $Script:SetupLog -Value "node,not_found,required,missing"
    }
    
    # Check npm
    if (Test-Command "npm") {
        $npmVersion = npm --version 2>$null
        Write-Success "npm $npmVersion found"
        Add-Content -Path $Script:SetupLog -Value "npm,$npmVersion,required,installed"
    }
    else {
        Write-Warning "npm not found (usually comes with Node.js)"
        $Script:OptionalDeps += "npm"
        Add-Content -Path $Script:SetupLog -Value "npm,not_found,required,missing"
    }
}

# Function to check Java
function Test-Java {
    Write-Info "Checking Java installation..."
    
    if (Test-Command "java") {
        $javaVersion = java -version 2>&1
        if ($javaVersion -match '"(\d+)\.(\d+)\.(\d+)') {
            $major = [int]$matches[1]
            $version = "$major.$($matches[2]).$($matches[3])"
            
            if ($major -ge 11) {
                Write-Success "Java $version found (>= 11 required)"
                Add-Content -Path $Script:SetupLog -Value "java,$version,required,installed"
            }
            else {
                Write-Error "Java $version found, but >= 11 required"
                $Script:MissingDeps += "java>=11"
                Add-Content -Path $Script:SetupLog -Value "java,$version,required,version_too_old"
            }
        }
        elseif ($javaVersion -match 'version "(\d+)') {
            $major = [int]$matches[1]
            if ($major -ge 11) {
                Write-Success "Java $major found (>= 11 required)"
                Add-Content -Path $Script:SetupLog -Value "java,$major,required,installed"
            }
            else {
                Write-Error "Java $major found, but >= 11 required"
                $Script:MissingDeps += "java>=11"
                Add-Content -Path $Script:SetupLog -Value "java,$major,required,version_too_old"
            }
        }
    }
    else {
        Write-Error "Java not found"
        $Script:MissingDeps += "java"
        Add-Content -Path $Script:SetupLog -Value "java,not_found,required,missing"
    }
    
    # Check javac
    if (Test-Command "javac") {
        $javacVersion = javac -version 2>&1
        Write-Success "javac found"
        Add-Content -Path $Script:SetupLog -Value "javac,found,required,installed"
    }
    else {
        Write-Warning "javac not found (Java compiler)"
        $Script:MissingDeps += "javac"
        Add-Content -Path $Script:SetupLog -Value "javac,not_found,required,missing"
    }
}

# Function to check C/C++ compilers
function Test-CppCompilers {
    Write-Info "Checking C/C++ compilers..."
    
    $foundCompiler = $false
    
    # Check for Visual Studio Build Tools / MSVC
    if (Test-Command "cl") {
        Write-Success "MSVC compiler found"
        Add-Content -Path $Script:SetupLog -Value "msvc,found,required,installed"
        $foundCompiler = $true
    }
    
    # Check GCC (MinGW)
    if (Test-Command "gcc") {
        $gccVersion = gcc --version 2>$null | Select-Object -First 1
        if ($gccVersion -match "(\d+\.\d+\.\d+)") {
            Write-Success "GCC $($matches[1]) found"
            Add-Content -Path $Script:SetupLog -Value "gcc,$($matches[1]),required,installed"
            $foundCompiler = $true
        }
    }
    
    # Check Clang
    if (Test-Command "clang") {
        $clangVersion = clang --version 2>$null | Select-Object -First 1
        if ($clangVersion -match "(\d+\.\d+\.\d+)") {
            Write-Success "Clang $($matches[1]) found"
            Add-Content -Path $Script:SetupLog -Value "clang,$($matches[1]),required,installed"
            $foundCompiler = $true
        }
    }
    
    if (-not $foundCompiler) {
        Write-Error "No C/C++ compiler found (need MSVC, gcc, or clang)"
        $Script:MissingDeps += "cpp_compiler"
        Add-Content -Path $Script:SetupLog -Value "cpp_compiler,not_found,required,missing"
    }
}

# Function to check Go
function Test-Go {
    Write-Info "Checking Go installation..."
    
    if (Test-Command "go") {
        $goVersion = go version 2>$null
        if ($goVersion -match "go(\d+)\.(\d+)\.(\d+)") {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]
            $version = "$major.$minor.$($matches[3])"
            
            if ($major -eq 1 -and $minor -ge 18) {
                Write-Success "Go $version found (>= 1.18 required)"
                Add-Content -Path $Script:SetupLog -Value "go,$version,optional,installed"
            }
            else {
                Write-Warning "Go $version found, but >= 1.18 recommended"
                $Script:OptionalDeps += "go>=1.18"
                Add-Content -Path $Script:SetupLog -Value "go,$version,optional,version_too_old"
            }
        }
    }
    else {
        Write-Warning "Go not found (optional for Go language support)"
        $Script:OptionalDeps += "go"
        Add-Content -Path $Script:SetupLog -Value "go,not_found,optional,missing"
    }
}

# Function to check Rust
function Test-Rust {
    Write-Info "Checking Rust installation..."
    
    if (Test-Command "rustc") {
        $rustVersion = rustc --version 2>$null
        if ($rustVersion -match "(\d+\.\d+\.\d+)") {
            Write-Success "Rust $($matches[1]) found"
            Add-Content -Path $Script:SetupLog -Value "rustc,$($matches[1]),optional,installed"
            
            # Check Cargo
            if (Test-Command "cargo") {
                $cargoVersion = cargo --version 2>$null
                if ($cargoVersion -match "(\d+\.\d+\.\d+)") {
                    Write-Success "Cargo $($matches[1]) found"
                    Add-Content -Path $Script:SetupLog -Value "cargo,$($matches[1]),optional,installed"
                }
            }
            else {
                Write-Warning "Cargo not found (usually comes with Rust)"
                $Script:OptionalDeps += "cargo"
                Add-Content -Path $Script:SetupLog -Value "cargo,not_found,optional,missing"
            }
        }
    }
    else {
        Write-Warning "Rust not found (optional for Rust language support)"
        $Script:OptionalDeps += "rust"
        Add-Content -Path $Script:SetupLog -Value "rust,not_found,optional,missing"
    }
}

# Function to check Docker
function Test-Docker {
    Write-Info "Checking Docker installation..."
    
    if (Test-Command "docker") {
        $dockerVersion = docker --version 2>$null
        if ($dockerVersion -match "(\d+\.\d+\.\d+)") {
            Write-Success "Docker $($matches[1]) found"
            Add-Content -Path $Script:SetupLog -Value "docker,$($matches[1]),required,installed"
            
            # Check if Docker daemon is running
            try {
                docker info >$null 2>&1
                Write-Success "Docker daemon is running"
                Add-Content -Path $Script:SetupLog -Value "docker_daemon,running,required,active"
            }
            catch {
                Write-Error "Docker daemon is not running"
                $Script:MissingDeps += "docker_daemon_running"
                Add-Content -Path $Script:SetupLog -Value "docker_daemon,not_running,required,inactive"
            }
        }
    }
    else {
        Write-Error "Docker not found (required for sandbox execution)"
        $Script:MissingDeps += "docker"
        Add-Content -Path $Script:SetupLog -Value "docker,not_found,required,missing"
    }
}

# Function to install Python dependencies
function Install-PythonDeps {
    Write-Info "Installing Python dependencies..."
    
    # Check if pip is available
    $pipCmd = $null
    if (Test-Command "pip") {
        $pipCmd = "pip"
    }
    elseif (Test-Command "pip3") {
        $pipCmd = "pip3"
    }
    else {
        Write-Error "pip not found. Please install pip first."
        return
    }
    
    try {
        # Install core dependencies
        Write-Info "Installing core lm-eval dependencies..."
        & $pipCmd install --upgrade pip
        & $pipCmd install "lm-eval[api]"
        
        # Install additional dependencies for single_turn_scenarios
        Write-Info "Installing single_turn_scenarios specific dependencies..."
        & $pipCmd install docker
        & $pipCmd install matplotlib seaborn
        & $pipCmd install pandas numpy
        & $pipCmd install pytest
        & $pipCmd install pyyaml
        & $pipCmd install jsonschema
        & $pipCmd install nltk
        & $pipCmd install rouge-score
        & $pipCmd install codebleu
        
        Write-Success "Python dependencies installed"
        Add-Content -Path $Script:SetupLog -Value "python_deps,installed,required,success"
    }
    catch {
        Write-Error "Failed to install Python dependencies: $_"
        Add-Content -Path $Script:SetupLog -Value "python_deps,failed,required,error"
    }
}

# Function to check API keys
function Test-ApiKeys {
    Write-Info "Checking API key configuration..."
    
    $envFile = ".env"
    $envTemplate = ".env.template"
    
    # Create .env.template if it doesn't exist
    if (-not (Test-Path $envTemplate)) {
        Write-Info "Creating .env.template file..."
        $templateContent = @"
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
"@
        Set-Content -Path $envTemplate -Value $templateContent
        Write-Success "Created .env.template file"
    }
    
    # Check if .env file exists
    if (-not (Test-Path $envFile)) {
        Write-Warning ".env file not found. Please copy .env.template to .env and configure your API keys."
        $Script:ApiKeysMissing += "env_file"
        Add-Content -Path $Script:SetupLog -Value "env_file,not_found,required,missing"
        return
    }
    
    # Read .env file and check API keys
    $envContent = Get-Content $envFile -ErrorAction SilentlyContinue
    $keysFound = 0
    
    # Check individual API keys
    $openaiKey = $envContent | Where-Object { $_ -match "^OPENAI_API_KEY=(.+)" } | ForEach-Object { $matches[1] }
    if ($openaiKey -and $openaiKey -ne "your_openai_api_key_here") {
        Write-Success "OpenAI API key configured"
        Add-Content -Path $Script:SetupLog -Value "openai_api_key,configured,optional,found"
        $keysFound++
    }
    else {
        Write-Warning "OpenAI API key not configured"
        $Script:ApiKeysMissing += "OPENAI_API_KEY"
        Add-Content -Path $Script:SetupLog -Value "openai_api_key,not_configured,optional,missing"
    }
    
    $anthropicKey = $envContent | Where-Object { $_ -match "^ANTHROPIC_API_KEY=(.+)" } | ForEach-Object { $matches[1] }
    if ($anthropicKey -and $anthropicKey -ne "your_anthropic_api_key_here") {
        Write-Success "Anthropic API key configured"
        Add-Content -Path $Script:SetupLog -Value "anthropic_api_key,configured,optional,found"
        $keysFound++
    }
    else {
        Write-Warning "Anthropic API key not configured"
        $Script:ApiKeysMissing += "ANTHROPIC_API_KEY"
        Add-Content -Path $Script:SetupLog -Value "anthropic_api_key,not_configured,optional,missing"
    }
    
    $deepseekKey = $envContent | Where-Object { $_ -match "^DEEPSEEK_API_KEY=(.+)" } | ForEach-Object { $matches[1] }
    if ($deepseekKey -and $deepseekKey -ne "your_deepseek_api_key_here") {
        Write-Success "DeepSeek API key configured"
        Add-Content -Path $Script:SetupLog -Value "deepseek_api_key,configured,optional,found"
        $keysFound++
    }
    else {
        Write-Warning "DeepSeek API key not configured"
        $Script:ApiKeysMissing += "DEEPSEEK_API_KEY"
        Add-Content -Path $Script:SetupLog -Value "deepseek_api_key,not_configured,optional,missing"
    }
    
    if ($keysFound -eq 0) {
        Write-Error "No API keys configured. You need at least one API key to run evaluations."
        Add-Content -Path $Script:SetupLog -Value "api_keys,none_configured,required,critical"
    }
    else {
        Write-Success "$keysFound API key(s) configured"
        Add-Content -Path $Script:SetupLog -Value "api_keys,$keysFound,optional,partial"
    }
}

# Function to generate environment report
function New-EnvironmentReport {
    Write-Info "Generating environment diagnostic report..."
    
    $reportFile = "environment_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').md"
    
    $reportContent = @"
# Environment Diagnostic Report

Generated on: $(Get-Date)
System: $($env:COMPUTERNAME) - $(Get-WmiObject -Class Win32_OperatingSystem | Select-Object -ExpandProperty Caption)

## Dependency Status

### Required Dependencies
"@
    
    # Add dependency status to report
    $logContent = Get-Content $Script:SetupLog -ErrorAction SilentlyContinue
    foreach ($line in $logContent) {
        if ($line -match "^([^,]+),([^,]+),([^,]+),([^,]+)$") {
            $component = $matches[1]
            $version = $matches[2]
            $type = $matches[3]
            $status = $matches[4]
            
            if ($type -eq "required") {
                switch ($status) {
                    "installed" { $reportContent += "`n- ✅ $component`: $version" }
                    { $_ -in @("missing", "not_found") } { $reportContent += "`n- ❌ $component`: Not installed" }
                    "version_too_old" { $reportContent += "`n- ⚠️ $component`: $version (version too old)" }
                    default { $reportContent += "`n- ❓ $component`: $status" }
                }
            }
        }
    }
    
    $reportContent += "`n`n### Optional Dependencies"
    
    foreach ($line in $logContent) {
        if ($line -match "^([^,]+),([^,]+),([^,]+),([^,]+)$") {
            $component = $matches[1]
            $version = $matches[2]
            $type = $matches[3]
            $status = $matches[4]
            
            if ($type -eq "optional") {
                switch ($status) {
                    "installed" { $reportContent += "`n- ✅ $component`: $version" }
                    { $_ -in @("missing", "not_found") } { $reportContent += "`n- ⚪ $component`: Not installed (optional)" }
                    "version_too_old" { $reportContent += "`n- ⚠️ $component`: $version (older version)" }
                    default { $reportContent += "`n- ❓ $component`: $status" }
                }
            }
        }
    }
    
    # Add installation instructions for missing dependencies
    if ($Script:MissingDeps.Count -gt 0) {
        $reportContent += "`n`n## Missing Required Dependencies`n"
        $reportContent += "`nThe following required dependencies are missing:`n"
        
        foreach ($dep in $Script:MissingDeps) {
            $reportContent += "`n- $dep"
        }
        
        $reportContent += "`n`n### Installation Instructions`n"
        $reportContent += "`n#### Windows:`n"
        $reportContent += "``````powershell`n"
        
        foreach ($dep in $Script:MissingDeps) {
            switch -Regex ($dep) {
                "python" { $reportContent += "# Install Python from https://python.org or use winget:`nwinget install Python.Python.3`n" }
                "node" { $reportContent += "# Install Node.js from https://nodejs.org or use winget:`nwinget install OpenJS.NodeJS`n" }
                "java" { $reportContent += "# Install Java from https://adoptium.net or use winget:`nwinget install Eclipse.Temurin.11.JDK`n" }
                "cpp_compiler" { $reportContent += "# Install Visual Studio Build Tools or Visual Studio Community`nwinget install Microsoft.VisualStudio.2022.BuildTools`n" }
                "docker" { $reportContent += "# Install Docker Desktop from https://docker.com or use winget:`nwinget install Docker.DockerDesktop`n" }
            }
        }
        
        $reportContent += "``````"
    }
    
    # Add API key configuration section
    if ($Script:ApiKeysMissing.Count -gt 0) {
        $reportContent += "`n`n## API Key Configuration`n"
        $reportContent += "`nConfigure your API keys in the .env file:`n"
        $reportContent += "``````powershell`n"
        $reportContent += "Copy-Item .env.template .env`n"
        $reportContent += "# Edit .env file with your actual API keys`n"
        $reportContent += "``````"
    }
    
    Set-Content -Path $reportFile -Value $reportContent
    Write-Success "Environment report generated: $reportFile"
    Add-Content -Path $Script:SetupLog -Value "report,$reportFile,info,generated"
}

# Function to display summary
function Show-Summary {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "         SETUP SUMMARY" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    
    if ($Script:MissingDeps.Count -eq 0) {
        Write-Success "All required dependencies are installed!"
    }
    else {
        Write-Error "Missing $($Script:MissingDeps.Count) required dependencies:"
        foreach ($dep in $Script:MissingDeps) {
            Write-Host "  - $dep" -ForegroundColor Red
        }
    }
    
    if ($Script:OptionalDeps.Count -gt 0) {
        Write-Warning "Missing $($Script:OptionalDeps.Count) optional dependencies:"
        foreach ($dep in $Script:OptionalDeps) {
            Write-Host "  - $dep" -ForegroundColor Yellow
        }
    }
    
    if ($Script:ApiKeysMissing.Count -gt 0) {
        Write-Warning "API keys need configuration:"
        foreach ($key in $Script:ApiKeysMissing) {
            Write-Host "  - $key" -ForegroundColor Yellow
        }
    }
    
    Write-Host ""
    Write-Host "Setup log: $Script:SetupLog" -ForegroundColor Gray
    Write-Host "Environment report: environment_report_*.md" -ForegroundColor Gray
    Write-Host ""
    
    if ($Script:MissingDeps.Count -eq 0) {
        Write-Success "Environment setup complete! You can now run evaluations."
        return $true
    }
    else {
        Write-Error "Please install missing dependencies before running evaluations."
        return $false
    }
}

# Function to show help
function Show-Help {
    Write-Host "Single Turn Scenarios - Environment Setup Script (PowerShell)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\setupEvaluationEnvironment.ps1 [OPTIONS]" -ForegroundColor White
    Write-Host ""
    Write-Host "Options:" -ForegroundColor White
    Write-Host "  -Help               Show this help message" -ForegroundColor Gray
    Write-Host "  -InstallDeps        Install Python dependencies automatically" -ForegroundColor Gray
    Write-Host ""
    Write-Host "This script checks for required dependencies and generates a diagnostic report." -ForegroundColor White
    Write-Host "Required dependencies: Python 3.8+, Node.js 16+, Java 11+, C/C++ compiler, Docker" -ForegroundColor White
    Write-Host "Optional dependencies: Go 1.18+, Rust" -ForegroundColor White
    Write-Host ""
}

# Main function
function Main {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "  Single Turn Scenarios - Environment Setup" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    
    if ($Help) {
        Show-Help
        return
    }
    
    Write-Info "Starting environment setup..."
    Write-Info "Setup log: $Script:SetupLog"
    Write-Host ""
    
    # Initialize log file
    Set-Content -Path $Script:SetupLog -Value "component,version,type,status"
    
    # Check all dependencies
    Test-Python
    Test-Node
    Test-Java
    Test-CppCompilers
    Test-Go
    Test-Rust
    Test-Docker
    
    Write-Host ""
    
    # Install Python dependencies if requested
    if ($InstallDeps) {
        Install-PythonDeps
        Write-Host ""
    }
    
    # Check API keys
    Test-ApiKeys
    
    Write-Host ""
    
    # Generate report
    New-EnvironmentReport
    
    # Display summary
    $success = Show-Summary
    
    if ($success) {
        exit 0
    }
    else {
        exit 1
    }
}

# Run main function
Main