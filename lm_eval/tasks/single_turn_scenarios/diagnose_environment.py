#!/usr/bin/env python3
"""
Environment Diagnostic Script

This script provides comprehensive environment diagnostics including
system information, dependency versions, and configuration status.
"""

import os
import sys
import json
import platform
import subprocess
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class DiagnosticInfo:
    """Container for diagnostic information."""
    timestamp: str
    system_info: Dict[str, Any]
    python_info: Dict[str, Any]
    dependencies: Dict[str, Any]
    executables: Dict[str, Any]
    environment_vars: Dict[str, Any]
    configuration_status: Dict[str, Any]
    recommendations: List[str]

class EnvironmentDiagnostic:
    """Environment diagnostic collector."""
    
    def __init__(self):
        self.current_dir = Path(__file__).parent
        self.recommendations = []
    
    def get_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        try:
            return {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "hostname": platform.node(),
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
                "python_compiler": platform.python_compiler(),
                "python_build": platform.python_build(),
                "cpu_count": os.cpu_count(),
                "current_directory": str(Path.cwd()),
                "script_directory": str(self.current_dir),
                "home_directory": str(Path.home()),
                "path_separator": os.pathsep,
                "line_separator": repr(os.linesep)
            }
        except Exception as e:
            return {"error": f"Failed to collect system info: {str(e)}"}
    
    def get_python_info(self) -> Dict[str, Any]:
        """Collect Python-specific information."""
        try:
            info = {
                "version": sys.version,
                "version_info": {
                    "major": sys.version_info.major,
                    "minor": sys.version_info.minor,
                    "micro": sys.version_info.micro,
                    "releaselevel": sys.version_info.releaselevel,
                    "serial": sys.version_info.serial
                },
                "executable": sys.executable,
                "prefix": sys.prefix,
                "exec_prefix": sys.exec_prefix,
                "path": sys.path[:10],  # First 10 entries
                "modules": list(sys.modules.keys())[:20],  # First 20 modules
                "platform": sys.platform,
                "maxsize": sys.maxsize,
                "float_info": {
                    "max": sys.float_info.max,
                    "min": sys.float_info.min,
                    "epsilon": sys.float_info.epsilon
                }
            }
            
            # Check if we're in a virtual environment
            if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
                info["virtual_env"] = True
                info["virtual_env_path"] = os.environ.get("VIRTUAL_ENV", "Unknown")
            else:
                info["virtual_env"] = False
            
            return info
        except Exception as e:
            return {"error": f"Failed to collect Python info: {str(e)}"}
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check Python package dependencies."""
        dependencies = {}
        
        # Core dependencies
        core_packages = [
            "lm_eval", "datasets", "transformers", "torch", "numpy", "pandas",
            "requests", "tqdm", "pyyaml", "jsonschema"
        ]
        
        # Optional dependencies
        optional_packages = [
            "docker", "matplotlib", "seaborn", "plotly", "nltk", "rouge_score",
            "codebleu", "pytest", "black", "flake8", "mypy"
        ]
        
        # Model-specific dependencies
        model_packages = [
            "openai", "anthropic", "cohere", "huggingface_hub", "tiktoken"
        ]
        
        all_packages = {
            "core": core_packages,
            "optional": optional_packages,
            "model_specific": model_packages
        }
        
        for category, packages in all_packages.items():
            dependencies[category] = {}
            
            for package in packages:
                try:
                    module = importlib.import_module(package)
                    version = getattr(module, "__version__", "Unknown")
                    dependencies[category][package] = {
                        "status": "installed",
                        "version": version,
                        "location": getattr(module, "__file__", "Unknown")
                    }
                except ImportError:
                    dependencies[category][package] = {
                        "status": "not_installed",
                        "version": None,
                        "location": None
                    }
                except Exception as e:
                    dependencies[category][package] = {
                        "status": "error",
                        "version": None,
                        "location": None,
                        "error": str(e)
                    }
        
        return dependencies
    
    def check_executables(self) -> Dict[str, Any]:
        """Check external executable dependencies."""
        executables = {}
        
        # Required executables
        required_exes = {
            "python": ["--version"],
            "python3": ["--version"],
            "pip": ["--version"],
            "pip3": ["--version"],
            "node": ["--version"],
            "npm": ["--version"],
            "java": ["-version"],
            "javac": ["-version"],
            "docker": ["--version"],
            "git": ["--version"]
        }
        
        # Optional executables
        optional_exes = {
            "gcc": ["--version"],
            "g++": ["--version"],
            "clang": ["--version"],
            "go": ["version"],
            "rustc": ["--version"],
            "cargo": ["--version"]
        }
        
        all_exes = {
            "required": required_exes,
            "optional": optional_exes
        }
        
        for category, exes in all_exes.items():
            executables[category] = {}
            
            for exe, args in exes.items():
                try:
                    result = subprocess.run(
                        [exe] + args,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0:
                        version_output = result.stdout.strip() or result.stderr.strip()
                        executables[category][exe] = {
                            "status": "available",
                            "version": version_output.split('\n')[0],
                            "path": self._which(exe)
                        }
                    else:
                        executables[category][exe] = {
                            "status": "error",
                            "version": None,
                            "path": self._which(exe),
                            "error": result.stderr.strip()
                        }
                        
                except subprocess.TimeoutExpired:
                    executables[category][exe] = {
                        "status": "timeout",
                        "version": None,
                        "path": self._which(exe)
                    }
                except FileNotFoundError:
                    executables[category][exe] = {
                        "status": "not_found",
                        "version": None,
                        "path": None
                    }
                except Exception as e:
                    executables[category][exe] = {
                        "status": "error",
                        "version": None,
                        "path": self._which(exe),
                        "error": str(e)
                    }
        
        return executables
    
    def _which(self, executable: str) -> Optional[str]:
        """Find the path to an executable."""
        try:
            result = subprocess.run(
                ["which", executable] if os.name != 'nt' else ["where", executable],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except:
            pass
        return None
    
    def check_environment_vars(self) -> Dict[str, Any]:
        """Check relevant environment variables."""
        relevant_vars = [
            "PATH", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV",
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
            "HUGGINGFACE_API_KEY", "COHERE_API_KEY",
            "DOCKER_HOST", "DOCKER_CERT_PATH", "DOCKER_TLS_VERIFY",
            "HOME", "USER", "SHELL", "TERM"
        ]
        
        env_vars = {}
        
        for var in relevant_vars:
            value = os.environ.get(var)
            if value is not None:
                # Mask API keys for security
                if "API_KEY" in var and value:
                    if len(value) > 8:
                        masked_value = value[:4] + "*" * (len(value) - 8) + value[-4:]
                    else:
                        masked_value = "*" * len(value)
                    env_vars[var] = {
                        "set": True,
                        "value": masked_value,
                        "length": len(value)
                    }
                else:
                    env_vars[var] = {
                        "set": True,
                        "value": value[:100] + "..." if len(value) > 100 else value,
                        "length": len(value)
                    }
            else:
                env_vars[var] = {
                    "set": False,
                    "value": None,
                    "length": 0
                }
        
        return env_vars
    
    def check_configuration_status(self) -> Dict[str, Any]:
        """Check configuration file status."""
        config_status = {}
        
        # Check configuration files
        config_files = [
            ".env",
            ".env.template", 
            "context_configs.json",
            "problems.jsonl",
            "single_turn_scenarios_suite.yaml"
        ]
        
        for config_file in config_files:
            file_path = self.current_dir / config_file
            if file_path.exists():
                try:
                    stat = file_path.stat()
                    config_status[config_file] = {
                        "exists": True,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "readable": os.access(file_path, os.R_OK),
                        "writable": os.access(file_path, os.W_OK)
                    }
                except Exception as e:
                    config_status[config_file] = {
                        "exists": True,
                        "error": str(e)
                    }
            else:
                config_status[config_file] = {
                    "exists": False
                }
        
        # Check directories
        directories = [
            "model_configs",
            "docker", 
            "tests",
            "analysis_tools"
        ]
        
        for directory in directories:
            dir_path = self.current_dir / directory
            if dir_path.exists() and dir_path.is_dir():
                try:
                    files = list(dir_path.iterdir())
                    config_status[f"{directory}_dir"] = {
                        "exists": True,
                        "file_count": len(files),
                        "files": [f.name for f in files[:10]]  # First 10 files
                    }
                except Exception as e:
                    config_status[f"{directory}_dir"] = {
                        "exists": True,
                        "error": str(e)
                    }
            else:
                config_status[f"{directory}_dir"] = {
                    "exists": False
                }
        
        return config_status
    
    def generate_recommendations(self, diagnostic_info: DiagnosticInfo) -> List[str]:
        """Generate recommendations based on diagnostic results."""
        recommendations = []
        
        # Check Python version
        python_version = diagnostic_info.python_info.get("version_info", {})
        if python_version.get("major") == 3 and python_version.get("minor", 0) < 8:
            recommendations.append("Upgrade Python to version 3.8 or higher")
        
        # Check core dependencies
        core_deps = diagnostic_info.dependencies.get("core", {})
        missing_core = [pkg for pkg, info in core_deps.items() 
                       if info.get("status") != "installed"]
        if missing_core:
            recommendations.append(f"Install missing core dependencies: {', '.join(missing_core)}")
        
        # Check Docker
        docker_info = diagnostic_info.executables.get("required", {}).get("docker", {})
        if docker_info.get("status") != "available":
            recommendations.append("Install Docker for sandbox execution")
        
        # Check API keys
        env_vars = diagnostic_info.environment_vars
        api_keys = [var for var in env_vars if "API_KEY" in var and env_vars[var].get("set")]
        if not api_keys:
            recommendations.append("Configure at least one API key in .env file")
        
        # Check configuration files
        config_status = diagnostic_info.configuration_status
        if not config_status.get("problems.jsonl", {}).get("exists"):
            recommendations.append("problems.jsonl dataset file is missing")
        
        if not config_status.get("context_configs.json", {}).get("exists"):
            recommendations.append("context_configs.json configuration file is missing")
        
        # Check virtual environment
        if not diagnostic_info.python_info.get("virtual_env"):
            recommendations.append("Consider using a virtual environment for better dependency isolation")
        
        return recommendations
    
    def run_diagnostics(self) -> DiagnosticInfo:
        """Run complete environment diagnostics."""
        print("🔍 Running Environment Diagnostics")
        print("=" * 50)
        
        print("📊 Collecting system information...")
        system_info = self.get_system_info()
        
        print("🐍 Collecting Python information...")
        python_info = self.get_python_info()
        
        print("📦 Checking dependencies...")
        dependencies = self.check_dependencies()
        
        print("⚙️  Checking executables...")
        executables = self.check_executables()
        
        print("🌍 Checking environment variables...")
        environment_vars = self.check_environment_vars()
        
        print("📋 Checking configuration status...")
        configuration_status = self.check_configuration_status()
        
        # Create diagnostic info
        diagnostic_info = DiagnosticInfo(
            timestamp=datetime.now().isoformat(),
            system_info=system_info,
            python_info=python_info,
            dependencies=dependencies,
            executables=executables,
            environment_vars=environment_vars,
            configuration_status=configuration_status,
            recommendations=[]
        )
        
        print("💡 Generating recommendations...")
        diagnostic_info.recommendations = self.generate_recommendations(diagnostic_info)
        
        return diagnostic_info
    
    def print_summary(self, diagnostic_info: DiagnosticInfo):
        """Print diagnostic summary."""
        print("\n" + "=" * 50)
        print("📋 Diagnostic Summary")
        print("=" * 50)
        
        # System info
        system = diagnostic_info.system_info
        print(f"🖥️  System: {system.get('system')} {system.get('release')}")
        print(f"🐍 Python: {diagnostic_info.python_info.get('version', 'Unknown').split()[0]}")
        
        # Dependencies summary
        deps = diagnostic_info.dependencies
        core_installed = sum(1 for info in deps.get("core", {}).values() 
                           if info.get("status") == "installed")
        core_total = len(deps.get("core", {}))
        print(f"📦 Core Dependencies: {core_installed}/{core_total} installed")
        
        # Executables summary
        exes = diagnostic_info.executables
        required_available = sum(1 for info in exes.get("required", {}).values() 
                               if info.get("status") == "available")
        required_total = len(exes.get("required", {}))
        print(f"⚙️  Required Executables: {required_available}/{required_total} available")
        
        # Configuration summary
        config = diagnostic_info.configuration_status
        config_files = ["problems.jsonl", "context_configs.json", ".env"]
        config_present = sum(1 for file in config_files 
                           if config.get(file, {}).get("exists"))
        print(f"📋 Configuration Files: {config_present}/{len(config_files)} present")
        
        # API keys
        env_vars = diagnostic_info.environment_vars
        api_keys_set = sum(1 for var in env_vars 
                          if "API_KEY" in var and env_vars[var].get("set"))
        print(f"🔑 API Keys: {api_keys_set} configured")
        
        # Recommendations
        if diagnostic_info.recommendations:
            print(f"\n💡 Recommendations ({len(diagnostic_info.recommendations)}):")
            for i, rec in enumerate(diagnostic_info.recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print("\n✅ No recommendations - environment looks good!")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Environment Diagnostic Tool")
    parser.add_argument("--output", "-o", help="Output diagnostic report (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--summary-only", "-s", action="store_true", help="Show summary only")
    
    args = parser.parse_args()
    
    diagnostic = EnvironmentDiagnostic()
    info = diagnostic.run_diagnostics()
    
    if not args.summary_only:
        diagnostic.print_summary(info)
    
    if args.verbose and not args.summary_only:
        print("\n" + "=" * 50)
        print("📊 Detailed Information")
        print("=" * 50)
        print(json.dumps(asdict(info), indent=2))
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(asdict(info), f, indent=2)
        print(f"\n📄 Diagnostic report saved to: {args.output}")
    
    # Exit code based on critical issues
    critical_issues = 0
    
    # Check for critical missing dependencies
    core_deps = info.dependencies.get("core", {})
    missing_critical = ["lm_eval", "datasets", "transformers"]
    for dep in missing_critical:
        if core_deps.get(dep, {}).get("status") != "installed":
            critical_issues += 1
    
    # Check for Docker
    docker_status = info.executables.get("required", {}).get("docker", {}).get("status")
    if docker_status != "available":
        critical_issues += 1
    
    if critical_issues > 0:
        print(f"\n❌ {critical_issues} critical issues found")
        sys.exit(1)
    else:
        print("\n✅ Environment diagnostic complete")
        sys.exit(0)

if __name__ == "__main__":
    main()