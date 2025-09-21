#!/usr/bin/env python3

import os
import shutil
import argparse
import json
from pathlib import Path


def setup_output_directories(problems_file: str = None):
    """Create output directory structure for all problems."""
    if problems_file is None:
        problems_file = os.path.join(os.path.dirname(__file__), "problems.jsonl")
    
    # Create main output directory
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load problems and create directories
    with open(problems_file, 'r') as f:
        problems = json.load(f)
    
    created_dirs = []
    for problem in problems:
        problem_id = problem['problem_id']
        problem_dir = os.path.join(output_dir, problem_id)
        src_dir = os.path.join(problem_dir, "src")
        
        os.makedirs(problem_dir, exist_ok=True)
        os.makedirs(src_dir, exist_ok=True)
        
        created_dirs.append(problem_dir)
    
    print(f"✅ Created {len(created_dirs)} problem directories in {output_dir}")
    return created_dirs


def cleanup_output_directories(confirm: bool = False):
    """Clean up all output directories."""
    output_dir = "./output"
    
    if not os.path.exists(output_dir):
        print("No output directory found to clean up.")
        return
    
    if not confirm:
        response = input(f"Are you sure you want to delete {output_dir} and all its contents? (y/N): ")
        if response.lower() != 'y':
            print("Cleanup cancelled.")
            return
    
    try:
        shutil.rmtree(output_dir)
        print(f"✅ Cleaned up {output_dir}")
    except Exception as e:
        print(f"❌ Error cleaning up {output_dir}: {e}")


def validate_environment():
    """Validate that the environment is set up correctly."""
    print("🔍 Validating environment...")
    
    # Check Python version
    import sys
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check required packages
    required_packages = [
        'claude_code_sdk',
        'datasets', 
        'pytest',
        'ast',
        'subprocess'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # Check Claude Code SDK
    try:
        import claude_code_sdk
        print("✅ Claude Code SDK available")
    except ImportError:
        print("❌ Claude Code SDK not available")
        print("Install with: pip install claude-code-sdk")
        return False
    
    # Check API key
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if api_key:
        print("✅ ANTHROPIC_API_KEY set")
    else:
        print("⚠️  ANTHROPIC_API_KEY not set (required for Claude Code)")
        print("Set with: export ANTHROPIC_API_KEY='your_key_here'")
    
    return len(missing_packages) == 0


def create_sample_config():
    """Create a sample configuration file."""
    config = {
        "model": "claude-3-haiku-20240307",
        "debug": False,
        "allowed_tools": ["Bash", "Python", "FileEditor"],
        "cwd": "./output",
        "max_problems": 5,
        "timeout_per_phase": 300,
        "cleanup_between_runs": True
    }
    
    config_path = "multi_turn_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Created sample config: {config_path}")
    return config_path


def run_quick_test():
    """Run a quick test to verify everything works."""
    print("🧪 Running quick test...")
    
    try:
        # Test dataset loading
        from utils import load_dataset
        dataset = load_dataset()
        print(f"✅ Dataset loaded: {len(dataset['test'])} problems")
        
        # Test metrics import
        from custom_metrics import file_existence_check
        print("✅ Custom metrics imported")
        
        # Test Claude Code SDK
        import claude_code_sdk
        options = claude_code_sdk.ClaudeCodeOptions(model="claude-3-haiku-20240307")
        print("✅ Claude Code SDK initialized")
        
        print("🎉 Quick test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup multi-turn coding evaluation")
    parser.add_argument("--setup", action="store_true", help="Setup output directories")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup output directories")
    parser.add_argument("--validate", action="store_true", help="Validate environment")
    parser.add_argument("--config", action="store_true", help="Create sample config")
    parser.add_argument("--test", action="store_true", help="Run quick test")
    parser.add_argument("--all", action="store_true", help="Run all setup steps")
    parser.add_argument("--force", action="store_true", help="Force cleanup without confirmation")
    
    args = parser.parse_args()
    
    if args.all:
        print("🚀 Running full setup...")
        validate_environment()
        setup_output_directories()
        create_sample_config()
        run_quick_test()
        print("\n✅ Setup complete!")
        print("\n🚀 Next step: Run simple test")
        print("./simple_test.sh")
        
    elif args.setup:
        setup_output_directories()
        
    elif args.cleanup:
        cleanup_output_directories(confirm=args.force)
        
    elif args.validate:
        if validate_environment():
            print("\n✅ Environment validation passed!")
        else:
            print("\n❌ Environment validation failed!")
            
    elif args.config:
        create_sample_config()
        
    elif args.test:
        run_quick_test()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()