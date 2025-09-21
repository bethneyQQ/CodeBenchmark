#!/usr/bin/env python3
"""
Dependency checker for Multi-Turn Coding Evaluation
Run this script to verify all required packages are installed correctly.
"""

import sys
import importlib
from typing import Dict, List, Tuple

def check_package(package_name: str, import_name: str = None, required: bool = True) -> Tuple[bool, str]:
    """Check if a package is available and return status."""
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)

def main():
    """Main dependency checking function."""
    
    print("🔍 Multi-Turn Coding Evaluation - Dependency Check")
    print("=" * 60)
    
    # Define packages to check
    packages = {
        'Core (Required)': [
            ('claude-code-sdk', 'claude_code_sdk', True),
            ('datasets', 'datasets', True),
            ('pytest', 'pytest', True),
        ],
        'Analysis (Recommended)': [
            ('seaborn', 'seaborn', False),
            ('matplotlib', 'matplotlib', False),
            ('pandas', 'pandas', False),
            ('numpy', 'numpy', False),
        ],
        'ML/Data Science (Optional)': [
            ('torch', 'torch', False),
            ('transformers', 'transformers', False),
            ('scikit-learn', 'sklearn', False),
        ],
        'Utilities (Optional)': [
            ('tqdm', 'tqdm', False),
            ('requests', 'requests', False),
        ]
    }
    
    all_good = True
    missing_required = []
    missing_recommended = []
    
    for category, package_list in packages.items():
        print(f"\n📦 {category}")
        print("-" * 40)
        
        for package_name, import_name, required in package_list:
            available, version_or_error = check_package(package_name, import_name, required)
            
            if available:
                status = "✅"
                info = f"v{version_or_error}" if version_or_error != 'unknown' else "installed"
            else:
                if required:
                    status = "❌"
                    missing_required.append(package_name)
                    all_good = False
                else:
                    status = "⚠️ "
                    missing_recommended.append(package_name)
                info = "not installed"
            
            print(f"  {status} {package_name:<20} {info}")
    
    # Summary
    print(f"\n📊 Summary")
    print("=" * 60)
    
    if all_good and not missing_required:
        print("✅ All required packages are installed!")
    else:
        print("❌ Some required packages are missing!")
    
    if missing_required:
        print(f"\n🚨 Missing Required Packages:")
        for pkg in missing_required:
            print(f"   - {pkg}")
        print(f"\n💡 Install with:")
        print(f"   pip install {' '.join(missing_required)}")
    
    if missing_recommended:
        print(f"\n⚠️  Missing Recommended Packages:")
        for pkg in missing_recommended:
            print(f"   - {pkg}")
        print(f"\n💡 Install with:")
        print(f"   pip install {' '.join(missing_recommended)}")
    
    # Installation commands
    print(f"\n🛠️  Installation Commands")
    print("-" * 40)
    print("# Install all at once:")
    print("pip install -r requirements.txt")
    print()
    print("# Install core only:")
    print("pip install claude-code-sdk datasets pytest")
    print()
    print("# Install analysis packages:")
    print("pip install seaborn matplotlib pandas numpy")
    print()
    print("# Install ML packages:")
    print("pip install torch transformers scikit-learn")
    
    # Next steps
    print(f"\n🚀 Next Steps")
    print("-" * 40)
    if all_good:
        print("1. Set your API key: export ANTHROPIC_API_KEY='your_key_here'")
        print("2. Run setup: python setup_evaluation.py --all")
        print("3. Test evaluation: ./simple_test.sh --difficulty easy")
    else:
        print("1. Install missing packages (see commands above)")
        print("2. Re-run this check: python check_dependencies.py")
        print("3. Continue with setup once all packages are installed")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())