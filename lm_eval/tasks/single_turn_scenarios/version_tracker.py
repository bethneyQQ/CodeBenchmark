#!/usr/bin/env python3
"""
Version Tracking and Dependency Management

This script tracks versions of dependencies, configurations, and datasets
to ensure reproducible evaluations and proper version control.
"""

import os
import sys
import json
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class VersionInfo:
    """Version information for a component."""
    name: str
    version: str
    source: str  # "pip", "git", "file", "system"
    location: Optional[str] = None
    hash: Optional[str] = None
    last_modified: Optional[str] = None

@dataclass
class VersionSnapshot:
    """Complete version snapshot of the environment."""
    timestamp: str
    snapshot_hash: str
    python_packages: List[VersionInfo]
    system_executables: List[VersionInfo]
    configuration_files: List[VersionInfo]
    dataset_files: List[VersionInfo]
    git_info: Dict[str, Any]
    environment_vars: Dict[str, str]

class VersionTracker:
    """Version tracking and management."""
    
    def __init__(self, project_root: Path = None):
        self.current_dir = Path(__file__).parent
        self.project_root = project_root or self.current_dir.parent.parent.parent
        self.versions_dir = self.current_dir / "versions"
        self.versions_dir.mkdir(exist_ok=True)
    
    def get_python_packages(self) -> List[VersionInfo]:
        """Get versions of installed Python packages."""
        packages = []
        
        try:
            # Get pip list output
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                pip_packages = json.loads(result.stdout)
                
                for pkg in pip_packages:
                    packages.append(VersionInfo(
                        name=pkg["name"],
                        version=pkg["version"],
                        source="pip",
                        location=self._get_package_location(pkg["name"])
                    ))
            
        except Exception as e:
            print(f"Warning: Could not get pip packages: {e}")
        
        # Add key packages with additional info
        key_packages = [
            "lm_eval", "torch", "transformers", "datasets", "numpy", "pandas",
            "docker", "matplotlib", "seaborn", "pytest", "jsonschema"
        ]
        
        for pkg_name in key_packages:
            try:
                import importlib
                module = importlib.import_module(pkg_name)
                version = getattr(module, "__version__", "unknown")
                location = getattr(module, "__file__", None)
                
                # Update existing entry or add new one
                existing = next((p for p in packages if p.name == pkg_name), None)
                if existing:
                    existing.location = location
                else:
                    packages.append(VersionInfo(
                        name=pkg_name,
                        version=version,
                        source="import",
                        location=location
                    ))
                    
            except ImportError:
                pass
        
        return packages
    
    def _get_package_location(self, package_name: str) -> Optional[str]:
        """Get the installation location of a package."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", package_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Location:'):
                        return line.split(':', 1)[1].strip()
        except:
            pass
        return None
    
    def get_system_executables(self) -> List[VersionInfo]:
        """Get versions of system executables."""
        executables = []
        
        # Key executables to track
        exe_configs = {
            "python": ["--version"],
            "python3": ["--version"],
            "node": ["--version"],
            "npm": ["--version"],
            "java": ["-version"],
            "javac": ["-version"],
            "docker": ["--version"],
            "git": ["--version"],
            "gcc": ["--version"],
            "g++": ["--version"],
            "clang": ["--version"],
            "go": ["version"],
            "rustc": ["--version"],
            "cargo": ["--version"]
        }
        
        for exe_name, version_args in exe_configs.items():
            try:
                result = subprocess.run(
                    [exe_name] + version_args,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    version_output = result.stdout.strip() or result.stderr.strip()
                    version_line = version_output.split('\n')[0]
                    
                    # Extract version number
                    version = self._extract_version(version_line)
                    
                    # Get executable location
                    location = self._which(exe_name)
                    
                    executables.append(VersionInfo(
                        name=exe_name,
                        version=version,
                        source="system",
                        location=location
                    ))
                    
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        return executables
    
    def _extract_version(self, version_string: str) -> str:
        """Extract version number from version string."""
        import re
        
        # Common version patterns
        patterns = [
            r'(\d+\.\d+\.\d+)',  # x.y.z
            r'(\d+\.\d+)',       # x.y
            r'v(\d+\.\d+\.\d+)', # vx.y.z
            r'version (\d+\.\d+\.\d+)',  # version x.y.z
        ]
        
        for pattern in patterns:
            match = re.search(pattern, version_string)
            if match:
                return match.group(1)
        
        # If no pattern matches, return first word that looks like a version
        words = version_string.split()
        for word in words:
            if re.match(r'\d+\.\d+', word):
                return word
        
        return version_string[:50]  # Truncate if too long
    
    def _which(self, executable: str) -> Optional[str]:
        """Find the path to an executable."""
        try:
            if os.name == 'nt':  # Windows
                result = subprocess.run(
                    ["where", executable],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            else:  # Unix-like
                result = subprocess.run(
                    ["which", executable],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except:
            pass
        return None
    
    def get_configuration_files(self) -> List[VersionInfo]:
        """Get version info for configuration files."""
        config_files = []
        
        # Configuration files to track
        files_to_track = [
            "context_configs.json",
            "single_turn_scenarios_suite.yaml",
            ".env.template",
            "setupEvaluationEnvironment.sh",
            "setupEvaluationEnvironment.ps1"
        ]
        
        # Add model config files
        model_configs_dir = self.current_dir / "model_configs"
        if model_configs_dir.exists():
            files_to_track.extend([f"model_configs/{f.name}" for f in model_configs_dir.glob("*.yaml")])
        
        # Add task config files
        task_configs = list(self.current_dir.glob("*.yaml"))
        files_to_track.extend([f.name for f in task_configs])
        
        for file_path in files_to_track:
            full_path = self.current_dir / file_path
            if full_path.exists():
                try:
                    stat = full_path.stat()
                    
                    # Calculate file hash
                    with open(full_path, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
                    
                    config_files.append(VersionInfo(
                        name=file_path,
                        version=f"modified_{datetime.fromtimestamp(stat.st_mtime).strftime('%Y%m%d_%H%M%S')}",
                        source="file",
                        location=str(full_path),
                        hash=file_hash,
                        last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat()
                    ))
                    
                except Exception as e:
                    print(f"Warning: Could not process {file_path}: {e}")
        
        return config_files
    
    def get_dataset_files(self) -> List[VersionInfo]:
        """Get version info for dataset files."""
        dataset_files = []
        
        # Dataset files to track
        files_to_track = [
            "problems.jsonl"
        ]
        
        # Add test files
        tests_dir = self.current_dir / "tests"
        if tests_dir.exists():
            test_files = list(tests_dir.glob("test_*.py")) + list(tests_dir.glob("test_*.js")) + list(tests_dir.glob("test_*.java"))
            files_to_track.extend([f"tests/{f.name}" for f in test_files[:10]])  # First 10 test files
        
        for file_path in files_to_track:
            full_path = self.current_dir / file_path
            if full_path.exists():
                try:
                    stat = full_path.stat()
                    
                    # Calculate file hash
                    with open(full_path, 'rb') as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
                    
                    # For JSONL files, also count lines
                    version_info = f"size_{stat.st_size}"
                    if file_path.endswith('.jsonl'):
                        with open(full_path, 'r', encoding='utf-8') as f:
                            line_count = sum(1 for line in f if line.strip())
                        version_info = f"lines_{line_count}_size_{stat.st_size}"
                    
                    dataset_files.append(VersionInfo(
                        name=file_path,
                        version=version_info,
                        source="file",
                        location=str(full_path),
                        hash=file_hash,
                        last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat()
                    ))
                    
                except Exception as e:
                    print(f"Warning: Could not process {file_path}: {e}")
        
        return dataset_files
    
    def get_git_info(self) -> Dict[str, Any]:
        """Get Git repository information."""
        git_info = {}
        
        try:
            # Check if we're in a git repository
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                # Get commit hash
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=self.project_root
                )
                if result.returncode == 0:
                    git_info["commit_hash"] = result.stdout.strip()
                    git_info["short_hash"] = result.stdout.strip()[:8]
                
                # Get branch name
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=self.project_root
                )
                if result.returncode == 0:
                    git_info["branch"] = result.stdout.strip()
                
                # Get remote URL
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=self.project_root
                )
                if result.returncode == 0:
                    git_info["remote_url"] = result.stdout.strip()
                
                # Check for uncommitted changes
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=self.project_root
                )
                if result.returncode == 0:
                    git_info["has_uncommitted_changes"] = bool(result.stdout.strip())
                    git_info["uncommitted_files"] = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                
                # Get last commit info
                result = subprocess.run(
                    ["git", "log", "-1", "--format=%H|%an|%ae|%ad|%s"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=self.project_root
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split('|')
                    if len(parts) >= 5:
                        git_info["last_commit"] = {
                            "hash": parts[0],
                            "author_name": parts[1],
                            "author_email": parts[2],
                            "date": parts[3],
                            "message": parts[4]
                        }
                
        except Exception as e:
            git_info["error"] = f"Could not get git info: {e}"
        
        return git_info
    
    def get_environment_vars(self) -> Dict[str, str]:
        """Get relevant environment variables."""
        relevant_vars = [
            "PYTHONPATH", "PATH", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV",
            "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM"
        ]
        
        env_vars = {}
        for var in relevant_vars:
            value = os.environ.get(var)
            if value is not None:
                # Truncate very long values
                if len(value) > 200:
                    env_vars[var] = value[:200] + "..."
                else:
                    env_vars[var] = value
        
        return env_vars
    
    def create_version_snapshot(self) -> VersionSnapshot:
        """Create a complete version snapshot."""
        print("📸 Creating version snapshot...")
        
        # Collect all version information
        python_packages = self.get_python_packages()
        system_executables = self.get_system_executables()
        configuration_files = self.get_configuration_files()
        dataset_files = self.get_dataset_files()
        git_info = self.get_git_info()
        environment_vars = self.get_environment_vars()
        
        # Create snapshot
        snapshot = VersionSnapshot(
            timestamp=datetime.now().isoformat(),
            snapshot_hash="",  # Will be calculated below
            python_packages=python_packages,
            system_executables=system_executables,
            configuration_files=configuration_files,
            dataset_files=dataset_files,
            git_info=git_info,
            environment_vars=environment_vars
        )
        
        # Calculate snapshot hash
        snapshot_dict = asdict(snapshot)
        snapshot_dict.pop("snapshot_hash")  # Remove hash field for calculation
        snapshot_str = json.dumps(snapshot_dict, sort_keys=True)
        snapshot.snapshot_hash = hashlib.sha256(snapshot_str.encode()).hexdigest()[:16]
        
        return snapshot
    
    def save_version_snapshot(self, snapshot: VersionSnapshot, name: str = None) -> Path:
        """Save version snapshot to file."""
        if name is None:
            name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{snapshot.snapshot_hash}"
        
        snapshot_file = self.versions_dir / f"{name}.json"
        
        with open(snapshot_file, 'w') as f:
            json.dump(asdict(snapshot), f, indent=2)
        
        print(f"💾 Version snapshot saved: {snapshot_file}")
        return snapshot_file
    
    def load_version_snapshot(self, snapshot_file: Path) -> VersionSnapshot:
        """Load version snapshot from file."""
        with open(snapshot_file, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to VersionInfo objects
        data["python_packages"] = [VersionInfo(**pkg) for pkg in data["python_packages"]]
        data["system_executables"] = [VersionInfo(**exe) for exe in data["system_executables"]]
        data["configuration_files"] = [VersionInfo(**cfg) for cfg in data["configuration_files"]]
        data["dataset_files"] = [VersionInfo(**ds) for ds in data["dataset_files"]]
        
        return VersionSnapshot(**data)
    
    def compare_snapshots(self, snapshot1: VersionSnapshot, snapshot2: VersionSnapshot) -> Dict[str, Any]:
        """Compare two version snapshots."""
        comparison = {
            "timestamp1": snapshot1.timestamp,
            "timestamp2": snapshot2.timestamp,
            "hash1": snapshot1.snapshot_hash,
            "hash2": snapshot2.snapshot_hash,
            "identical": snapshot1.snapshot_hash == snapshot2.snapshot_hash,
            "differences": {}
        }
        
        # Compare each category
        categories = [
            ("python_packages", "Python Packages"),
            ("system_executables", "System Executables"),
            ("configuration_files", "Configuration Files"),
            ("dataset_files", "Dataset Files")
        ]
        
        for attr_name, display_name in categories:
            items1 = {item.name: item for item in getattr(snapshot1, attr_name)}
            items2 = {item.name: item for item in getattr(snapshot2, attr_name)}
            
            category_diff = {
                "added": [],
                "removed": [],
                "changed": []
            }
            
            # Find added items
            for name in items2.keys() - items1.keys():
                category_diff["added"].append(name)
            
            # Find removed items
            for name in items1.keys() - items2.keys():
                category_diff["removed"].append(name)
            
            # Find changed items
            for name in items1.keys() & items2.keys():
                item1, item2 = items1[name], items2[name]
                if item1.version != item2.version or item1.hash != item2.hash:
                    category_diff["changed"].append({
                        "name": name,
                        "old_version": item1.version,
                        "new_version": item2.version,
                        "old_hash": item1.hash,
                        "new_hash": item2.hash
                    })
            
            comparison["differences"][attr_name] = category_diff
        
        # Compare git info
        if snapshot1.git_info != snapshot2.git_info:
            comparison["differences"]["git_info"] = {
                "old": snapshot1.git_info,
                "new": snapshot2.git_info
            }
        
        return comparison
    
    def print_version_summary(self, snapshot: VersionSnapshot):
        """Print a summary of the version snapshot."""
        print("\n" + "=" * 60)
        print("📋 Version Snapshot Summary")
        print("=" * 60)
        
        print(f"🕒 Timestamp: {snapshot.timestamp}")
        print(f"🔍 Hash: {snapshot.snapshot_hash}")
        
        # Git info
        if snapshot.git_info:
            git = snapshot.git_info
            if "commit_hash" in git:
                print(f"📝 Git Commit: {git.get('short_hash', git['commit_hash'][:8])}")
                print(f"🌿 Branch: {git.get('branch', 'unknown')}")
                if git.get('has_uncommitted_changes'):
                    print(f"⚠️  Uncommitted changes: {git.get('uncommitted_files', 0)} files")
        
        # Package counts
        print(f"\n📦 Components:")
        print(f"   Python Packages: {len(snapshot.python_packages)}")
        print(f"   System Executables: {len(snapshot.system_executables)}")
        print(f"   Configuration Files: {len(snapshot.configuration_files)}")
        print(f"   Dataset Files: {len(snapshot.dataset_files)}")
        
        # Key packages
        key_packages = ["lm_eval", "torch", "transformers", "docker"]
        print(f"\n🔑 Key Packages:")
        for pkg in snapshot.python_packages:
            if pkg.name in key_packages:
                print(f"   {pkg.name}: {pkg.version}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Version Tracking Tool")
    parser.add_argument("--create", "-c", action="store_true",
                       help="Create new version snapshot")
    parser.add_argument("--name", "-n", type=str,
                       help="Name for the snapshot")
    parser.add_argument("--compare", nargs=2, metavar=("SNAP1", "SNAP2"),
                       help="Compare two snapshots")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available snapshots")
    parser.add_argument("--show", "-s", type=str,
                       help="Show details of a specific snapshot")
    
    args = parser.parse_args()
    
    tracker = VersionTracker()
    
    if args.create:
        # Create new snapshot
        snapshot = tracker.create_version_snapshot()
        snapshot_file = tracker.save_version_snapshot(snapshot, args.name)
        tracker.print_version_summary(snapshot)
        
    elif args.compare:
        # Compare snapshots
        snap1_file = tracker.versions_dir / f"{args.compare[0]}.json"
        snap2_file = tracker.versions_dir / f"{args.compare[1]}.json"
        
        if not snap1_file.exists():
            print(f"❌ Snapshot not found: {snap1_file}")
            sys.exit(1)
        
        if not snap2_file.exists():
            print(f"❌ Snapshot not found: {snap2_file}")
            sys.exit(1)
        
        snapshot1 = tracker.load_version_snapshot(snap1_file)
        snapshot2 = tracker.load_version_snapshot(snap2_file)
        
        comparison = tracker.compare_snapshots(snapshot1, snapshot2)
        
        print("🔍 Snapshot Comparison")
        print("=" * 40)
        print(f"Snapshot 1: {comparison['timestamp1']} ({comparison['hash1']})")
        print(f"Snapshot 2: {comparison['timestamp2']} ({comparison['hash2']})")
        print(f"Identical: {'✅ Yes' if comparison['identical'] else '❌ No'}")
        
        if not comparison['identical']:
            for category, changes in comparison['differences'].items():
                if category == 'git_info':
                    continue
                
                total_changes = len(changes.get('added', [])) + len(changes.get('removed', [])) + len(changes.get('changed', []))
                if total_changes > 0:
                    print(f"\n{category.replace('_', ' ').title()}: {total_changes} changes")
                    
                    for item in changes.get('added', []):
                        print(f"  + {item}")
                    for item in changes.get('removed', []):
                        print(f"  - {item}")
                    for item in changes.get('changed', []):
                        print(f"  ~ {item['name']}: {item['old_version']} → {item['new_version']}")
    
    elif args.list:
        # List snapshots
        snapshots = list(tracker.versions_dir.glob("*.json"))
        
        if not snapshots:
            print("📁 No snapshots found")
        else:
            print(f"📁 Available Snapshots ({len(snapshots)}):")
            for snapshot_file in sorted(snapshots):
                name = snapshot_file.stem
                try:
                    snapshot = tracker.load_version_snapshot(snapshot_file)
                    print(f"   {name} - {snapshot.timestamp} ({snapshot.snapshot_hash})")
                except Exception as e:
                    print(f"   {name} - Error loading: {e}")
    
    elif args.show:
        # Show snapshot details
        snapshot_file = tracker.versions_dir / f"{args.show}.json"
        
        if not snapshot_file.exists():
            print(f"❌ Snapshot not found: {snapshot_file}")
            sys.exit(1)
        
        snapshot = tracker.load_version_snapshot(snapshot_file)
        tracker.print_version_summary(snapshot)
        
        # Show detailed information
        print(f"\n📦 Python Packages ({len(snapshot.python_packages)}):")
        for pkg in sorted(snapshot.python_packages, key=lambda x: x.name):
            print(f"   {pkg.name}: {pkg.version}")
        
        print(f"\n⚙️  System Executables ({len(snapshot.system_executables)}):")
        for exe in sorted(snapshot.system_executables, key=lambda x: x.name):
            print(f"   {exe.name}: {exe.version}")
    
    else:
        # Default: create snapshot and show summary
        snapshot = tracker.create_version_snapshot()
        tracker.print_version_summary(snapshot)
        
        # Ask if user wants to save
        try:
            save = input("\n💾 Save this snapshot? (y/N): ").lower().strip()
            if save in ['y', 'yes']:
                name = input("📝 Enter snapshot name (optional): ").strip()
                snapshot_file = tracker.save_version_snapshot(snapshot, name if name else None)
        except KeyboardInterrupt:
            print("\n👋 Snapshot not saved")

if __name__ == "__main__":
    main()