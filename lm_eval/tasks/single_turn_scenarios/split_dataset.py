#!/usr/bin/env python3
"""
Dataset Splitting and Reproducibility Tool

This script creates reproducible train/validation/test splits of the problems dataset
and manages baseline results for comparison and validation.
"""

import os
import sys
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    stratify_by: List[str] = None  # Fields to stratify by (e.g., ['scenario', 'difficulty'])
    min_per_category: int = 1  # Minimum samples per category in each split

@dataclass
class SplitMetadata:
    """Metadata for a dataset split."""
    timestamp: str
    config: SplitConfig
    total_problems: int
    train_count: int
    val_count: int
    test_count: int
    stratification_stats: Dict[str, Any]
    split_hash: str  # Hash of the split for reproducibility verification

class DatasetSplitter:
    """Dataset splitting and management class."""
    
    def __init__(self, problems_file: Path, output_dir: Path = None):
        self.problems_file = problems_file
        self.output_dir = output_dir or problems_file.parent / "splits"
        self.output_dir.mkdir(exist_ok=True)
        
    def load_problems(self) -> List[Dict[str, Any]]:
        """Load problems from JSONL file."""
        problems = []
        
        if not self.problems_file.exists():
            raise FileNotFoundError(f"Problems file not found: {self.problems_file}")
        
        with open(self.problems_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        problem = json.loads(line)
                        problems.append(problem)
                    except json.JSONDecodeError as e:
                        print(f"Warning: JSON decode error at line {line_num}: {e}")
        
        if not problems:
            raise ValueError("No valid problems found in dataset")
        
        return problems
    
    def create_stratified_split(self, problems: List[Dict[str, Any]], 
                              config: SplitConfig) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create stratified train/val/test splits."""
        if config.stratify_by:
            return self._stratified_split(problems, config)
        else:
            return self._random_split(problems, config)
    
    def _random_split(self, problems: List[Dict[str, Any]], 
                     config: SplitConfig) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create random train/val/test splits."""
        # Set seed for reproducibility
        random.seed(config.seed)
        
        # Shuffle problems
        shuffled_problems = problems.copy()
        random.shuffle(shuffled_problems)
        
        # Calculate split indices
        total = len(shuffled_problems)
        train_end = int(total * config.train_ratio)
        val_end = train_end + int(total * config.val_ratio)
        
        train_set = shuffled_problems[:train_end]
        val_set = shuffled_problems[train_end:val_end]
        test_set = shuffled_problems[val_end:]
        
        return train_set, val_set, test_set
    
    def _stratified_split(self, problems: List[Dict[str, Any]], 
                         config: SplitConfig) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create stratified train/val/test splits."""
        # Set seed for reproducibility
        random.seed(config.seed)
        
        # Group problems by stratification keys
        groups = {}
        for problem in problems:
            # Create stratification key
            key_parts = []
            for field in config.stratify_by:
                value = problem.get(field, "unknown")
                key_parts.append(f"{field}:{value}")
            key = "|".join(key_parts)
            
            if key not in groups:
                groups[key] = []
            groups[key].append(problem)
        
        # Split each group proportionally
        train_set, val_set, test_set = [], [], []
        
        for key, group_problems in groups.items():
            # Shuffle group
            random.shuffle(group_problems)
            
            group_size = len(group_problems)
            
            # Ensure minimum samples per split if possible
            if group_size >= 3 * config.min_per_category:
                # Calculate split sizes
                train_size = max(config.min_per_category, 
                               int(group_size * config.train_ratio))
                val_size = max(config.min_per_category,
                             int(group_size * config.val_ratio))
                test_size = group_size - train_size - val_size
                
                # Adjust if test_size is too small
                if test_size < config.min_per_category:
                    test_size = config.min_per_category
                    val_size = max(config.min_per_category, 
                                 group_size - train_size - test_size)
                    train_size = group_size - val_size - test_size
            else:
                # For small groups, distribute as evenly as possible
                if group_size >= 3:
                    train_size = max(1, group_size // 2)
                    val_size = max(1, (group_size - train_size) // 2)
                    test_size = group_size - train_size - val_size
                elif group_size == 2:
                    train_size, val_size, test_size = 1, 1, 0
                else:
                    train_size, val_size, test_size = 1, 0, 0
            
            # Split the group
            train_set.extend(group_problems[:train_size])
            val_set.extend(group_problems[train_size:train_size + val_size])
            test_set.extend(group_problems[train_size + val_size:])
        
        # Final shuffle of each set
        random.shuffle(train_set)
        random.shuffle(val_set)
        random.shuffle(test_set)
        
        return train_set, val_set, test_set
    
    def calculate_stratification_stats(self, train_set: List[Dict], 
                                     val_set: List[Dict], 
                                     test_set: List[Dict],
                                     stratify_by: List[str]) -> Dict[str, Any]:
        """Calculate stratification statistics."""
        stats = {}
        
        if not stratify_by:
            return stats
        
        for field in stratify_by:
            field_stats = {
                "train": {},
                "val": {},
                "test": {},
                "total": {}
            }
            
            # Count occurrences in each set
            for split_name, split_data in [("train", train_set), ("val", val_set), ("test", test_set)]:
                for problem in split_data:
                    value = problem.get(field, "unknown")
                    field_stats[split_name][value] = field_stats[split_name].get(value, 0) + 1
            
            # Calculate totals
            all_values = set()
            for split_data in [field_stats["train"], field_stats["val"], field_stats["test"]]:
                all_values.update(split_data.keys())
            
            for value in all_values:
                train_count = field_stats["train"].get(value, 0)
                val_count = field_stats["val"].get(value, 0)
                test_count = field_stats["test"].get(value, 0)
                total_count = train_count + val_count + test_count
                
                field_stats["total"][value] = {
                    "train": train_count,
                    "val": val_count,
                    "test": test_count,
                    "total": total_count,
                    "train_ratio": train_count / total_count if total_count > 0 else 0,
                    "val_ratio": val_count / total_count if total_count > 0 else 0,
                    "test_ratio": test_count / total_count if total_count > 0 else 0
                }
            
            stats[field] = field_stats
        
        return stats
    
    def calculate_split_hash(self, train_set: List[Dict], 
                           val_set: List[Dict], 
                           test_set: List[Dict]) -> str:
        """Calculate hash of the split for reproducibility verification."""
        # Create a deterministic representation of the split
        split_repr = {
            "train_ids": sorted([p.get("id", "") for p in train_set]),
            "val_ids": sorted([p.get("id", "") for p in val_set]),
            "test_ids": sorted([p.get("id", "") for p in test_set])
        }
        
        # Calculate hash
        split_str = json.dumps(split_repr, sort_keys=True)
        return hashlib.sha256(split_str.encode()).hexdigest()[:16]
    
    def save_splits(self, train_set: List[Dict], val_set: List[Dict], 
                   test_set: List[Dict], config: SplitConfig) -> SplitMetadata:
        """Save splits to files and return metadata."""
        # Calculate statistics
        stratification_stats = self.calculate_stratification_stats(
            train_set, val_set, test_set, config.stratify_by or []
        )
        
        # Calculate split hash
        split_hash = self.calculate_split_hash(train_set, val_set, test_set)
        
        # Create metadata
        metadata = SplitMetadata(
            timestamp=datetime.now().isoformat(),
            config=config,
            total_problems=len(train_set) + len(val_set) + len(test_set),
            train_count=len(train_set),
            val_count=len(val_set),
            test_count=len(test_set),
            stratification_stats=stratification_stats,
            split_hash=split_hash
        )
        
        # Create split directory
        split_dir = self.output_dir / f"split_{config.seed}_{split_hash}"
        split_dir.mkdir(exist_ok=True)
        
        # Save splits as JSONL files
        self._save_jsonl(train_set, split_dir / "train.jsonl")
        self._save_jsonl(val_set, split_dir / "val.jsonl")
        self._save_jsonl(test_set, split_dir / "test.jsonl")
        
        # Save metadata
        with open(split_dir / "metadata.json", 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Save split IDs for easy reference
        split_ids = {
            "train": [p.get("id") for p in train_set],
            "val": [p.get("id") for p in val_set],
            "test": [p.get("id") for p in test_set]
        }
        with open(split_dir / "split_ids.json", 'w') as f:
            json.dump(split_ids, f, indent=2)
        
        print(f"✅ Splits saved to: {split_dir}")
        print(f"   📊 Train: {len(train_set)} problems")
        print(f"   📊 Val: {len(val_set)} problems")
        print(f"   📊 Test: {len(test_set)} problems")
        print(f"   🔍 Split hash: {split_hash}")
        
        return metadata
    
    def _save_jsonl(self, problems: List[Dict], file_path: Path):
        """Save problems to JSONL file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for problem in problems:
                f.write(json.dumps(problem) + '\n')
    
    def load_existing_split(self, split_hash: str) -> Optional[Tuple[List[Dict], List[Dict], List[Dict], SplitMetadata]]:
        """Load an existing split by hash."""
        # Find split directory
        split_dirs = list(self.output_dir.glob(f"split_*_{split_hash}"))
        
        if not split_dirs:
            return None
        
        split_dir = split_dirs[0]
        
        # Load metadata
        metadata_file = split_dir / "metadata.json"
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
        
        # Convert back to dataclass
        config_dict = metadata_dict.pop("config")
        config = SplitConfig(**config_dict)
        metadata = SplitMetadata(config=config, **metadata_dict)
        
        # Load splits
        train_set = self._load_jsonl(split_dir / "train.jsonl")
        val_set = self._load_jsonl(split_dir / "val.jsonl")
        test_set = self._load_jsonl(split_dir / "test.jsonl")
        
        return train_set, val_set, test_set, metadata
    
    def _load_jsonl(self, file_path: Path) -> List[Dict]:
        """Load problems from JSONL file."""
        problems = []
        
        if not file_path.exists():
            return problems
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    problems.append(json.loads(line))
        
        return problems
    
    def verify_split_reproducibility(self, config: SplitConfig) -> bool:
        """Verify that a split can be reproduced with the same config."""
        print(f"🔍 Verifying reproducibility for seed {config.seed}...")
        
        # Load problems
        problems = self.load_problems()
        
        # Create split twice
        split1 = self.create_stratified_split(problems, config)
        split2 = self.create_stratified_split(problems, config)
        
        # Calculate hashes
        hash1 = self.calculate_split_hash(*split1)
        hash2 = self.calculate_split_hash(*split2)
        
        if hash1 == hash2:
            print(f"✅ Split is reproducible (hash: {hash1})")
            return True
        else:
            print(f"❌ Split is not reproducible (hash1: {hash1}, hash2: {hash2})")
            return False
    
    def create_baseline_results(self, split_metadata: SplitMetadata) -> Dict[str, Any]:
        """Create baseline results structure for comparison."""
        baseline = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "split_hash": split_metadata.split_hash,
                "split_config": asdict(split_metadata.config),
                "problem_counts": {
                    "train": split_metadata.train_count,
                    "val": split_metadata.val_count,
                    "test": split_metadata.test_count,
                    "total": split_metadata.total_problems
                }
            },
            "baseline_metrics": {
                "random_baseline": {
                    "description": "Random prediction baseline",
                    "exact_match": 0.0,
                    "pass_at_1": 0.0,
                    "syntax_valid": 0.0
                },
                "simple_heuristic": {
                    "description": "Simple heuristic baseline (e.g., return empty function)",
                    "exact_match": 0.0,
                    "pass_at_1": 0.1,  # Might pass some trivial tests
                    "syntax_valid": 1.0
                }
            },
            "expected_ranges": {
                "exact_match": {"min": 0.0, "max": 1.0, "good_threshold": 0.3},
                "pass_at_1": {"min": 0.0, "max": 1.0, "good_threshold": 0.5},
                "syntax_valid": {"min": 0.0, "max": 1.0, "good_threshold": 0.8},
                "codebleu": {"min": 0.0, "max": 1.0, "good_threshold": 0.4}
            },
            "validation_checks": {
                "min_problems_per_scenario": 5,
                "min_problems_per_difficulty": 10,
                "max_evaluation_time_minutes": 60,
                "required_metrics": ["exact_match", "pass_at_1", "syntax_valid", "codebleu"]
            }
        }
        
        return baseline
    
    def print_split_summary(self, metadata: SplitMetadata):
        """Print a summary of the split."""
        print("\n" + "=" * 60)
        print("📊 Dataset Split Summary")
        print("=" * 60)
        
        print(f"🕒 Created: {metadata.timestamp}")
        print(f"🎲 Seed: {metadata.config.seed}")
        print(f"🔍 Hash: {metadata.split_hash}")
        print(f"📈 Ratios: {metadata.config.train_ratio:.1%} / {metadata.config.val_ratio:.1%} / {metadata.config.test_ratio:.1%}")
        
        print(f"\n📊 Problem Counts:")
        print(f"   Train: {metadata.train_count}")
        print(f"   Val: {metadata.val_count}")
        print(f"   Test: {metadata.test_count}")
        print(f"   Total: {metadata.total_problems}")
        
        if metadata.config.stratify_by and metadata.stratification_stats:
            print(f"\n🎯 Stratification by: {', '.join(metadata.config.stratify_by)}")
            
            for field, stats in metadata.stratification_stats.items():
                print(f"\n   {field.upper()}:")
                for value, counts in stats["total"].items():
                    print(f"     {value}: {counts['train']}/{counts['val']}/{counts['test']} "
                          f"({counts['train_ratio']:.1%}/{counts['val_ratio']:.1%}/{counts['test_ratio']:.1%})")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Dataset Splitting and Reproducibility Tool")
    parser.add_argument("--problems", "-p", type=Path, 
                       help="Path to problems.jsonl file")
    parser.add_argument("--output", "-o", type=Path,
                       help="Output directory for splits")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                       help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                       help="Test set ratio")
    parser.add_argument("--stratify", nargs="+", 
                       help="Fields to stratify by (e.g., scenario difficulty)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify reproducibility of the split")
    parser.add_argument("--create-baseline", action="store_true",
                       help="Create baseline results file")
    parser.add_argument("--load-split", type=str,
                       help="Load existing split by hash")
    
    args = parser.parse_args()
    
    # Default problems file location
    if not args.problems:
        current_dir = Path(__file__).parent
        args.problems = current_dir / "problems.jsonl"
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"❌ Error: Ratios must sum to 1.0 (current sum: {total_ratio})")
        sys.exit(1)
    
    # Create splitter
    splitter = DatasetSplitter(args.problems, args.output)
    
    # Create config
    config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_by=args.stratify,
        min_per_category=1
    )
    
    try:
        if args.load_split:
            # Load existing split
            result = splitter.load_existing_split(args.load_split)
            if result:
                train_set, val_set, test_set, metadata = result
                print(f"✅ Loaded existing split: {args.load_split}")
                splitter.print_split_summary(metadata)
            else:
                print(f"❌ Split not found: {args.load_split}")
                sys.exit(1)
        
        elif args.verify:
            # Verify reproducibility
            if splitter.verify_split_reproducibility(config):
                print("✅ Split reproducibility verified")
            else:
                print("❌ Split reproducibility failed")
                sys.exit(1)
        
        else:
            # Create new split
            print("🚀 Creating dataset split...")
            print(f"📁 Problems file: {args.problems}")
            print(f"📁 Output directory: {splitter.output_dir}")
            
            # Load problems
            problems = splitter.load_problems()
            print(f"📊 Loaded {len(problems)} problems")
            
            # Create split
            train_set, val_set, test_set = splitter.create_stratified_split(problems, config)
            
            # Save splits
            metadata = splitter.save_splits(train_set, val_set, test_set, config)
            
            # Print summary
            splitter.print_split_summary(metadata)
            
            # Create baseline if requested
            if args.create_baseline:
                baseline = splitter.create_baseline_results(metadata)
                baseline_file = splitter.output_dir / f"split_{config.seed}_{metadata.split_hash}" / "baseline.json"
                
                with open(baseline_file, 'w') as f:
                    json.dump(baseline, f, indent=2)
                
                print(f"\n📄 Baseline results created: {baseline_file}")
        
        print("\n✅ Dataset splitting completed successfully")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()