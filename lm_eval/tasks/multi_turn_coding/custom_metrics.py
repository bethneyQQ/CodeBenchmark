import os
import json
import ast
import subprocess
import sys
import tempfile
import shutil
from typing import List, Dict, Any
import re
from pathlib import Path


def _get_output_base_path() -> str:
    """Get the correct output base path depending on working directory."""
    # Prefer the task-specific output directory over the root one
    possible_paths = [
        "./lm_eval/tasks/multi_turn_coding/output",  # From root directory
        "lm_eval/tasks/multi_turn_coding/output",    # Alternative from root
        "./output",                                   # From task directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return "./output"  # fallback


def _get_available_problem_dirs() -> List[str]:
    """Get list of available problem directories that have evaluation artifacts."""
    output_path = _get_output_base_path()
    
    if not os.path.exists(output_path):
        return []
    
    # Get directories that have at least one of the expected files
    valid_dirs = []
    all_dirs = os.listdir(output_path)
    
    for d in all_dirs:
        dir_path = os.path.join(output_path, d)
        if os.path.isdir(dir_path):
            # Check if this directory has any evaluation artifacts
            prd_exists = os.path.exists(os.path.join(dir_path, "prd.md"))
            design_exists = os.path.exists(os.path.join(dir_path, "design.md"))
            src_exists = os.path.exists(os.path.join(dir_path, "src"))
            
            has_artifacts = prd_exists or design_exists or src_exists
            
            # Debug output (commented out for production)
            # if d == "easy_001":
            #     print(f"DEBUG easy_001: prd={prd_exists}, design={design_exists}, quality={quality_exists}, src={src_exists}")
            #     print(f"DEBUG easy_001: has_artifacts={has_artifacts}")
            #     print(f"DEBUG easy_001: dir_path={dir_path}")
            
            if has_artifacts:
                valid_dirs.append(d)
    
    # Sort by problem ID (easy_001, easy_002, ..., simple_001, etc.)
    def sort_key(dirname):
        if '_' in dirname:
            complexity, num = dirname.split('_', 1)
            complexity_order = {'easy': 0, 'simple': 1, 'medium': 2, 'complex': 3}
            return (complexity_order.get(complexity, 999), int(num))
        return (999, 0)
    
    valid_dirs.sort(key=sort_key)
    # print(f"DEBUG: output_path={output_path}, all_dirs count={len(all_dirs)}, valid_dirs count={len(valid_dirs)}")
    return valid_dirs


def file_existence_check(references: List[str], predictions: List[List[str]]) -> Dict[str, float]:
    """Check if all required files were created."""
    total_score = 0.0
    total_problems = len(predictions)
    
    # Get all available problem directories
    output_dirs = _get_available_problem_dirs()
    
    for i, pred_list in enumerate(predictions):
        if not pred_list:
            continue
            
        # Get the problem_id for this index
        if i < len(output_dirs):
            problem_id = output_dirs[i]
        else:
            continue
        
        base_path = _get_output_base_path()
        required_files = [
            f"{base_path}/{problem_id}/prd.md",
            f"{base_path}/{problem_id}/design.md", 
            f"{base_path}/{problem_id}/src/"
        ]
        
        files_exist = 0
        for file_path in required_files:
            if os.path.exists(file_path):
                files_exist += 1
        
        score = files_exist / len(required_files)
        total_score += score
    
    return {'file_existence_check': total_score / total_problems if total_problems > 0 else 0.0}


def prd_quality_from_file(references: List[str], predictions: List[List[str]]) -> Dict[str, float]:
    """Evaluate PRD quality from saved files."""
    total_score = 0.0
    total_problems = len(predictions)
    
    # Get all available problem directories
    output_dirs = _get_available_problem_dirs()
    
    for i, pred_list in enumerate(predictions):
        if not pred_list:
            continue
            
        # Get the problem_id for this index
        if i < len(output_dirs):
            problem_id = output_dirs[i]
        else:
            continue
            
        base_path = _get_output_base_path()
        prd_path = f"{base_path}/{problem_id}/prd.md"
        
        if not os.path.exists(prd_path):
            continue
            
        try:
            with open(prd_path, 'r', encoding='utf-8') as f:
                prd_content = f.read()
            
            score = _evaluate_prd_content(prd_content)
            total_score += score
            
        except Exception as e:
            print(f"Error reading PRD file {prd_path}: {e}")
            continue
    
    return {'prd_quality_from_file': total_score / total_problems if total_problems > 0 else 0.0}


def _evaluate_prd_content(content: str) -> float:
    """Evaluate PRD content quality with more discriminative scoring."""
    score = 0.0
    content_lower = content.lower()
    
    # Basic sections (0.1 each - 0.5 total)
    basic_sections = [
        r"problem\s+statement|overview|summary",
        r"user\s+stor(y|ies)|requirements", 
        r"acceptance\s+criteria|success\s+criteria",
        r"functional\s+requirements|features",
        r"non-functional\s+requirements|constraints"
    ]
    
    for section_pattern in basic_sections:
        if re.search(section_pattern, content, re.IGNORECASE):
            score += 0.1
    
    # Advanced quality indicators (0.1 each - 0.5 total)
    quality_indicators = [
        # Specific metrics and KPIs
        (r"\d+%|\d+\s*(users?|requests?|ms|seconds?|mb|gb)", 0.1),
        # Security considerations
        (r"security|authentication|authorization|encryption|compliance", 0.1),
        # Performance requirements
        (r"performance|latency|throughput|scalability|load", 0.1),
        # Integration requirements
        (r"integration|api|third[- ]party|external|legacy", 0.1),
        # Business value/ROI
        (r"roi|revenue|cost|business\s+value|market|competitive", 0.1)
    ]
    
    for pattern, points in quality_indicators:
        if re.search(pattern, content, re.IGNORECASE):
            score += points
    
    # Penalty for too short content (less than 200 words)
    word_count = len(content.split())
    if word_count < 200:
        score *= (word_count / 200)
    
    return min(score, 1.0)


def design_coherence_from_file(references: List[str], predictions: List[List[str]]) -> Dict[str, float]:
    """Evaluate design document coherence from saved files."""
    total_score = 0.0
    total_problems = len(predictions)
    
    # Get all available problem directories
    output_dirs = _get_available_problem_dirs()
    
    for i, pred_list in enumerate(predictions):
        if not pred_list:
            continue
            
        # Get the problem_id for this index
        if i < len(output_dirs):
            problem_id = output_dirs[i]
        else:
            continue
            
        base_path = _get_output_base_path()
        design_path = f"{base_path}/{problem_id}/design.md"
        
        if not os.path.exists(design_path):
            continue
            
        try:
            with open(design_path, 'r', encoding='utf-8') as f:
                design_content = f.read()
            
            score = _evaluate_design_content(design_content)
            total_score += score
            
        except Exception as e:
            print(f"Error reading design file {design_path}: {e}")
            continue
    
    return {'design_coherence_from_file': total_score / total_problems if total_problems > 0 else 0.0}


def _evaluate_design_content(content: str) -> float:
    """Evaluate design document quality with more discriminative scoring."""
    score = 0.0
    content_lower = content.lower()
    
    # Basic design elements (0.08 each - 0.32 total)
    basic_elements = [
        r"architecture|system\s+design|components",
        r"api|endpoint|interface|schema", 
        r"database|data\s+model|storage",
        r"security|authentication|authorization"
    ]
    
    for element_pattern in basic_elements:
        if re.search(element_pattern, content, re.IGNORECASE):
            score += 0.08
    
    # Advanced design considerations (0.08 each - 0.64 total)
    advanced_elements = [
        # Scalability and performance
        (r"scalability|horizontal\s+scaling|load\s+balancing|caching", 0.08),
        # Reliability patterns
        (r"circuit\s+breaker|retry|timeout|bulkhead|failover", 0.08),
        # Data consistency
        (r"consistency|transaction|acid|eventual\s+consistency|saga", 0.08),
        # Monitoring and observability
        (r"monitoring|logging|metrics|tracing|observability", 0.08),
        # Deployment and infrastructure
        (r"deployment|infrastructure|kubernetes|docker|ci/cd", 0.08),
        # Design patterns
        (r"pattern|microservices|event\s+driven|cqrs|ddd", 0.08),
        # Technology choices with rationale
        (r"because|rationale|trade[- ]off|pros\s+and\s+cons|decision", 0.08),
        # Specific technologies mentioned
        (r"spring|react|postgresql|redis|kafka|elasticsearch", 0.08)
    ]
    
    for pattern, points in advanced_elements:
        if re.search(pattern, content, re.IGNORECASE):
            score += points
    
    # Bonus for diagrams or structured content (0.04)
    if re.search(r"diagram|flowchart|sequence|component|class", content, re.IGNORECASE):
        score += 0.04
    
    # Penalty for too short content (less than 300 words)
    word_count = len(content.split())
    if word_count < 300:
        score *= (word_count / 300)
    
    return min(score, 1.0)


def code_execution_test(references: List[str], predictions: List[List[str]]) -> Dict[str, float]:
    """Test if generated code can be executed without errors."""
    total_score = 0.0
    total_problems = len(predictions)
    
    # Get all available problem directories
    output_dirs = _get_available_problem_dirs()
    
    for i, pred_list in enumerate(predictions):
        if not pred_list:
            continue
            
        # Get the problem_id for this index
        if i < len(output_dirs):
            problem_id = output_dirs[i]
        else:
            continue
            
        base_path = _get_output_base_path()
        src_path = f"{base_path}/{problem_id}/src"
        
        if not os.path.exists(src_path):
            continue
            
        score = _test_code_execution(src_path)
        total_score += score
    
    return {'code_execution_test': total_score / total_problems if total_problems > 0 else 0.0}


def _test_code_execution(src_path: str) -> float:
    """Test code execution with more comprehensive quality assessment."""
    try:
        # Find Python files
        python_files = []
        for root, dirs, files in os.walk(src_path):
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    python_files.append(os.path.join(root, file))
        
        if not python_files:
            return 0.0
        
        total_score = 0.0
        
        # 1. Syntax validation (0.2)
        syntax_score = 0.0
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                ast.parse(code)
                syntax_score += 1
            except SyntaxError:
                continue
            except Exception:
                continue
        
        syntax_score = (syntax_score / len(python_files)) * 0.2 if python_files else 0.0
        total_score += syntax_score
        
        # 2. Code quality indicators (0.3)
        quality_score = _assess_code_quality(python_files) * 0.3
        total_score += quality_score
        
        # 3. Test coverage and execution (0.3)
        test_score = _run_tests(src_path) * 0.3
        total_score += test_score
        
        # 4. Project structure quality (0.2)
        structure_score = _assess_project_structure(src_path) * 0.2
        total_score += structure_score
        
        return min(total_score, 1.0)
        
    except Exception as e:
        print(f"Error testing code execution: {e}")
        return 0.0


def _assess_code_quality(python_files: List[str]) -> float:
    """Assess code quality based on various indicators."""
    if not python_files:
        return 0.0
    
    total_quality = 0.0
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                code = f.read()
            
            file_quality = 0.0
            
            # Check for type hints
            if re.search(r':\s*\w+\s*=|def\s+\w+\([^)]*:\s*\w+', code):
                file_quality += 0.2
            
            # Check for docstrings
            if re.search(r'""".*?"""', code, re.DOTALL) or re.search(r"'''.*?'''", code, re.DOTALL):
                file_quality += 0.2
            
            # Check for error handling
            if re.search(r'try:|except|raise|finally:', code):
                file_quality += 0.2
            
            # Check for logging
            if re.search(r'import\s+logging|logger\.|log\.|print\(', code):
                file_quality += 0.1
            
            # Check for configuration/constants
            if re.search(r'[A-Z_]{3,}\s*=|config|settings', code, re.IGNORECASE):
                file_quality += 0.1
            
            # Check for proper imports organization
            lines = code.split('\n')
            import_section = []
            for line in lines[:20]:  # Check first 20 lines
                if line.strip().startswith(('import ', 'from ')):
                    import_section.append(line)
            
            if len(import_section) > 1:
                # Check if imports are organized (stdlib, third-party, local)
                file_quality += 0.1
            
            # Penalty for very long functions (>50 lines)
            functions = re.findall(r'def\s+\w+.*?(?=\ndef|\nclass|\Z)', code, re.DOTALL)
            long_functions = sum(1 for func in functions if len(func.split('\n')) > 50)
            if long_functions > 0:
                file_quality *= 0.8
            
            total_quality += file_quality
            
        except Exception:
            continue
    
    return total_quality / len(python_files) if python_files else 0.0


def _assess_project_structure(src_path: str) -> float:
    """Assess project structure quality."""
    score = 0.0
    
    # Check for proper Python package structure
    if os.path.exists(os.path.join(src_path, '__init__.py')):
        score += 0.2
    
    # Check for configuration files
    config_files = ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile']
    for config_file in config_files:
        if os.path.exists(os.path.join(src_path, config_file)):
            score += 0.1
            break
    
    # Check for test directory or test files
    has_tests = False
    for root, dirs, files in os.walk(src_path):
        if 'tests' in dirs or 'test' in dirs:
            has_tests = True
            break
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                has_tests = True
                break
    
    if has_tests:
        score += 0.2
    
    # Check for documentation
    doc_files = ['README.md', 'README.rst', 'docs']
    for doc_file in doc_files:
        if os.path.exists(os.path.join(src_path, doc_file)):
            score += 0.1
            break
    
    # Check for proper separation of concerns (multiple modules)
    py_files = []
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                py_files.append(file)
    
    if len(py_files) > 1:
        score += 0.2
    
    # Check for main entry point
    if 'main.py' in py_files or '__main__.py' in py_files:
        score += 0.1
    
    # Check for configuration/settings module
    config_modules = ['config.py', 'settings.py', 'constants.py']
    if any(module in py_files for module in config_modules):
        score += 0.1
    
    return min(score, 1.0)


def _run_tests(src_path: str) -> float:
    """Run tests if they exist."""
    try:
        # Look for test files
        test_files = []
        for root, dirs, files in os.walk(src_path):
            for file in files:
                if file.startswith('test_') and file.endswith('.py'):
                    test_files.append(os.path.join(root, file))
        
        if not test_files:
            return 0.5  # No tests found, give partial credit
        
        # Try to run pytest
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', src_path, '-v'],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=src_path
            )
            
            if result.returncode == 0:
                return 1.0  # All tests passed
            else:
                return 0.3  # Tests exist but failed
                
        except subprocess.TimeoutExpired:
            return 0.2  # Tests timed out
        except Exception:
            return 0.3  # Tests exist but couldn't run
            
    except Exception:
        return 0.0


def project_structure_validation(references: List[str], predictions: List[List[str]]) -> Dict[str, float]:
    """Validate Python project structure."""
    total_score = 0.0
    total_problems = len(predictions)
    
    # Get all available problem directories
    output_dirs = _get_available_problem_dirs()
    
    for i, pred_list in enumerate(predictions):
        if not pred_list:
            continue
            
        # Get the problem_id for this index
        if i < len(output_dirs):
            problem_id = output_dirs[i]
        else:
            continue
            
        base_path = _get_output_base_path()
        src_path = f"{base_path}/{problem_id}/src"
        
        if not os.path.exists(src_path):
            continue
            
        score = _validate_project_structure(src_path)
        total_score += score
    
    return {'project_structure_validation': total_score / total_problems if total_problems > 0 else 0.0}


def _validate_project_structure(src_path: str) -> float:
    """Validate project structure with context-aware scoring."""
    score = 0.0
    
    # Basic structure requirements (0.4 total)
    basic_structure = [
        ('requirements.txt', 0.15),
        ('__init__.py', 0.05),
        ('main.py', 0.1),
        ('.py files', 0.1)  # At least one Python file
    ]
    
    for check_item, points in basic_structure:
        if check_item == '.py files':
            py_files = [f for f in os.listdir(src_path) if f.endswith('.py')]
            if py_files:
                score += points
        else:
            file_path = os.path.join(src_path, check_item)
            if os.path.exists(file_path):
                score += points
    
    # Advanced structure indicators (0.6 total)
    advanced_checks = [
        # Testing structure
        ('tests/', 0.15, lambda: _check_test_structure(src_path)),
        # Configuration files
        ('config/', 0.1, lambda: _check_config_structure(src_path)),
        # Documentation
        ('docs/', 0.1, lambda: _check_documentation_structure(src_path)),
        # Code organization
        ('modules/', 0.1, lambda: _check_module_organization(src_path)),
        # DevOps files
        ('devops/', 0.1, lambda: _check_devops_files(src_path)),
        # Quality assurance
        ('qa/', 0.05, lambda: _check_qa_files(src_path))
    ]
    
    for check_name, points, check_func in advanced_checks:
        try:
            if check_func():
                score += points
        except Exception:
            continue
    
    return min(score, 1.0)


def _check_test_structure(src_path: str) -> bool:
    """Check for comprehensive test structure."""
    # Check for test directory or test files
    test_dir = os.path.join(src_path, 'tests')
    if os.path.exists(test_dir):
        return True
    
    # Check for test files in main directory
    test_files = [f for f in os.listdir(src_path) if f.startswith('test_') and f.endswith('.py')]
    return len(test_files) > 0


def _check_config_structure(src_path: str) -> bool:
    """Check for configuration files and structure."""
    config_indicators = [
        'config.py', 'settings.py', 'constants.py', 'config.yml', 
        'config.yaml', 'config.json', '.env.example'
    ]
    
    for indicator in config_indicators:
        if os.path.exists(os.path.join(src_path, indicator)):
            return True
    
    # Check for config directory
    config_dir = os.path.join(src_path, 'config')
    return os.path.exists(config_dir)


def _check_documentation_structure(src_path: str) -> bool:
    """Check for documentation files."""
    doc_indicators = ['README.md', 'README.rst', 'API.md', 'CHANGELOG.md']
    
    for indicator in doc_indicators:
        if os.path.exists(os.path.join(src_path, indicator)):
            return True
    
    # Check for docs directory
    docs_dir = os.path.join(src_path, 'docs')
    return os.path.exists(docs_dir)


def _check_module_organization(src_path: str) -> bool:
    """Check for proper module organization."""
    # Count Python files and directories
    py_files = []
    py_dirs = []
    
    for item in os.listdir(src_path):
        item_path = os.path.join(src_path, item)
        if os.path.isfile(item_path) and item.endswith('.py'):
            py_files.append(item)
        elif os.path.isdir(item_path) and not item.startswith('.'):
            # Check if directory contains Python files
            for subitem in os.listdir(item_path):
                if subitem.endswith('.py'):
                    py_dirs.append(item)
                    break
    
    # Good organization: multiple modules or organized directories
    return len(py_files) > 2 or len(py_dirs) > 0


def _check_devops_files(src_path: str) -> bool:
    """Check for DevOps and deployment files."""
    devops_indicators = [
        'Dockerfile', 'docker-compose.yml', 'docker-compose.yaml',
        '.github/', 'Jenkinsfile', '.gitlab-ci.yml', 'azure-pipelines.yml',
        'kubernetes/', 'k8s/', 'helm/', 'terraform/'
    ]
    
    for indicator in devops_indicators:
        if os.path.exists(os.path.join(src_path, indicator)):
            return True
    
    return False


def _check_qa_files(src_path: str) -> bool:
    """Check for quality assurance files."""
    qa_indicators = [
        '.pylintrc', 'pyproject.toml', 'setup.cfg', 'tox.ini',
        '.pre-commit-config.yaml', '.flake8', 'mypy.ini'
    ]
    
    for indicator in qa_indicators:
        if os.path.exists(os.path.join(src_path, indicator)):
            return True
    
    return False


def integration_test(references: List[str], predictions: List[List[str]]) -> Dict[str, float]:
    """Test integration between PRD, Design, and Code."""
    total_score = 0.0
    total_problems = len(predictions)
    
    # Get all available problem directories
    output_dirs = _get_available_problem_dirs()
    
    for i, pred_list in enumerate(predictions):
        if not pred_list:
            continue
            
        # Get the problem_id for this index
        if i < len(output_dirs):
            problem_id = output_dirs[i]
        else:
            continue
        
        # Check if all artifacts exist
        base_path = _get_output_base_path()
        prd_path = f"{base_path}/{problem_id}/prd.md"
        design_path = f"{base_path}/{problem_id}/design.md"
        src_path = f"{base_path}/{problem_id}/src"
        
        if not all(os.path.exists(p) for p in [prd_path, design_path, src_path]):
            continue
            
        score = _test_integration(prd_path, design_path, src_path)
        total_score += score
    
    return {'integration_test': total_score / total_problems if total_problems > 0 else 0.0}


def _test_integration(prd_path: str, design_path: str, src_path: str) -> float:
    """Test consistency between PRD, design, and implementation with context awareness."""
    try:
        # Read all documents
        with open(prd_path, 'r', encoding='utf-8') as f:
            prd_content = f.read().lower()
        
        with open(design_path, 'r', encoding='utf-8') as f:
            design_content = f.read().lower()
        
        # Get code content
        code_content = ""
        config_content = ""
        for root, dirs, files in os.walk(src_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        if file.endswith('.py'):
                            code_content += content + "\n"
                        elif file.endswith(('.yml', '.yaml', '.json', '.txt', '.md')):
                            config_content += content + "\n"
                except Exception:
                    continue
        
        score = 0.0
        
        # 1. Technical consistency (0.25)
        # Check if technical decisions in design are reflected in code
        tech_mappings = [
            (['flask', 'fastapi', 'django'], ['from flask', 'import flask', 'fastapi', 'django']),
            (['postgresql', 'mongodb', 'redis'], ['psycopg', 'pymongo', 'redis']),
            (['jwt', 'oauth'], ['jwt', 'oauth', 'token']),
            (['async', 'asynchronous'], ['async def', 'await', 'asyncio']),
            (['microservice', 'api'], ['router', 'endpoint', 'api'])
        ]
        
        tech_score = 0.0
        for design_terms, code_terms in tech_mappings:
            design_has = any(term in design_content for term in design_terms)
            code_has = any(term in code_content for term in code_terms)
            if design_has and code_has:
                tech_score += 0.05
        
        score += min(tech_score, 0.25)
        
        # 2. Architecture alignment (0.25)
        # Check if architectural patterns mentioned in design are implemented
        arch_patterns = [
            ('mvc', ['model', 'view', 'controller']),
            ('repository', ['repository', 'dao']),
            ('factory', ['factory', 'create']),
            ('singleton', ['singleton', 'instance']),
            ('observer', ['observer', 'notify', 'event'])
        ]
        
        arch_score = 0.0
        for pattern, implementations in arch_patterns:
            if pattern in design_content:
                if any(impl in code_content for impl in implementations):
                    arch_score += 0.05
        
        score += min(arch_score, 0.25)
        
        # 3. Feature completeness (0.25)
        # Check if features mentioned in PRD are addressed in design and code
        feature_keywords = _extract_keywords(prd_content)
        design_keywords = _extract_keywords(design_content)
        code_keywords = _extract_keywords(code_content)
        
        if feature_keywords:
            # Features mentioned in PRD should appear in design
            prd_design_overlap = len(feature_keywords.intersection(design_keywords)) / len(feature_keywords)
            # Features should also be implemented in code
            prd_code_overlap = len(feature_keywords.intersection(code_keywords)) / len(feature_keywords)
            
            feature_score = (prd_design_overlap + prd_code_overlap) / 2
            score += feature_score * 0.25
        
        # 4. Quality requirements traceability (0.25)
        # Check if quality requirements are addressed throughout
        quality_terms = [
            'performance', 'security', 'scalability', 'reliability',
            'maintainability', 'testability', 'usability', 'accessibility'
        ]
        
        quality_score = 0.0
        for term in quality_terms:
            in_prd = term in prd_content
            in_design = term in design_content
            in_code = term in code_content or term in config_content
            
            if in_prd and (in_design or in_code):
                quality_score += 1.0 / len(quality_terms)
        
        score += quality_score * 0.25
        
        return min(score, 1.0)
        
    except Exception as e:
        print(f"Error in integration test: {e}")
        return 0.0


def _extract_keywords(text: str) -> set:
    """Extract meaningful keywords from text."""
    # Simple keyword extraction
    words = re.findall(r'\b[a-z]{3,}\b', text)
    
    # Filter out common words
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
    
    keywords = set(word for word in words if word not in stop_words and len(word) > 3)
    return keywords


def architecture_quality_assessment(references: List[str], predictions: List[List[str]]) -> Dict[str, float]:
    """Assess architecture quality based on design decisions and implementation."""
    total_score = 0.0
    total_problems = len(predictions)
    
    # Get all available problem directories
    output_dirs = _get_available_problem_dirs()
    
    for i, pred_list in enumerate(predictions):
        if not pred_list:
            continue
            
        # Get the problem_id for this index
        if i < len(output_dirs):
            problem_id = output_dirs[i]
        else:
            continue
            
        base_path = _get_output_base_path()
        design_path = f"{base_path}/{problem_id}/design.md"
        src_path = f"{base_path}/{problem_id}/src"
        
        if not os.path.exists(design_path) or not os.path.exists(src_path):
            continue
            
        score = _assess_architecture_quality(design_path, src_path)
        total_score += score
    
    return {'architecture_quality_assessment': total_score / total_problems if total_problems > 0 else 0.0}


def _assess_architecture_quality(design_path: str, src_path: str) -> float:
    """Assess architecture quality based on design and implementation alignment."""
    try:
        with open(design_path, 'r', encoding='utf-8') as f:
            design_content = f.read().lower()
        
        score = 0.0
        
        # 1. Design pattern recognition (0.3)
        design_patterns = [
            'microservices', 'mvc', 'mvp', 'mvvm', 'repository', 'factory', 
            'singleton', 'observer', 'strategy', 'adapter', 'facade'
        ]
        pattern_score = sum(0.03 for pattern in design_patterns if pattern in design_content)
        score += min(pattern_score, 0.3)
        
        # 2. Scalability considerations (0.2)
        scalability_terms = [
            'scalability', 'horizontal scaling', 'load balancing', 'caching',
            'database sharding', 'cdn', 'microservices', 'stateless'
        ]
        scalability_score = sum(0.025 for term in scalability_terms if term in design_content)
        score += min(scalability_score, 0.2)
        
        # 3. Security considerations (0.2)
        security_terms = [
            'authentication', 'authorization', 'encryption', 'https', 'oauth',
            'jwt', 'csrf', 'xss', 'sql injection', 'input validation'
        ]
        security_score = sum(0.02 for term in security_terms if term in design_content)
        score += min(security_score, 0.2)
        
        # 4. Performance considerations (0.15)
        performance_terms = [
            'performance', 'latency', 'throughput', 'optimization', 'indexing',
            'query optimization', 'connection pooling', 'async'
        ]
        performance_score = sum(0.02 for term in performance_terms if term in design_content)
        score += min(performance_score, 0.15)
        
        # 5. Implementation alignment (0.15)
        # Check if design concepts are reflected in code structure
        code_structure_score = _check_design_implementation_alignment(design_content, src_path)
        score += code_structure_score * 0.15
        
        return min(score, 1.0)
        
    except Exception as e:
        print(f"Error assessing architecture quality: {e}")
        return 0.0


def _check_design_implementation_alignment(design_content: str, src_path: str) -> float:
    """Check if implementation aligns with design decisions."""
    try:
        alignment_score = 0.0
        
        # Get all Python files
        python_files = []
        for root, dirs, files in os.walk(src_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        if not python_files:
            return 0.0
        
        # Read all code content
        all_code = ""
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    all_code += f.read().lower() + "\n"
            except Exception:
                continue
        
        # Check alignment indicators
        alignment_checks = [
            # If design mentions API, check for API implementation
            ('api', ['flask', 'fastapi', 'django', 'router', 'endpoint']),
            # If design mentions database, check for database code
            ('database', ['sqlalchemy', 'pymongo', 'psycopg', 'sqlite', 'connection']),
            # If design mentions authentication, check for auth code
            ('authentication', ['login', 'password', 'token', 'session', 'auth']),
            # If design mentions testing, check for test files
            ('testing', ['test_', 'unittest', 'pytest', 'mock', 'assert']),
            # If design mentions logging, check for logging code
            ('logging', ['logging', 'logger', 'log.', 'debug', 'info']),
        ]
        
        for design_term, code_indicators in alignment_checks:
            if design_term in design_content:
                if any(indicator in all_code for indicator in code_indicators):
                    alignment_score += 0.2
        
        return min(alignment_score, 1.0)
        
    except Exception:
        return 0.0


def policy_utilization_score(references: List[str], predictions: List[List[str]]) -> Dict[str, float]:
    """Assess how well the solution utilizes the provided context."""
    total_score = 0.0
    total_problems = len(predictions)
    
    # Get all available problem directories
    output_dirs = _get_available_problem_dirs()
    
    for i, pred_list in enumerate(predictions):
        if not pred_list:
            continue
            
        # Get the problem_id for this index
        if i < len(output_dirs):
            problem_id = output_dirs[i]
        else:
            continue
            
        # Load the original problem context
        try:
            # Read the problems.jsonl to get context for this problem
            current_dir = os.path.dirname(__file__)
            dataset_path = os.path.join(current_dir, "problems.jsonl")
            
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            problem_data = None
            for item in data:
                if item['problem_id'] == problem_id:
                    problem_data = item
                    break
            
            if not problem_data:
                continue
                
            base_path = _get_output_base_path()
            prd_path = f"{base_path}/{problem_id}/prd.md"
            design_path = f"{base_path}/{problem_id}/design.md"
            src_path = f"{base_path}/{problem_id}/src"
            
            score = _assess_context_utilization(problem_data, prd_path, design_path, src_path)
            total_score += score
            
        except Exception as e:
            print(f"Error assessing context utilization for {problem_id}: {e}")
            continue
    
    return {'policy_utilization_score': total_score / total_problems if total_problems > 0 else 0.0}


def _assess_context_utilization(problem_data: Dict, prd_path: str, design_path: str, src_path: str) -> float:
    """Assess how well the context information was utilized in the solution."""
    try:
        score = 0.0
        
        # Extract context keywords
        contexts = {
            'prd_context': problem_data.get('prd_context', ''),
            'design_context': problem_data.get('design_context', ''),
            'code_context': problem_data.get('code_context', ''),
            'quality_context': problem_data.get('quality_context', '')
        }
        
        # Read generated files
        generated_content = {}
        
        if os.path.exists(prd_path):
            with open(prd_path, 'r', encoding='utf-8') as f:
                generated_content['prd'] = f.read().lower()
        
        if os.path.exists(design_path):
            with open(design_path, 'r', encoding='utf-8') as f:
                generated_content['design'] = f.read().lower()
        
        if os.path.exists(src_path):
            code_content = ""
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    if file.endswith('.py'):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                code_content += f.read().lower() + "\n"
                        except Exception:
                            continue
            generated_content['code'] = code_content
        
        # Check context utilization for each phase
        context_mappings = [
            ('prd_context', 'prd', 0.25),
            ('design_context', 'design', 0.25),
            ('code_context', 'code', 0.25),
            ('quality_context', ['prd', 'design', 'code'], 0.25)
        ]
        
        for context_key, target_files, weight in context_mappings:
            context_text = contexts.get(context_key, '').lower()
            if not context_text:
                continue
            
            # Extract key terms from context (excluding common words)
            context_terms = set()
            for word in re.findall(r'\b[a-z]{4,}\b', context_text):
                if word not in {'must', 'should', 'include', 'support', 'with', 'using', 'implementation'}:
                    context_terms.add(word)
            
            if not context_terms:
                continue
            
            # Check if context terms appear in generated content
            if isinstance(target_files, str):
                target_files = [target_files]
            
            utilization_score = 0.0
            for target_file in target_files:
                if target_file in generated_content:
                    content = generated_content[target_file]
                    matched_terms = sum(1 for term in context_terms if term in content)
                    utilization_score = max(utilization_score, matched_terms / len(context_terms))
            
            score += utilization_score * weight
        
        return min(score, 1.0)
        
    except Exception as e:
        print(f"Error in context utilization assessment: {e}")
        return 0.0


def policy_adherence_score(references: List[str], predictions: List[List[str]]) -> Dict[str, float]:
    """Check if specific requirements from context are addressed in the solution."""
    total_score = 0.0
    total_problems = len(predictions)
    
    # Get all available problem directories
    output_dirs = _get_available_problem_dirs()
    
    for i, pred_list in enumerate(predictions):
        if not pred_list:
            continue
            
        # Get the problem_id for this index
        if i < len(output_dirs):
            problem_id = output_dirs[i]
        else:
            continue
            
        # Load the original problem context
        try:
            current_dir = os.path.dirname(__file__)
            dataset_path = os.path.join(current_dir, "problems.jsonl")
            
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            problem_data = None
            for item in data:
                if item['problem_id'] == problem_id:
                    problem_data = item
                    break
            
            if not problem_data:
                continue
                
            base_path = _get_output_base_path()
            prd_path = f"{base_path}/{problem_id}/prd.md"
            design_path = f"{base_path}/{problem_id}/design.md"
            src_path = f"{base_path}/{problem_id}/src"
            
            score = _check_specific_requirements(problem_data, prd_path, design_path, src_path)
            total_score += score
            
        except Exception as e:
            print(f"Error checking specific requirements for {problem_id}: {e}")
            continue
    
    return {'policy_adherence_score': total_score / total_problems if total_problems > 0 else 0.0}


def _check_specific_requirements(problem_data: Dict, prd_path: str, design_path: str, src_path: str) -> float:
    """Check for specific requirements that should only appear with context."""
    try:
        score = 0.0
        
        # Read all generated content
        all_content = ""
        for file_path in [prd_path, design_path]:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_content += f.read().lower() + "\n"
        
        if os.path.exists(src_path):
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    if file.endswith(('.py', '.txt', '.md', '.yml', '.yaml', '.json')):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                all_content += f.read().lower() + "\n"
                        except Exception:
                            continue
        
        # Define context-specific requirement patterns
        context_requirements = {
            # Browser compatibility requirements
            'browser_compatibility': [
                r'ie11|internet explorer|cross[- ]browser|browser compatibility',
                r'chrome|firefox|safari|edge compatibility'
            ],
            # Performance requirements
            'performance_metrics': [
                r'\d+\s*(ms|milliseconds?|seconds?)\s*(latency|response|load)',
                r'lighthouse\s+score|performance\s+budget|page\s+load\s+time',
                r'\d+k?\s*(users?|concurrent|requests?)\s*per'
            ],
            # Accessibility requirements
            'accessibility': [
                r'wcag|accessibility|screen\s+reader|aria|alt\s+text',
                r'contrast\s+ratio|keyboard\s+navigation|focus\s+management'
            ],
            # Security requirements
            'security_standards': [
                r'oauth|jwt|encryption|https|ssl|tls',
                r'csrf|xss|sql\s+injection|input\s+validation',
                r'gdpr|sox|compliance|audit|penetration\s+testing'
            ],
            # Technology constraints
            'technology_constraints': [
                r'spring\s+boot|flask|fastapi|django|react|vue|angular',
                r'postgresql|mongodb|redis|kafka|elasticsearch',
                r'kubernetes|docker|microservices|api\s+gateway'
            ],
            # Architecture patterns
            'architecture_patterns': [
                r'cqrs|event\s+sourcing|saga\s+pattern|circuit\s+breaker',
                r'domain\s+driven\s+design|ddd|bounded\s+context',
                r'microservices|event[- ]driven|message\s+queue'
            ]
        }
        
        # Check each requirement category
        for category, patterns in context_requirements.items():
            category_score = 0.0
            for pattern in patterns:
                if re.search(pattern, all_content, re.IGNORECASE):
                    category_score = 1.0
                    break
            score += category_score * (1.0 / len(context_requirements))
        
        return min(score, 1.0)
        
    except Exception as e:
        print(f"Error checking specific requirements: {e}")
        return 0.0


def technical_constraint_adherence(references: List[str], predictions: List[List[str]]) -> Dict[str, float]:
    """Check adherence to technical constraints specified in context."""
    total_score = 0.0
    total_problems = len(predictions)
    
    # Get all available problem directories
    output_dirs = _get_available_problem_dirs()
    
    for i, pred_list in enumerate(predictions):
        if not pred_list:
            continue
            
        # Get the problem_id for this index
        if i < len(output_dirs):
            problem_id = output_dirs[i]
        else:
            continue
            
        # Load the original problem context
        try:
            current_dir = os.path.dirname(__file__)
            dataset_path = os.path.join(current_dir, "problems.jsonl")
            
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            problem_data = None
            for item in data:
                if item['problem_id'] == problem_id:
                    problem_data = item
                    break
            
            if not problem_data:
                continue
                
            base_path = _get_output_base_path()
            src_path = f"{base_path}/{problem_id}/src"
            design_path = f"{base_path}/{problem_id}/design.md"
            
            score = _check_technical_constraints(problem_data, design_path, src_path)
            total_score += score
            
        except Exception as e:
            print(f"Error checking technical constraints for {problem_id}: {e}")
            continue
    
    return {'technical_constraint_adherence': total_score / total_problems if total_problems > 0 else 0.0}


def _check_technical_constraints(problem_data: Dict, design_path: str, src_path: str) -> float:
    """Check if technical constraints from context are followed."""
    try:
        score = 0.0
        
        # Get all context text
        all_context = ""
        for context_key in ['prd_context', 'design_context', 'code_context', 'quality_context']:
            context_text = problem_data.get(context_key, '')
            all_context += context_text.lower() + " "
        
        # Read generated content
        generated_content = ""
        if os.path.exists(design_path):
            with open(design_path, 'r', encoding='utf-8') as f:
                generated_content += f.read().lower() + "\n"
        
        if os.path.exists(src_path):
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    if file.endswith(('.py', '.js', '.html', '.css', '.yml', '.yaml', '.json', '.txt')):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                generated_content += f.read().lower() + "\n"
                        except Exception:
                            continue
        
        # Define constraint checks
        constraint_checks = [
            # Framework constraints
            {
                'context_patterns': [r'spring\s+boot', r'flask', r'fastapi', r'django'],
                'implementation_patterns': [r'from\s+flask', r'import\s+flask', r'spring', r'fastapi', r'django'],
                'weight': 0.2
            },
            # Database constraints
            {
                'context_patterns': [r'postgresql', r'mongodb', r'redis', r'sqlite'],
                'implementation_patterns': [r'psycopg', r'pymongo', r'redis', r'sqlite', r'sqlalchemy'],
                'weight': 0.15
            },
            # Testing constraints
            {
                'context_patterns': [r'pytest', r'unittest', r'testing', r'test\s+coverage'],
                'implementation_patterns': [r'import\s+pytest', r'import\s+unittest', r'def\s+test_', r'class\s+Test'],
                'weight': 0.15
            },
            # Code style constraints
            {
                'context_patterns': [r'pep\s*8', r'black\s+formatter', r'type\s+hints', r'docstrings'],
                'implementation_patterns': [r':\s*\w+\s*=', r'def\s+\w+\([^)]*:\s*\w+', r'"""', r"'''"],
                'weight': 0.15
            },
            # Architecture constraints
            {
                'context_patterns': [r'microservices', r'api\s+gateway', r'event[- ]driven', r'cqrs'],
                'implementation_patterns': [r'router', r'endpoint', r'event', r'command', r'query'],
                'weight': 0.2
            },
            # Security constraints
            {
                'context_patterns': [r'oauth', r'jwt', r'authentication', r'authorization'],
                'implementation_patterns': [r'auth', r'token', r'login', r'password', r'session'],
                'weight': 0.15
            }
        ]
        
        for constraint in constraint_checks:
            # Check if constraint is mentioned in context
            context_mentioned = any(
                re.search(pattern, all_context, re.IGNORECASE) 
                for pattern in constraint['context_patterns']
            )
            
            if context_mentioned:
                # Check if constraint is implemented
                implemented = any(
                    re.search(pattern, generated_content, re.IGNORECASE)
                    for pattern in constraint['implementation_patterns']
                )
                
                if implemented:
                    score += constraint['weight']
        
        return min(score, 1.0)
        
    except Exception as e:
        print(f"Error checking technical constraints: {e}")
        return 0.0


def performance_requirement_coverage(references: List[str], predictions: List[List[str]]) -> Dict[str, float]:
    """Check if performance requirements from context are addressed."""
    total_score = 0.0
    total_problems = len(predictions)
    
    # Get all available problem directories
    output_dirs = _get_available_problem_dirs()
    
    for i, pred_list in enumerate(predictions):
        if not pred_list:
            continue
            
        # Get the problem_id for this index
        if i < len(output_dirs):
            problem_id = output_dirs[i]
        else:
            continue
            
        # Load the original problem context
        try:
            current_dir = os.path.dirname(__file__)
            dataset_path = os.path.join(current_dir, "problems.jsonl")
            
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            problem_data = None
            for item in data:
                if item['problem_id'] == problem_id:
                    problem_data = item
                    break
            
            if not problem_data:
                continue
                
            base_path = _get_output_base_path()
            prd_path = f"{base_path}/{problem_id}/prd.md"
            design_path = f"{base_path}/{problem_id}/design.md"
            src_path = f"{base_path}/{problem_id}/src"
            
            score = _check_performance_requirements(problem_data, prd_path, design_path, src_path)
            total_score += score
            
        except Exception as e:
            print(f"Error checking performance requirements for {problem_id}: {e}")
            continue
    
    return {'performance_requirement_coverage': total_score / total_problems if total_problems > 0 else 0.0}


def _check_performance_requirements(problem_data: Dict, prd_path: str, design_path: str, src_path: str) -> float:
    """Check if performance requirements are addressed."""
    try:
        score = 0.0
        
        # Get all context text
        all_context = ""
        for context_key in ['prd_context', 'design_context', 'code_context', 'quality_context']:
            context_text = problem_data.get(context_key, '')
            all_context += context_text.lower() + " "
        
        # Read generated content
        all_generated = ""
        for file_path in [prd_path, design_path]:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_generated += f.read().lower() + "\n"
        
        if os.path.exists(src_path):
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    if file.endswith(('.py', '.md', '.txt', '.yml', '.yaml')):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                all_generated += f.read().lower() + "\n"
                        except Exception:
                            continue
        
        # Performance requirement patterns in context
        perf_context_patterns = [
            r'\d+\s*(ms|milliseconds?|seconds?)\s*(latency|response|load)',
            r'\d+k?\s*(users?|concurrent|requests?)',
            r'performance\s+budget|lighthouse\s+score',
            r'page\s+load\s+time|response\s+time',
            r'throughput|scalability|optimization'
        ]
        
        # Check if performance requirements are mentioned in context
        perf_mentioned = any(
            re.search(pattern, all_context, re.IGNORECASE)
            for pattern in perf_context_patterns
        )
        
        if not perf_mentioned:
            return 0.5  # No performance requirements in context
        
        # Performance implementation patterns in generated content
        perf_implementation_patterns = [
            # Caching
            (r'cach(e|ing)|redis|memcached', 0.2),
            # Database optimization
            (r'index|query\s+optimization|connection\s+pool', 0.2),
            # Async/concurrent processing
            (r'async|await|concurrent|threading|multiprocess', 0.2),
            # Load balancing/scaling
            (r'load\s+balanc|horizontal\s+scal|auto\s*scal', 0.15),
            # Performance monitoring
            (r'monitor|metric|profil|benchmark|performance\s+test', 0.15),
            # CDN/static optimization
            (r'cdn|static\s+file|minif|compress|gzip', 0.1)
        ]
        
        for pattern, weight in perf_implementation_patterns:
            if re.search(pattern, all_generated, re.IGNORECASE):
                score += weight
        
        return min(score, 1.0)
        
    except Exception as e:
        print(f"Error checking performance requirements: {e}")
        return 0.0


def security_compliance_check(references: List[str], predictions: List[List[str]]) -> Dict[str, float]:
    """Check if security and compliance requirements from context are addressed."""
    total_score = 0.0
    total_problems = len(predictions)
    
    # Get all available problem directories
    output_dirs = _get_available_problem_dirs()
    
    for i, pred_list in enumerate(predictions):
        if not pred_list:
            continue
            
        # Get the problem_id for this index
        if i < len(output_dirs):
            problem_id = output_dirs[i]
        else:
            continue
            
        # Load the original problem context
        try:
            current_dir = os.path.dirname(__file__)
            dataset_path = os.path.join(current_dir, "problems.jsonl")
            
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            problem_data = None
            for item in data:
                if item['problem_id'] == problem_id:
                    problem_data = item
                    break
            
            if not problem_data:
                continue
                
            base_path = _get_output_base_path()
            prd_path = f"{base_path}/{problem_id}/prd.md"
            design_path = f"{base_path}/{problem_id}/design.md"
            src_path = f"{base_path}/{problem_id}/src"
            
            score = _check_security_compliance(problem_data, prd_path, design_path, src_path)
            total_score += score
            
        except Exception as e:
            print(f"Error checking security compliance for {problem_id}: {e}")
            continue
    
    return {'security_compliance_check': total_score / total_problems if total_problems > 0 else 0.0}


def _check_security_compliance(problem_data: Dict, prd_path: str, design_path: str, src_path: str) -> float:
    """Check if security and compliance requirements are addressed."""
    try:
        score = 0.0
        
        # Get all context text
        all_context = ""
        for context_key in ['prd_context', 'design_context', 'code_context', 'quality_context']:
            context_text = problem_data.get(context_key, '')
            all_context += context_text.lower() + " "
        
        # Read generated content
        all_generated = ""
        for file_path in [prd_path, design_path]:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_generated += f.read().lower() + "\n"
        
        if os.path.exists(src_path):
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    if file.endswith(('.py', '.md', '.txt', '.yml', '.yaml', '.json')):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                all_generated += f.read().lower() + "\n"
                        except Exception:
                            continue
        
        # Security requirement patterns in context
        security_context_patterns = [
            r'oauth|jwt|authentication|authorization',
            r'encryption|https|ssl|tls',
            r'gdpr|sox|compliance|audit',
            r'csrf|xss|sql\s+injection',
            r'penetration\s+test|security\s+scan',
            r'wcag|accessibility|screen\s+reader'
        ]
        
        # Check if security requirements are mentioned in context
        security_mentioned = any(
            re.search(pattern, all_context, re.IGNORECASE)
            for pattern in security_context_patterns
        )
        
        if not security_mentioned:
            return 0.5  # No security requirements in context
        
        # Security implementation patterns in generated content
        security_implementation_patterns = [
            # Authentication/Authorization
            (r'auth|login|password|token|session|jwt|oauth', 0.25),
            # Input validation
            (r'validat|sanitiz|escap|input\s+check', 0.2),
            # HTTPS/Encryption
            (r'https|ssl|tls|encrypt|hash|bcrypt', 0.2),
            # Security headers
            (r'csrf|cors|content\s+security\s+policy|x-frame', 0.15),
            # Compliance measures
            (r'audit|log|monitor|compliance|gdpr|privacy', 0.1),
            # Accessibility
            (r'aria|alt\s+text|screen\s+reader|accessibility|wcag', 0.1)
        ]
        
        for pattern, weight in security_implementation_patterns:
            if re.search(pattern, all_generated, re.IGNORECASE):
                score += weight
        
        return min(score, 1.0)
        
    except Exception as e:
        print(f"Error checking security compliance: {e}")
        return 0.0


def execution_time_efficiency(references: List[str], predictions: List[List[str]]) -> Dict[str, float]:
    """Measure execution time efficiency - lower is better."""
    total_time = 0.0
    total_problems = len(predictions)
    
    # Get all available problem directories
    output_dirs = _get_available_problem_dirs()
    
    for i, pred_list in enumerate(predictions):
        if not pred_list:
            continue
            
        # Get the problem_id for this index
        if i < len(output_dirs):
            problem_id = output_dirs[i]
        else:
            continue
            
        # Try to extract timing information from the evaluation logs
        try:
            # Look for timing information in the current evaluation context
            # This will be populated by the evaluation framework
            execution_time = _extract_execution_time(problem_id)
            total_time += execution_time
            
        except Exception as e:
            print(f"Error extracting execution time for {problem_id}: {e}")
            # Use a default time if extraction fails
            total_time += 60.0  # Default 60 seconds
            continue
    
    # Return average execution time in seconds
    avg_time = total_time / total_problems if total_problems > 0 else 0.0
    return {'execution_time_efficiency': avg_time}


def _extract_execution_time(problem_id: str) -> float:
    """Extract execution time from evaluation logs or metadata."""
    try:
        # Try to find timing information from recent log files or result metadata
        import glob
        import time
        
        # Look for recent result files that might contain timing info
        base_path = _get_output_base_path()
        result_files = glob.glob(f"results/**/samples_*{problem_id}*.jsonl", recursive=True)
        
        if result_files:
            # Get the most recent file
            latest_file = max(result_files, key=os.path.getmtime)
            
            try:
                with open(latest_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            # Look for timing information in the response data
                            if 'resps' in data and data['resps']:
                                resp = data['resps'][0][0]
                                # Extract timing from Claude Code response if available
                                if 'duration_ms' in resp:
                                    return float(resp.split('duration_ms=')[1].split(',')[0]) / 1000.0
                                elif 'ResultMessage' in resp and 'duration_ms' in resp:
                                    # Parse duration from ResultMessage
                                    import re
                                    match = re.search(r'duration_ms=(\d+)', resp)
                                    if match:
                                        return float(match.group(1)) / 1000.0
            except Exception:
                pass
        
        # Fallback: estimate based on file modification times
        prd_path = f"{base_path}/{problem_id}/prd.md"
        design_path = f"{base_path}/{problem_id}/design.md"
        src_path = f"{base_path}/{problem_id}/src"
        
        timestamps = []
        for path in [prd_path, design_path, src_path]:
            if os.path.exists(path):
                timestamps.append(os.path.getmtime(path))
        
        if len(timestamps) >= 2:
            # Estimate execution time as difference between first and last file
            return max(timestamps) - min(timestamps)
        
        # Default fallback
        return 45.0  # Default 45 seconds
        
    except Exception as e:
        print(f"Error in execution time extraction: {e}")
        return 45.0  # Default fallback


def token_cost_estimation(references: List[str], predictions: List[List[str]]) -> Dict[str, float]:
    """Estimate token cost based on generated content - lower is better."""
    total_cost = 0.0
    total_problems = len(predictions)
    
    # Get all available problem directories
    output_dirs = _get_available_problem_dirs()
    
    for i, pred_list in enumerate(predictions):
        if not pred_list:
            continue
            
        # Get the problem_id for this index
        if i < len(output_dirs):
            problem_id = output_dirs[i]
        else:
            continue
            
        try:
            # Calculate cost based on generated content and API usage
            cost = _calculate_token_cost(problem_id, pred_list)
            total_cost += cost
            
        except Exception as e:
            print(f"Error calculating token cost for {problem_id}: {e}")
            continue
    
    # Return average cost in USD
    avg_cost = total_cost / total_problems if total_problems > 0 else 0.0
    return {'token_cost_estimation': avg_cost}


def _calculate_token_cost(problem_id: str, pred_list: List[str]) -> float:
    """Calculate estimated token cost for a problem."""
    try:
        total_cost = 0.0
        
        # 1. Extract actual API cost from logs if available
        api_cost = _extract_api_cost_from_logs(problem_id)
        if api_cost > 0:
            return api_cost
        
        # 2. Fallback: estimate based on content length
        base_path = _get_output_base_path()
        
        # Count tokens in generated content
        total_chars = 0
        
        # Count PRD content
        prd_path = f"{base_path}/{problem_id}/prd.md"
        if os.path.exists(prd_path):
            with open(prd_path, 'r', encoding='utf-8') as f:
                total_chars += len(f.read())
        
        # Count design content
        design_path = f"{base_path}/{problem_id}/design.md"
        if os.path.exists(design_path):
            with open(design_path, 'r', encoding='utf-8') as f:
                total_chars += len(f.read())
        
        # Count code content
        src_path = f"{base_path}/{problem_id}/src"
        if os.path.exists(src_path):
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    if file.endswith(('.py', '.md', '.txt', '.yml', '.yaml', '.json')):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                total_chars += len(f.read())
                        except Exception:
                            continue
        
        # Count prediction content
        for pred in pred_list:
            total_chars += len(pred)
        
        # Estimate tokens (roughly 4 characters per token)
        estimated_output_tokens = total_chars / 4
        
        # Estimate input tokens (context + prompts)
        # Load problem data to estimate input size
        current_dir = os.path.dirname(__file__)
        dataset_path = os.path.join(current_dir, "problems.jsonl")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        problem_data = None
        for item in data:
            if item['problem_id'] == problem_id:
                problem_data = item
                break
        
        if problem_data:
            # Estimate input tokens from problem description and context
            input_text = (
                problem_data.get('problem_description', '') +
                problem_data.get('prd_context', '') +
                problem_data.get('design_context', '') +
                problem_data.get('code_context', '') +
                problem_data.get('quality_context', '')
            )
            estimated_input_tokens = len(input_text) / 4
        else:
            estimated_input_tokens = 1000  # Default estimate
        
        # Claude 3.5 Sonnet pricing (as of 2024)
        # Input: $3.00 per 1M tokens, Output: $15.00 per 1M tokens
        input_cost = (estimated_input_tokens / 1_000_000) * 3.00
        output_cost = (estimated_output_tokens / 1_000_000) * 15.00
        
        total_cost = input_cost + output_cost
        
        return total_cost
        
    except Exception as e:
        print(f"Error calculating token cost: {e}")
        return 0.01  # Default small cost


def _extract_api_cost_from_logs(problem_id: str) -> float:
    """Extract actual API cost from evaluation logs."""
    try:
        import glob
        
        # Look for recent result files that might contain cost info
        result_files = glob.glob(f"results/**/samples_*{problem_id}*.jsonl", recursive=True)
        
        if result_files:
            # Get the most recent file
            latest_file = max(result_files, key=os.path.getmtime)
            
            try:
                with open(latest_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            # Look for cost information in the response data
                            if 'resps' in data and data['resps']:
                                resp = data['resps'][0][0]
                                # Extract cost from Claude Code response if available
                                if 'total_cost_usd' in resp:
                                    import re
                                    match = re.search(r'total_cost_usd=([0-9.]+)', resp)
                                    if match:
                                        return float(match.group(1))
            except Exception:
                pass
        
        return 0.0  # No cost information found
        
    except Exception:
        return 0.0