"""Single Turn Scenarios evaluation tasks for lm_eval.

This module provides a comprehensive evaluation framework that unifies the strengths 
of existing python_coding and multi_turn_generic tasks into a single, powerful 
evaluation system with multi-language support, contextual configurations, and 
authoritative metrics.

Task Registration:
- Individual scenario tasks: single_turn_scenarios_<scenario>
- Full suite evaluation: single_turn_scenarios_suite
- Metadata filtering support for difficulty, language, context_mode

Supported Scenarios:
- Basic: code_completion, bug_fix, code_translation, documentation, function_generation
- Advanced: system_design, algorithm_implementation, api_design, database_design, performance_optimization  
- Comprehensive: full_stack, testing_strategy, security
"""

# Import scenario-specific task modules for automatic discovery
# These imports ensure that the YAML task configurations are discoverable by lm-eval
from . import code_completion
from . import bug_fix
from . import code_translation
from . import documentation
from . import function_generation
from . import algorithm_implementation
from . import system_design
from . import api_design
from . import database_design
from . import performance_optimization
from . import full_stack
from . import testing_strategy
from . import security

# Task registration is handled automatically by lm-eval's TaskManager
# which discovers YAML files in this directory and registers them based on their 'task' field