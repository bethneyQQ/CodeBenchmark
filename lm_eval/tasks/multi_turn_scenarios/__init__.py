"""
Multi-Turn Scenarios Framework

This package provides a comprehensive framework for evaluating models across
different multi-turn scenarios with chat-template support.
"""

from .base_scenario import MultiTurnScenario, TurnConfig, ScenarioConfig
from .scenario_registry import register_scenario, get_scenario, get_scenario_registry
from .chat_template_support import ChatTemplateManager
from .evaluation_engine import MultiTurnEvaluationEngine
from .base_scenario import ScenarioType

__all__ = [
    'MultiTurnScenario',
    'TurnConfig', 
    'ScenarioConfig',
    'ScenarioType',
    'register_scenario',
    'get_scenario',
    'get_scenario_registry',
    'ChatTemplateManager',
    'MultiTurnEvaluationEngine'
]