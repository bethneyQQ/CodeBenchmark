"""
Scenario Registry for Multi-Turn Evaluation.

This module provides registration and discovery mechanisms for multi-turn scenarios.
"""

from typing import Dict, Type, List, Optional
from .base_scenario import MultiTurnScenario, ScenarioConfig, ScenarioType


class ScenarioRegistry:
    """Registry for multi-turn scenarios."""
    
    def __init__(self):
        self._scenarios: Dict[str, Type[MultiTurnScenario]] = {}
        self._configs: Dict[str, ScenarioConfig] = {}
        
    def register(self, scenario_id: str, scenario_class: Type[MultiTurnScenario], 
                config: Optional[ScenarioConfig] = None):
        """Register a scenario class."""
        self._scenarios[scenario_id] = scenario_class
        if config:
            self._configs[scenario_id] = config
            
    def get_scenario_class(self, scenario_id: str) -> Optional[Type[MultiTurnScenario]]:
        """Get a registered scenario class."""
        return self._scenarios.get(scenario_id)
        
    def get_scenario_config(self, scenario_id: str) -> Optional[ScenarioConfig]:
        """Get a registered scenario configuration."""
        return self._configs.get(scenario_id)
        
    def create_scenario(self, scenario_id: str, 
                       config: Optional[ScenarioConfig] = None) -> Optional[MultiTurnScenario]:
        """Create an instance of a registered scenario."""
        scenario_class = self.get_scenario_class(scenario_id)
        if not scenario_class:
            return None
            
        use_config = config or self.get_scenario_config(scenario_id)
        if not use_config:
            raise ValueError(f"No configuration available for scenario: {scenario_id}")
            
        return scenario_class(use_config)
        
    def list_scenarios(self) -> List[str]:
        """List all registered scenario IDs."""
        return list(self._scenarios.keys())
        
    def list_by_type(self, scenario_type: ScenarioType) -> List[str]:
        """List scenarios by type."""
        matching_scenarios = []
        for scenario_id in self._scenarios:
            config = self._configs.get(scenario_id)
            if config and config.scenario_type == scenario_type:
                matching_scenarios.append(scenario_id)
        return matching_scenarios


# Global registry instance
_global_registry = ScenarioRegistry()


def register_scenario(scenario_id: str, config: Optional[ScenarioConfig] = None):
    """
    Decorator to register a scenario class.
    
    Usage:
        @register_scenario("my_scenario", my_config)
        class MyScenario(MultiTurnScenario):
            pass
    """
    def decorator(scenario_class: Type[MultiTurnScenario]):
        _global_registry.register(scenario_id, scenario_class, config)
        return scenario_class
    return decorator


def get_scenario(scenario_id: str) -> Optional[Type[MultiTurnScenario]]:
    """Get a registered scenario class."""
    return _global_registry.get_scenario_class(scenario_id)


def create_scenario(scenario_id: str, 
                   config: Optional[ScenarioConfig] = None) -> Optional[MultiTurnScenario]:
    """Create a scenario instance."""
    return _global_registry.create_scenario(scenario_id, config)


def list_scenarios() -> List[str]:
    """List all registered scenarios."""
    return _global_registry.list_scenarios()


def list_scenarios_by_type(scenario_type: ScenarioType) -> List[str]:
    """List scenarios by type."""
    return _global_registry.list_by_type(scenario_type)


def get_scenario_registry() -> Dict[str, type]:
    """Get the complete scenario registry mapping.""" 
    return _global_registry._scenarios.copy()