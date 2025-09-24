"""
Multi-Turn Evaluation Engine.

This module provides the core evaluation engine that orchestrates
multi-turn scenario evaluation with chat template support.
"""

import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import asdict

from .base_scenario import MultiTurnScenario, ScenarioConfig
from .chat_template_support import ChatTemplateManager, MultiTurnChatTemplateIntegrator
from .scenario_registry import create_scenario
from .metrics import MultiTurnMetrics


logger = logging.getLogger(__name__)


class MultiTurnEvaluationEngine:
    """
    Core engine for multi-turn scenario evaluation.
    
    This engine coordinates the evaluation process, manages chat templates,
    executes scenarios, and collects metrics.
    """
    
    def __init__(self, 
                 chat_template_manager: Optional[ChatTemplateManager] = None,
                 enable_chat_templates: bool = True):
        self.chat_manager = chat_template_manager or ChatTemplateManager()
        self.chat_integrator = MultiTurnChatTemplateIntegrator(self.chat_manager)
        self.enable_chat_templates = enable_chat_templates
        self.metrics_calculator = MultiTurnMetrics()
        
    def evaluate_scenario(self,
                         scenario_id: str,
                         problem_data: Dict[str, Any],
                         model_generate_fn: Callable[[str], str],
                         scenario_config: Optional[ScenarioConfig] = None,
                         model_template_fn: Optional[Callable] = None,
                         evaluation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a multi-turn scenario.
        
        Args:
            scenario_id: ID of the scenario to evaluate
            problem_data: Input data for the scenario
            model_generate_fn: Function to generate model responses
            scenario_config: Optional custom scenario configuration
            model_template_fn: Optional model chat template function
            evaluation_config: Optional evaluation configuration
            
        Returns:
            Complete evaluation results
        """
        logger.info(f"Starting evaluation for scenario: {scenario_id}")
        
        # Create scenario instance
        scenario = create_scenario(scenario_id, scenario_config)
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_id}")
            
        # Configure chat templates if enabled
        if self.enable_chat_templates and scenario.config.chat_template_required:
            self._setup_chat_templates(scenario, model_template_fn)
            
        # Execute scenario
        try:
            results = self._execute_scenario(
                scenario, problem_data, model_generate_fn, evaluation_config or {}
            )
            
            logger.info(f"Completed evaluation for scenario: {scenario_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating scenario {scenario_id}: {e}")
            return {
                "scenario_id": scenario_id,
                "status": "error",
                "error": str(e),
                "partial_results": getattr(scenario, 'turn_results', {})
            }
            
    def _setup_chat_templates(self, 
                            scenario: MultiTurnScenario, 
                            model_template_fn: Optional[Callable]):
        """Setup chat templates for scenario."""
        if model_template_fn:
            # Update scenario config to use model's chat template
            scenario.config.metadata["model_template_fn"] = model_template_fn
            logger.info("Configured scenario with model chat template")
        else:
            logger.info("Using default chat template")
            
    def _execute_scenario(self,
                         scenario: MultiTurnScenario,
                         problem_data: Dict[str, Any],
                         model_generate_fn: Callable[[str], str],
                         evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete multi-turn scenario."""
        
        # Initialize scenario state
        scenario.global_state.update(problem_data)
        execution_log = []
        
        # Generate initial prompt
        initial_prompt = scenario.generate_initial_prompt(problem_data)
        
        if scenario.config.chat_template_required:
            initial_prompt = self._apply_chat_template(scenario, initial_prompt, [], {})
            
        execution_log.append({
            "phase": "initial_prompt",
            "prompt": initial_prompt,
            "timestamp": len(execution_log)
        })
        
        # Execute turns
        current_turn_idx = 0
        previous_responses = []
        
        for turn_config in scenario.config.turns:
            try:
                turn_result = self._execute_turn(
                    scenario, turn_config, model_generate_fn,
                    previous_responses, current_turn_idx, execution_log
                )
                
                if turn_result:
                    previous_responses.append(turn_result["response"])
                    current_turn_idx += 1
                    
            except Exception as e:
                logger.error(f"Error in turn {turn_config.turn_id}: {e}")
                execution_log.append({
                    "phase": "turn_error",
                    "turn_id": turn_config.turn_id,
                    "error": str(e),
                    "timestamp": len(execution_log)
                })
                break
                
        # Final scenario evaluation
        scenario_metrics = scenario.evaluate_scenario()
        
        # Compile results
        results = {
            "scenario_id": scenario.config.scenario_id,
            "scenario_type": scenario.config.scenario_type.value,
            "status": "completed",
            "execution_log": execution_log,
            "turn_results": scenario.turn_results,
            "conversation_history": scenario.conversation_history,
            "scenario_metrics": scenario_metrics,
            "aggregated_metrics": self._calculate_aggregated_metrics(scenario),
            "metadata": {
                "total_turns": len(scenario.config.turns),
                "completed_turns": current_turn_idx,
                "chat_template_used": scenario.config.chat_template_required,
                "scenario_config": asdict(scenario.config)
            }
        }
        
        return results
        
    def _execute_turn(self,
                     scenario: MultiTurnScenario,
                     turn_config,
                     model_generate_fn: Callable[[str], str],
                     previous_responses: List[str],
                     turn_idx: int,
                     execution_log: List[Dict]) -> Optional[Dict[str, Any]]:
        """Execute a single turn in the scenario."""
        
        turn_id = turn_config.turn_id
        logger.debug(f"Executing turn: {turn_id}")
        
        # Check dependencies
        if turn_config.depends_on:
            missing_deps = [
                dep for dep in turn_config.depends_on 
                if dep not in scenario.turn_results
            ]
            if missing_deps:
                logger.warning(f"Turn {turn_id} missing dependencies: {missing_deps}")
                return None
                
        # Generate prompt for this turn
        if turn_idx == 0:
            # Use initial prompt
            prompt = scenario.generate_initial_prompt(scenario.global_state)
        else:
            # Generate next prompt
            prompt = scenario.generate_next_prompt(turn_id, previous_responses)
            
        if not prompt:
            logger.info(f"No prompt generated for turn {turn_id}, skipping")
            return None
            
        # Apply chat template if needed
        if scenario.config.chat_template_required:
            context = scenario.get_context_for_turn(turn_id)
            prompt = self._apply_chat_template(scenario, prompt, scenario.conversation_history, context)
            
        execution_log.append({
            "phase": "turn_prompt",
            "turn_id": turn_id,
            "prompt": prompt,
            "timestamp": len(execution_log)
        })
        
        # Generate model response
        try:
            response = model_generate_fn(prompt)
            
            execution_log.append({
                "phase": "turn_response",
                "turn_id": turn_id,
                "response": response,
                "timestamp": len(execution_log)
            })
            
        except Exception as e:
            logger.error(f"Model generation failed for turn {turn_id}: {e}")
            execution_log.append({
                "phase": "generation_error",
                "turn_id": turn_id,
                "error": str(e),
                "timestamp": len(execution_log)
            })
            return None
            
        # Process turn response
        context = scenario.get_context_for_turn(turn_id)
        turn_result = scenario.process_turn_response(turn_id, response, context)
        
        # Validate response if configured
        validation_errors = scenario.validate_turn_response(turn_id, response)
        if validation_errors:
            turn_result["validation_errors"] = validation_errors
            execution_log.append({
                "phase": "validation_errors",
                "turn_id": turn_id,
                "errors": validation_errors,
                "timestamp": len(execution_log)
            })
            
        execution_log.append({
            "phase": "turn_processed",
            "turn_id": turn_id,
            "result_summary": {k: v for k, v in turn_result.items() if k != "response"},
            "timestamp": len(execution_log)
        })
        
        return turn_result
        
    def _apply_chat_template(self,
                           scenario: MultiTurnScenario,
                           prompt: str,
                           conversation_history: List[Dict[str, Any]],
                           context: Dict[str, Any]) -> str:
        """Apply chat template to prompt."""
        
        model_template_fn = scenario.config.metadata.get("model_template_fn")
        
        if model_template_fn:
            # Use model's chat template
            chat_messages = scenario.format_chat_history(model_template_fn)
            chat_messages.append({"role": "user", "content": prompt})
            return model_template_fn(chat_messages)
        else:
            # Use default chat template
            return self.chat_manager.create_multi_turn_prompt(
                scenario.config.system_message,
                conversation_history,
                prompt,
                template="default"
            )
            
    def _calculate_aggregated_metrics(self, scenario: MultiTurnScenario) -> Dict[str, float]:
        """Calculate aggregated metrics across all turns."""
        return self.metrics_calculator.calculate_aggregated_metrics(
            scenario.turn_results,
            scenario.conversation_history,
            scenario.config
        )
        
    def batch_evaluate(self,
                      evaluations: List[Dict[str, Any]],
                      model_generate_fn: Callable[[str], str],
                      model_template_fn: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Evaluate multiple scenarios in batch.
        
        Args:
            evaluations: List of evaluation configurations
            model_generate_fn: Model generation function
            model_template_fn: Optional model chat template function
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, eval_config in enumerate(evaluations):
            logger.info(f"Processing evaluation {i+1}/{len(evaluations)}")
            
            try:
                result = self.evaluate_scenario(
                    scenario_id=eval_config["scenario_id"],
                    problem_data=eval_config["problem_data"],
                    model_generate_fn=model_generate_fn,
                    scenario_config=eval_config.get("scenario_config"),
                    model_template_fn=model_template_fn,
                    evaluation_config=eval_config.get("evaluation_config")
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed evaluation {i+1}: {e}")
                results.append({
                    "scenario_id": eval_config.get("scenario_id", "unknown"),
                    "status": "error",
                    "error": str(e)
                })
                
        return results
    
    def get_scenario(self, scenario_id: str, problem_data: Dict[str, Any] = None) -> Optional[MultiTurnScenario]:
        """Get a scenario instance for testing."""
        return create_scenario(scenario_id)
        
    def get_scenario_summary(self, scenario_id: str) -> Dict[str, Any]:
        """Get summary information about a scenario."""
        scenario = create_scenario(scenario_id)
        if not scenario:
            return {"error": f"Unknown scenario: {scenario_id}"}
            
        return {
            "scenario_id": scenario.config.scenario_id,
            "scenario_type": scenario.config.scenario_type.value,
            "name": scenario.config.name,
            "description": scenario.config.description,
            "turn_count": len(scenario.config.turns),
            "chat_template_required": scenario.config.chat_template_required,
            "evaluation_strategy": scenario.config.evaluation_strategy,
            "success_criteria": scenario.config.success_criteria
        }