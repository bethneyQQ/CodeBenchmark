"""
Base classes for multi-turn scenarios.

This module defines the core abstractions for creating different types of
multi-turn evaluation scenarios.
"""

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable
from enum import Enum


class TurnType(Enum):
    """Types of turns in a multi-turn scenario."""
    USER_QUERY = "user"
    ASSISTANT_RESPONSE = "assistant"  
    SYSTEM_INSTRUCTION = "system"
    EVALUATION_POINT = "evaluation"
    FEEDBACK = "feedback"
    REVISION = "revision"


class ScenarioType(Enum):
    """Types of multi-turn scenarios."""
    CONVERSATIONAL = "conversational"  # Natural dialogue
    WORKFLOW = "workflow"              # Sequential task completion
    ITERATIVE = "iterative"           # Iterative refinement
    COLLABORATIVE = "collaborative"   # Multi-agent collaboration
    INSTRUCTIONAL = "instructional"   # Teaching/learning dialogue
    DEBUG_SESSION = "debug"           # Problem-solving session
    CODE_REVIEW = "code_review"       # Code review process
    DESIGN_ITERATION = "design_iteration"  # Design refinement


@dataclass
class TurnConfig:
    """Configuration for a single turn in a multi-turn scenario."""
    turn_id: str
    turn_type: TurnType
    role: str  # user, assistant, system, etc.
    prompt_template: str
    expected_format: Optional[str] = None
    validation_rules: List[Callable] = field(default_factory=list)
    evaluation_metrics: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)  # Turn IDs this depends on
    context_window: int = -1  # -1 means use all previous turns
    temperature: float = 0.0
    max_tokens: int = 2000
    stop_sequences: List[str] = field(default_factory=list)


@dataclass  
class ScenarioConfig:
    """Configuration for a complete multi-turn scenario."""
    scenario_id: str
    scenario_type: ScenarioType
    name: str
    description: str
    turns: List[TurnConfig]
    global_context: Dict[str, Any] = field(default_factory=dict)
    chat_template_required: bool = False
    system_message: Optional[str] = None
    success_criteria: List[str] = field(default_factory=list)
    evaluation_strategy: str = "per_turn"  # per_turn, cumulative, final_only
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiTurnScenario(abc.ABC):
    """
    Abstract base class for multi-turn evaluation scenarios.
    
    Each scenario defines a specific type of multi-turn interaction pattern
    and provides methods for generating prompts, evaluating responses,
    and managing the conversation flow.
    """
    
    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.conversation_history: List[Dict[str, Any]] = []
        self.turn_results: Dict[str, Any] = {}
        self.global_state: Dict[str, Any] = {}
        
    @abc.abstractmethod
    def generate_initial_prompt(self, problem_data: Dict[str, Any]) -> str:
        """Generate the initial prompt for the scenario."""
        pass
        
    @abc.abstractmethod
    def process_turn_response(self, turn_id: str, response: str, 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a response from a specific turn."""
        pass
        
    @abc.abstractmethod
    def generate_next_prompt(self, turn_id: str, 
                           previous_responses: List[str]) -> Optional[str]:
        """Generate the next prompt based on previous responses."""
        pass
        
    @abc.abstractmethod
    def evaluate_scenario(self) -> Dict[str, float]:
        """Evaluate the complete scenario and return scores."""
        pass
        
    def format_chat_history(self, chat_template_fn: Optional[Callable] = None) -> List[Dict[str, str]]:
        """
        Format the conversation history for chat template usage.
        
        Args:
            chat_template_fn: Optional function to apply chat template
            
        Returns:
            List of chat history dictionaries with role and content
        """
        chat_history = []
        
        # Add system message if present
        if self.config.system_message:
            chat_history.append({
                "role": "system",
                "content": self.config.system_message
            })
            
        # Convert conversation history to chat format
        for entry in self.conversation_history:
            if entry.get("role") in ["user", "assistant", "system"]:
                chat_history.append({
                    "role": entry["role"],
                    "content": entry["content"]
                })
                
        return chat_history
        
    def get_turn_config(self, turn_id: str) -> Optional[TurnConfig]:
        """Get configuration for a specific turn."""
        for turn in self.config.turns:
            if turn.turn_id == turn_id:
                return turn
        return None
        
    def add_to_history(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add an entry to the conversation history."""
        entry = {
            "role": role,
            "content": content,
            "timestamp": len(self.conversation_history)
        }
        if metadata:
            entry.update(metadata)
        self.conversation_history.append(entry)
        
    def get_context_for_turn(self, turn_id: str) -> Dict[str, Any]:
        """Get relevant context for a specific turn."""
        turn_config = self.get_turn_config(turn_id)
        if not turn_config:
            return {}
            
        context = {
            "turn_id": turn_id,
            "turn_config": turn_config,
            "global_state": self.global_state,
            "conversation_history": self.conversation_history,
        }
        
        # Add dependent turn results
        if turn_config.depends_on:
            context["dependencies"] = {
                dep_id: self.turn_results.get(dep_id)
                for dep_id in turn_config.depends_on
            }
            
        # Apply context window limitation
        if turn_config.context_window > 0:
            context["conversation_history"] = self.conversation_history[-turn_config.context_window:]
            
        return context
        
    def validate_turn_response(self, turn_id: str, response: str) -> List[str]:
        """
        Validate a turn response against configured rules.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        turn_config = self.get_turn_config(turn_id)
        if not turn_config:
            return ["Turn configuration not found"]
            
        errors = []
        for validation_rule in turn_config.validation_rules:
            try:
                if not validation_rule(response):
                    errors.append(f"Validation rule failed for turn {turn_id}")
            except Exception as e:
                errors.append(f"Validation error: {e}")
                
        return errors
    
    def get_config(self) -> ScenarioConfig:
        """Get the scenario configuration."""
        return self.config


class ConversationalScenario(MultiTurnScenario):
    """
    Implementation for conversational/dialogue scenarios.
    
    This scenario type focuses on natural conversation flow,
    context understanding, and dialogue quality.
    """
    
    def generate_initial_prompt(self, problem_data: Dict[str, Any]) -> str:
        """Generate initial conversational prompt."""
        base_prompt = problem_data.get("initial_query", "Hello! How can I help you today?")
        
        if self.config.system_message:
            return f"System: {self.config.system_message}\n\nUser: {base_prompt}"
        
        return f"User: {base_prompt}"
        
    def process_turn_response(self, turn_id: str, response: str, 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Process conversational turn response."""
        turn_config = self.get_turn_config(turn_id)
        
        # Store the response
        self.add_to_history("assistant", response, {"turn_id": turn_id})
        
        # Basic conversational metrics
        result = {
            "response": response,
            "length": len(response),
            "turn_id": turn_id,
            "coherence_score": self._evaluate_coherence(response, context),
            "relevance_score": self._evaluate_relevance(response, context),
        }
        
        self.turn_results[turn_id] = result
        return result
        
    def generate_next_prompt(self, turn_id: str, 
                           previous_responses: List[str]) -> Optional[str]:
        """Generate next conversational prompt."""
        turn_config = self.get_turn_config(turn_id)
        if not turn_config:
            return None
            
        # Use template with conversation context
        context = {
            "previous_responses": previous_responses,
            "conversation_history": self.conversation_history,
            "global_state": self.global_state
        }
        
        try:
            return turn_config.prompt_template.format(**context)
        except KeyError as e:
            return f"Error in prompt template: {e}"
            
    def evaluate_scenario(self) -> Dict[str, float]:
        """Evaluate conversational scenario."""
        if not self.turn_results:
            return {"overall_score": 0.0}
            
        # Aggregate conversational metrics
        coherence_scores = [r.get("coherence_score", 0) for r in self.turn_results.values()]
        relevance_scores = [r.get("relevance_score", 0) for r in self.turn_results.values()]
        
        return {
            "overall_score": (sum(coherence_scores) + sum(relevance_scores)) / (2 * len(self.turn_results)),
            "average_coherence": sum(coherence_scores) / len(coherence_scores),
            "average_relevance": sum(relevance_scores) / len(relevance_scores),
            "conversation_length": len(self.conversation_history),
        }
        
    def _evaluate_coherence(self, response: str, context: Dict[str, Any]) -> float:
        """Evaluate response coherence (simplified implementation)."""
        # This is a simplified coherence check
        # In practice, you'd use more sophisticated NLP metrics
        if len(response.strip()) == 0:
            return 0.0
        
        # Check if response is relevant to conversation history
        if len(self.conversation_history) > 0:
            prev_content = " ".join([h.get("content", "") for h in self.conversation_history[-3:]])
            # Simple overlap-based coherence (would use better metrics in practice)
            common_words = set(response.lower().split()) & set(prev_content.lower().split())
            return min(1.0, len(common_words) / 10)  # Normalize to 0-1
            
        return 0.8  # Default score for first response
        
    def _evaluate_relevance(self, response: str, context: Dict[str, Any]) -> float:
        """Evaluate response relevance (simplified implementation)."""
        # Simplified relevance check
        if len(response.strip()) == 0:
            return 0.0
            
        # Check if response addresses the query (basic implementation)
        if len(response) > 20 and not response.lower().startswith(("i don't", "i can't", "sorry")):
            return 0.9
        elif len(response) > 10:
            return 0.6
        else:
            return 0.3