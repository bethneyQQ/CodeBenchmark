"""
Multi-Turn Scenario Metrics.

This module provides comprehensive metrics for evaluating different types
of multi-turn scenarios with scenario-specific evaluation logic.
"""

import re
import statistics
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict

from .base_scenario import ScenarioConfig, ScenarioType


class MultiTurnMetrics:
    """
    Comprehensive metrics calculator for multi-turn scenarios.
    
    Provides both generic and scenario-specific metrics.
    """
    
    def __init__(self):
        self.scenario_specific_metrics = {
            ScenarioType.CONVERSATIONAL: self._calculate_conversational_metrics,
            ScenarioType.WORKFLOW: self._calculate_workflow_metrics,
            ScenarioType.ITERATIVE: self._calculate_iterative_metrics,
            ScenarioType.COLLABORATIVE: self._calculate_collaborative_metrics,
            ScenarioType.INSTRUCTIONAL: self._calculate_instructional_metrics,
            ScenarioType.CODE_REVIEW: self._calculate_code_review_metrics,
            ScenarioType.DEBUG_SESSION: self._calculate_debug_metrics,
        }
        
    def calculate_aggregated_metrics(self,
                                   turn_results: Dict[str, Any],
                                   conversation_history: List[Dict[str, Any]],
                                   config: ScenarioConfig) -> Dict[str, float]:
        """
        Calculate aggregated metrics for a complete scenario.
        
        Args:
            turn_results: Results from individual turns
            conversation_history: Complete conversation history
            config: Scenario configuration
            
        Returns:
            Dictionary of aggregated metrics
        """
        # Generic metrics
        generic_metrics = self._calculate_generic_metrics(turn_results, conversation_history)
        
        # Scenario-specific metrics
        specific_metrics = {}
        if config.scenario_type in self.scenario_specific_metrics:
            specific_calculator = self.scenario_specific_metrics[config.scenario_type]
            specific_metrics = specific_calculator(turn_results, conversation_history, config)
            
        # Quality metrics
        quality_metrics = self._calculate_quality_metrics(turn_results, conversation_history)
        
        # Coherence and flow metrics
        flow_metrics = self._calculate_flow_metrics(conversation_history)
        
        # Combine all metrics
        all_metrics = {
            **generic_metrics,
            **specific_metrics,
            **quality_metrics,
            **flow_metrics
        }
        
        # Calculate overall score
        all_metrics["overall_multi_turn_score"] = self._calculate_overall_score(all_metrics, config)
        
        return all_metrics
        
    def _calculate_generic_metrics(self,
                                 turn_results: Dict[str, Any],
                                 conversation_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate generic multi-turn metrics."""
        if not turn_results:
            return {"turn_completion_rate": 0.0, "average_response_length": 0.0}
            
        metrics = {}
        
        # Turn completion rate
        completed_turns = sum(1 for result in turn_results.values() if result.get("response"))
        total_turns = len(turn_results)
        metrics["turn_completion_rate"] = completed_turns / total_turns if total_turns > 0 else 0.0
        
        # Response length statistics
        response_lengths = [
            len(result.get("response", "")) 
            for result in turn_results.values()
            if result.get("response")
        ]
        
        if response_lengths:
            metrics["average_response_length"] = statistics.mean(response_lengths)
            metrics["response_length_std"] = statistics.stdev(response_lengths) if len(response_lengths) > 1 else 0.0
            metrics["min_response_length"] = min(response_lengths)
            metrics["max_response_length"] = max(response_lengths)
        else:
            metrics.update({
                "average_response_length": 0.0,
                "response_length_std": 0.0,
                "min_response_length": 0.0,
                "max_response_length": 0.0
            })
            
        # Turn success rate (based on individual turn scores if available)
        success_scores = []
        for result in turn_results.values():
            # Look for any score-like metrics in turn results
            for key, value in result.items():
                if "score" in key.lower() and isinstance(value, (int, float)):
                    success_scores.append(value)
                    break
                    
        metrics["average_turn_success"] = statistics.mean(success_scores) if success_scores else 0.5
        
        return metrics
        
    def _calculate_quality_metrics(self,
                                 turn_results: Dict[str, Any],
                                 conversation_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate response quality metrics."""
        if not turn_results:
            return {"response_quality_score": 0.0}
            
        quality_metrics = {}
        
        # Response completeness
        completeness_scores = []
        for result in turn_results.values():
            response = result.get("response", "")
            completeness_scores.append(self._calculate_response_completeness(response))
            
        quality_metrics["average_response_completeness"] = (
            statistics.mean(completeness_scores) if completeness_scores else 0.0
        )
        
        # Response relevance (simplified)
        relevance_scores = []
        for i, entry in enumerate(conversation_history):
            if entry.get("role") == "assistant":
                relevance = self._calculate_response_relevance(entry.get("content", ""), conversation_history[:i])
                relevance_scores.append(relevance)
                
        quality_metrics["average_response_relevance"] = (
            statistics.mean(relevance_scores) if relevance_scores else 0.0
        )
        
        # Information density
        info_density_scores = []
        for result in turn_results.values():
            response = result.get("response", "")
            info_density_scores.append(self._calculate_information_density(response))
            
        quality_metrics["average_information_density"] = (
            statistics.mean(info_density_scores) if info_density_scores else 0.0
        )
        
        # Overall quality score
        quality_components = [
            quality_metrics["average_response_completeness"],
            quality_metrics["average_response_relevance"],
            quality_metrics["average_information_density"]
        ]
        
        quality_metrics["response_quality_score"] = statistics.mean(quality_components)
        
        return quality_metrics
        
    def _calculate_flow_metrics(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate conversation flow and coherence metrics."""
        if len(conversation_history) < 2:
            return {"conversation_coherence": 0.5, "turn_transition_quality": 0.5}
            
        flow_metrics = {}
        
        # Conversation coherence
        coherence_scores = []
        for i in range(1, len(conversation_history)):
            prev_content = conversation_history[i-1].get("content", "")
            curr_content = conversation_history[i].get("content", "")
            coherence_scores.append(self._calculate_turn_coherence(prev_content, curr_content))
            
        flow_metrics["conversation_coherence"] = (
            statistics.mean(coherence_scores) if coherence_scores else 0.5
        )
        
        # Turn transition quality
        transition_scores = []
        assistant_turns = [
            entry for entry in conversation_history
            if entry.get("role") == "assistant"
        ]
        
        for i in range(1, len(assistant_turns)):
            prev_response = assistant_turns[i-1].get("content", "")
            curr_response = assistant_turns[i].get("content", "")
            transition_scores.append(self._calculate_transition_quality(prev_response, curr_response))
            
        flow_metrics["turn_transition_quality"] = (
            statistics.mean(transition_scores) if transition_scores else 0.5
        )
        
        # Context maintenance
        flow_metrics["context_maintenance_score"] = self._calculate_context_maintenance(conversation_history)
        
        return flow_metrics
        
    def _calculate_conversational_metrics(self,
                                        turn_results: Dict[str, Any],
                                        conversation_history: List[Dict[str, Any]],
                                        config: ScenarioConfig) -> Dict[str, float]:
        """Calculate conversational scenario specific metrics."""
        metrics = {}
        
        # Dialogue naturalness
        assistant_responses = [
            entry.get("content", "") for entry in conversation_history
            if entry.get("role") == "assistant"
        ]
        
        naturalness_scores = [
            self._calculate_naturalness(response) for response in assistant_responses
        ]
        
        metrics["dialogue_naturalness"] = (
            statistics.mean(naturalness_scores) if naturalness_scores else 0.0
        )
        
        # Question answering quality
        qa_pairs = self._extract_qa_pairs(conversation_history)
        qa_scores = [self._evaluate_qa_quality(q, a) for q, a in qa_pairs]
        
        metrics["qa_quality"] = statistics.mean(qa_scores) if qa_scores else 0.0
        
        # Engagement level
        engagement_indicators = sum([
            "?" in response for response in assistant_responses
        ])
        metrics["engagement_level"] = min(engagement_indicators / len(assistant_responses), 1.0) if assistant_responses else 0.0
        
        return metrics
        
    def _calculate_workflow_metrics(self,
                                  turn_results: Dict[str, Any],
                                  conversation_history: List[Dict[str, Any]],
                                  config: ScenarioConfig) -> Dict[str, float]:
        """Calculate workflow scenario specific metrics."""
        metrics = {}
        
        # Step completion tracking
        completed_steps = 0
        total_steps = len(config.turns)
        
        for turn_id in [turn.turn_id for turn in config.turns]:
            if turn_id in turn_results and turn_results[turn_id].get("response"):
                completed_steps += 1
                
        metrics["workflow_completion_rate"] = completed_steps / total_steps if total_steps > 0 else 0.0
        
        # Sequential consistency
        metrics["sequential_consistency"] = self._calculate_sequential_consistency(turn_results, config)
        
        # Dependency satisfaction
        metrics["dependency_satisfaction"] = self._calculate_dependency_satisfaction(turn_results, config)
        
        return metrics
        
    def _calculate_iterative_metrics(self,
                                   turn_results: Dict[str, Any],
                                   conversation_history: List[Dict[str, Any]],
                                   config: ScenarioConfig) -> Dict[str, float]:
        """Calculate iterative scenario specific metrics."""
        metrics = {}
        
        # Improvement trajectory
        improvement_scores = []
        for i, (turn_id, result) in enumerate(turn_results.items()):
            if "improvement" in str(result).lower():
                # Extract improvement-related scores
                for key, value in result.items():
                    if "improvement" in key.lower() and isinstance(value, (int, float)):
                        improvement_scores.append(value)
                        break
                        
        if len(improvement_scores) > 1:
            # Calculate trend
            trend = sum(improvement_scores[i] - improvement_scores[i-1] 
                       for i in range(1, len(improvement_scores))) / (len(improvement_scores) - 1)
            metrics["improvement_trend"] = max(0, min(1, trend + 0.5))  # Normalize to 0-1
        else:
            metrics["improvement_trend"] = 0.5
            
        # Convergence detection
        metrics["convergence_detected"] = self._detect_convergence(turn_results)
        
        # Iteration efficiency
        metrics["iteration_efficiency"] = self._calculate_iteration_efficiency(turn_results)
        
        return metrics
        
    def _calculate_collaborative_metrics(self,
                                       turn_results: Dict[str, Any],
                                       conversation_history: List[Dict[str, Any]],
                                       config: ScenarioConfig) -> Dict[str, float]:
        """Calculate collaborative scenario specific metrics."""
        # Placeholder for collaborative metrics
        return {
            "collaboration_quality": 0.5,
            "contribution_balance": 0.5,
            "consensus_building": 0.5
        }
        
    def _calculate_instructional_metrics(self,
                                       turn_results: Dict[str, Any],
                                       conversation_history: List[Dict[str, Any]],
                                       config: ScenarioConfig) -> Dict[str, float]:
        """Calculate instructional scenario specific metrics."""
        metrics = {}
        
        # Pedagogical effectiveness
        pedagogical_scores = []
        for result in turn_results.values():
            for key, value in result.items():
                if "pedagogical" in key.lower() and isinstance(value, (int, float)):
                    pedagogical_scores.append(value)
                    
        metrics["pedagogical_effectiveness"] = (
            statistics.mean(pedagogical_scores) if pedagogical_scores else 0.0
        )
        
        # Learning progression
        learning_indicators = sum([
            1 for entry in conversation_history
            if entry.get("role") == "user" and "?" in entry.get("content", "")
        ])
        
        metrics["student_engagement"] = min(learning_indicators / 5.0, 1.0)  # Normalize
        
        return metrics
        
    def _calculate_code_review_metrics(self,
                                     turn_results: Dict[str, Any],
                                     conversation_history: List[Dict[str, Any]],
                                     config: ScenarioConfig) -> Dict[str, float]:
        """Calculate code review scenario specific metrics."""
        metrics = {}
        
        # Review thoroughness
        review_completeness = 0
        for result in turn_results.values():
            if "review_quality" in result:
                review_completeness = max(review_completeness, result["review_quality"])
                
        metrics["review_thoroughness"] = review_completeness
        
        # Code improvement detection
        code_blocks = sum([
            entry.get("content", "").count("```") for entry in conversation_history
        ])
        metrics["code_iteration_count"] = code_blocks // 2  # Pairs of code block markers
        
        return metrics
        
    def _calculate_debug_metrics(self,
                               turn_results: Dict[str, Any],
                               conversation_history: List[Dict[str, Any]],
                               config: ScenarioConfig) -> Dict[str, float]:
        """Calculate debug session specific metrics."""
        # Placeholder for debug metrics
        return {
            "problem_identification": 0.5,
            "solution_effectiveness": 0.5,
            "debug_methodology": 0.5
        }
        
    def _calculate_overall_score(self, metrics: Dict[str, float], config: ScenarioConfig) -> float:
        """Calculate overall multi-turn score based on scenario type."""
        
        # Base components that apply to all scenarios
        base_components = [
            metrics.get("turn_completion_rate", 0.5),
            metrics.get("response_quality_score", 0.5),
            metrics.get("conversation_coherence", 0.5)
        ]
        
        # Add scenario-specific components
        scenario_components = []
        
        if config.scenario_type == ScenarioType.CONVERSATIONAL:
            scenario_components.extend([
                metrics.get("dialogue_naturalness", 0.5),
                metrics.get("qa_quality", 0.5)
            ])
        elif config.scenario_type == ScenarioType.WORKFLOW:
            scenario_components.extend([
                metrics.get("workflow_completion_rate", 0.5),
                metrics.get("sequential_consistency", 0.5)
            ])
        elif config.scenario_type == ScenarioType.ITERATIVE:
            scenario_components.extend([
                metrics.get("improvement_trend", 0.5),
                metrics.get("iteration_efficiency", 0.5)
            ])
        elif config.scenario_type == ScenarioType.INSTRUCTIONAL:
            scenario_components.extend([
                metrics.get("pedagogical_effectiveness", 0.5),
                metrics.get("student_engagement", 0.5)
            ])
            
        # Combine all components
        all_components = base_components + scenario_components
        return statistics.mean(all_components) if all_components else 0.5
        
    # Helper methods for metric calculations
    
    def _calculate_response_completeness(self, response: str) -> float:
        """Calculate completeness of a response."""
        if not response.strip():
            return 0.0
            
        completeness_indicators = [
            len(response) > 50,  # Sufficient length
            response.count('.') > 1,  # Multiple sentences
            any(word in response.lower() for word in ['because', 'since', 'therefore']),  # Reasoning
            len(response.split()) > 20,  # Sufficient word count
        ]
        
        return sum(completeness_indicators) / len(completeness_indicators)
        
    def _calculate_response_relevance(self, response: str, prior_context: List[Dict[str, Any]]) -> float:
        """Calculate relevance of response to prior context."""
        if not prior_context or not response.strip():
            return 0.5
            
        # Simple relevance based on word overlap with recent context
        recent_content = " ".join([
            entry.get("content", "") for entry in prior_context[-3:]
        ]).lower()
        
        response_words = set(response.lower().split())
        context_words = set(recent_content.split())
        
        if not context_words:
            return 0.5
            
        overlap = len(response_words & context_words)
        return min(overlap / 20.0, 1.0)  # Normalize to 0-1
        
    def _calculate_information_density(self, response: str) -> float:
        """Calculate information density of response."""
        if not response.strip():
            return 0.0
            
        # Simple heuristic based on content characteristics
        density_indicators = [
            len(set(response.lower().split())) / max(len(response.split()), 1),  # Unique word ratio
            response.count(',') > 2,  # List-like content
            response.count(':') > 0,  # Definitions or explanations
            len(re.findall(r'\d+', response)) > 0,  # Numbers/data
        ]
        
        numeric_score = density_indicators[0]  # Unique word ratio
        categorical_score = sum(density_indicators[1:]) / 3
        
        return (numeric_score + categorical_score) / 2
        
    def _calculate_turn_coherence(self, prev_content: str, curr_content: str) -> float:
        """Calculate coherence between consecutive turns."""
        if not prev_content.strip() or not curr_content.strip():
            return 0.5
            
        # Simple coherence based on topic continuity
        prev_words = set(prev_content.lower().split())
        curr_words = set(curr_content.lower().split())
        
        # Check for topic continuity
        overlap = len(prev_words & curr_words)
        coherence = min(overlap / 15.0, 1.0)  # Normalize
        
        # Boost score if there are explicit connection words
        connection_words = ['however', 'therefore', 'moreover', 'furthermore', 'additionally']
        if any(word in curr_content.lower() for word in connection_words):
            coherence = min(coherence + 0.2, 1.0)
            
        return coherence
        
    def _calculate_transition_quality(self, prev_response: str, curr_response: str) -> float:
        """Calculate quality of transition between responses."""
        if not prev_response.strip() or not curr_response.strip():
            return 0.5
            
        # Check for smooth transitions
        transition_indicators = [
            curr_response.lower().startswith(('yes', 'no', 'indeed', 'however')),
            'previous' in curr_response.lower() or 'earlier' in curr_response.lower(),
            len(curr_response) > len(prev_response) * 0.5,  # Reasonable length progression
        ]
        
        return sum(transition_indicators) / len(transition_indicators)
        
    def _calculate_context_maintenance(self, conversation_history: List[Dict[str, Any]]) -> float:
        """Calculate how well context is maintained throughout conversation."""
        if len(conversation_history) < 3:
            return 0.5
            
        # Track topic consistency across the conversation
        all_content = " ".join([
            entry.get("content", "") for entry in conversation_history
        ]).lower()
        
        # Extract key terms from the conversation
        words = all_content.split()
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 4:  # Focus on meaningful words
                word_freq[word] += 1
                
        # Check if key terms are maintained throughout
        total_words = len(words)
        repeated_words = sum(1 for freq in word_freq.values() if freq > 1)
        
        if total_words == 0:
            return 0.0
            
        return min(repeated_words / (total_words * 0.1), 1.0)  # Normalize
        
    def _calculate_naturalness(self, response: str) -> float:
        """Calculate naturalness of response."""
        if not response.strip():
            return 0.0
            
        naturalness_indicators = [
            not response.startswith('1.') and not response.startswith('Step'),  # Not overly structured
            any(word in response.lower() for word in ['i', 'you', 'we', 'me']),  # Personal pronouns
            response.count('!') + response.count('?') > 0,  # Natural punctuation
            len(response.split()) > 5,  # Reasonable length
            not all(line.strip().startswith('-') for line in response.split('\n') if line.strip())  # Not all bullet points
        ]
        
        return sum(naturalness_indicators) / len(naturalness_indicators)
        
    def _extract_qa_pairs(self, conversation_history: List[Dict[str, Any]]) -> List[tuple]:
        """Extract question-answer pairs from conversation."""
        qa_pairs = []
        
        for i in range(len(conversation_history) - 1):
            curr_entry = conversation_history[i]
            next_entry = conversation_history[i + 1]
            
            if (curr_entry.get("role") == "user" and 
                next_entry.get("role") == "assistant" and
                "?" in curr_entry.get("content", "")):
                
                qa_pairs.append((curr_entry.get("content", ""), next_entry.get("content", "")))
                
        return qa_pairs
        
    def _evaluate_qa_quality(self, question: str, answer: str) -> float:
        """Evaluate quality of question-answer pair."""
        if not question.strip() or not answer.strip():
            return 0.0
            
        quality_indicators = [
            len(answer) > len(question) * 0.5,  # Reasonable answer length
            not answer.lower().startswith("i don't"),  # Not a rejection
            len(answer.split()) > 10,  # Sufficient detail
            any(word in answer.lower() for word in question.lower().split()[-5:])  # Addresses question terms
        ]
        
        return sum(quality_indicators) / len(quality_indicators)
        
    def _calculate_sequential_consistency(self, turn_results: Dict[str, Any], config: ScenarioConfig) -> float:
        """Calculate consistency with expected sequential flow."""
        if not config.turns:
            return 0.5
            
        expected_order = [turn.turn_id for turn in config.turns]
        actual_order = list(turn_results.keys())
        
        # Calculate order similarity
        matches = sum(1 for i, turn_id in enumerate(expected_order) 
                     if i < len(actual_order) and actual_order[i] == turn_id)
        
        return matches / len(expected_order) if expected_order else 0.5
        
    def _calculate_dependency_satisfaction(self, turn_results: Dict[str, Any], config: ScenarioConfig) -> float:
        """Calculate how well turn dependencies are satisfied."""
        satisfied_deps = 0
        total_deps = 0
        
        for turn in config.turns:
            if turn.depends_on:
                total_deps += len(turn.depends_on)
                for dep in turn.depends_on:
                    if dep in turn_results:
                        satisfied_deps += 1
                        
        return satisfied_deps / total_deps if total_deps > 0 else 1.0
        
    def _detect_convergence(self, turn_results: Dict[str, Any]) -> float:
        """Detect if iterative process has converged."""
        # Look for convergence indicators in turn results
        convergence_indicators = []
        
        for result in turn_results.values():
            response = result.get("response", "")
            if any(word in response.lower() for word in ['final', 'complete', 'done', 'finished']):
                convergence_indicators.append(1.0)
            elif any(word in response.lower() for word in ['improve', 'better', 'refine']):
                convergence_indicators.append(0.3)
            else:
                convergence_indicators.append(0.0)
                
        return statistics.mean(convergence_indicators) if convergence_indicators else 0.0
        
    def _calculate_iteration_efficiency(self, turn_results: Dict[str, Any]) -> float:
        """Calculate efficiency of iterative process."""
        if not turn_results:
            return 0.0
            
        # Simple heuristic: fewer turns with higher quality is more efficient
        avg_quality = sum(
            result.get("quality", 0.5) for result in turn_results.values()
        ) / len(turn_results)
        
        # Penalize excessive turns
        turn_penalty = max(0, len(turn_results) - 5) * 0.1
        efficiency = max(0, avg_quality - turn_penalty)
        
        return min(efficiency, 1.0)
    
    def calculate_scenario_metrics(self, 
                                  scenario_result: Dict[str, Any], 
                                  scenario_type: ScenarioType) -> Dict[str, float]:
        """
        Calculate metrics for a specific scenario result.
        
        Args:
            scenario_result: Complete scenario evaluation result
            scenario_type: Type of scenario being evaluated
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Extract turn results
        turn_results = {}
        if 'turns' in scenario_result:
            for i, turn in enumerate(scenario_result['turns']):
                turn_results[turn.get('turn_id', f'turn_{i}')] = turn
        
        # Create mock conversation history from turns
        conversation_history = []
        for turn in scenario_result.get('turns', []):
            conversation_history.append({
                'turn_id': turn.get('turn_id', ''),
                'response': turn.get('response', ''),
                'role': 'assistant'  # Assume assistant responses for testing
            })
        
        # Calculate generic metrics
        generic_metrics = self._calculate_generic_metrics(turn_results, conversation_history)
        metrics.update(generic_metrics)
        
        # Create mock config for scenario-specific metrics
        mock_config = ScenarioConfig(
            scenario_id=scenario_result.get('scenario_id', 'test'),
            scenario_type=scenario_type,
            name="Test Scenario",
            description="Test scenario for metrics calculation",
            turns=[],
            chat_template_required=False,
            system_message=""
        )
        
        # Calculate scenario-specific metrics if handler exists
        if scenario_type in self.scenario_specific_metrics:
            specific_metrics = self.scenario_specific_metrics[scenario_type](
                turn_results, conversation_history, mock_config)
            metrics.update(specific_metrics)
        else:
            # Fallback to conversational metrics
            specific_metrics = self._calculate_conversational_metrics(
                turn_results, conversation_history, mock_config)
            metrics.update(specific_metrics)
            
        return metrics