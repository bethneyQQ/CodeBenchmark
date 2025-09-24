"""
Concrete Multi-Turn Scenario Implementations.

This module provides specific implementations of various multi-turn scenarios
for different evaluation use cases.
"""

import re
import json
from typing import Dict, List, Any, Optional
from .base_scenario import (
    MultiTurnScenario, TurnConfig, ScenarioConfig, 
    TurnType, ScenarioType, ConversationalScenario
)
from .scenario_registry import register_scenario


# ==================== Code Review Scenario ====================

@register_scenario("code_review")
class CodeReviewScenario(MultiTurnScenario):
    """
    Multi-turn code review scenario.
    
    Simulates a code review process with initial code submission,
    reviewer feedback, code revision, and final evaluation.
    """
    
    def __init__(self, config: ScenarioConfig):
        super().__init__(config)
        self.code_versions = []
        self.review_comments = []
        
    def generate_initial_prompt(self, problem_data: Dict[str, Any]) -> str:
        """Generate initial code review prompt."""
        code_to_review = problem_data.get("code", "")
        context = problem_data.get("context", "Please review the following code")
        
        prompt = f"""# Code Review Session

**Context**: {context}

**Code to Review**:
```python
{code_to_review}
```

Please perform a thorough code review. Consider:
1. Code quality and style
2. Functionality and correctness
3. Performance implications
4. Security considerations
5. Maintainability

Provide specific, actionable feedback with suggestions for improvement."""
        
        self.add_to_history("user", prompt)
        return prompt
        
    def process_turn_response(self, turn_id: str, response: str, 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Process code review turn response."""
        if turn_id == "initial_review":
            # Process initial review
            self.review_comments.append(response)
            self.add_to_history("assistant", response, {"turn_id": turn_id})
            
            result = {
                "response": response,
                "review_quality": self._evaluate_review_quality(response),
                "specificity_score": self._evaluate_specificity(response),
                "actionability_score": self._evaluate_actionability(response),
            }
            
        elif turn_id == "code_revision":
            # Process code revision
            revised_code = self._extract_code_from_response(response)
            if revised_code:
                self.code_versions.append(revised_code)
                
            self.add_to_history("assistant", response, {"turn_id": turn_id})
            
            result = {
                "response": response,
                "revision_quality": self._evaluate_revision_quality(response, context),
                "addresses_feedback": self._evaluate_feedback_addressing(response),
                "code_improvement": self._evaluate_code_improvement(),
            }
            
        else:
            # Default processing
            self.add_to_history("assistant", response, {"turn_id": turn_id})
            result = {"response": response}
            
        self.turn_results[turn_id] = result
        return result
        
    def generate_next_prompt(self, turn_id: str, 
                           previous_responses: List[str]) -> Optional[str]:
        """Generate next code review prompt."""
        if turn_id == "code_revision" and self.review_comments:
            # Generate revision prompt based on review comments
            latest_review = self.review_comments[-1]
            
            prompt = f"""Based on the code review feedback:

{latest_review}

Please provide an updated version of the code that addresses the review comments. 
Include a brief explanation of the changes made and how they address the feedback."""
            
            self.add_to_history("user", prompt)
            return prompt
            
        elif turn_id == "final_evaluation" and len(self.code_versions) > 1:
            # Generate final evaluation prompt
            prompt = f"""Please provide a final evaluation comparing the original code with the revised version.

Assess:
1. How well the revision addresses the original feedback
2. Overall improvement in code quality
3. Any remaining issues or suggestions
4. Overall rating (1-10) for the revision process"""

            self.add_to_history("user", prompt)
            return prompt
            
        return None
        
    def evaluate_scenario(self) -> Dict[str, float]:
        """Evaluate complete code review scenario."""
        if not self.turn_results:
            return {"overall_score": 0.0}
            
        # Aggregate review metrics
        review_scores = []
        revision_scores = []
        
        for turn_id, result in self.turn_results.items():
            if "review_quality" in result:
                review_scores.append(result["review_quality"])
            if "revision_quality" in result:
                revision_scores.append(result["revision_quality"])
                
        metrics = {
            "overall_score": 0.0,
            "review_completeness": len(self.review_comments) > 0,
            "revision_completeness": len(self.code_versions) > 1,
            "total_turns": len(self.conversation_history),
        }
        
        if review_scores:
            metrics["average_review_quality"] = sum(review_scores) / len(review_scores)
            
        if revision_scores:
            metrics["average_revision_quality"] = sum(revision_scores) / len(revision_scores)
            
        # Calculate overall score
        components = [
            metrics.get("average_review_quality", 0),
            metrics.get("average_revision_quality", 0),
            0.8 if metrics["review_completeness"] else 0,
            0.9 if metrics["revision_completeness"] else 0,
        ]
        
        metrics["overall_score"] = sum(components) / len(components)
        return metrics
        
    def _evaluate_review_quality(self, review: str) -> float:
        """Evaluate quality of code review."""
        score = 0.0
        review_lower = review.lower()
        
        # Check for key review aspects
        aspects = [
            ("functionality", ["function", "logic", "correct", "bug", "error"]),
            ("style", ["style", "formatting", "convention", "pep", "naming"]),
            ("performance", ["performance", "efficiency", "optimize", "slow"]),
            ("security", ["security", "vulnerability", "safe", "validate"]),
            ("maintainability", ["maintain", "readable", "clean", "document"])
        ]
        
        for aspect, keywords in aspects:
            if any(keyword in review_lower for keyword in keywords):
                score += 0.2
                
        return min(score, 1.0)
        
    def _evaluate_specificity(self, review: str) -> float:
        """Evaluate specificity of review comments."""
        # Count specific line references, code snippets, examples
        line_refs = len(re.findall(r"line \d+|line\d+|:\d+", review, re.IGNORECASE))
        code_blocks = len(re.findall(r"```|`[^`]+`", review))
        specific_suggestions = len(re.findall(r"should|could|suggest|recommend|try", review, re.IGNORECASE))
        
        specificity = (line_refs * 0.3 + code_blocks * 0.4 + specific_suggestions * 0.1)
        return min(specificity, 1.0)
        
    def _evaluate_actionability(self, review: str) -> float:
        """Evaluate how actionable the review feedback is."""
        actionable_words = ["change", "replace", "add", "remove", "modify", "fix", "implement"]
        count = sum(1 for word in actionable_words if word in review.lower())
        return min(count / 10.0, 1.0)
        
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract code blocks from response."""
        code_blocks = re.findall(r"```(?:python)?\n?(.*?)\n?```", response, re.DOTALL)
        return code_blocks[0].strip() if code_blocks else None
        
    def _evaluate_revision_quality(self, response: str, context: Dict[str, Any]) -> float:
        """Evaluate quality of code revision."""
        # Simple heuristic based on response characteristics
        has_code = "```" in response
        has_explanation = len(response.split()) > 20
        addresses_feedback = "review" in response.lower() or "feedback" in response.lower()
        
        score = 0.3 if has_code else 0
        score += 0.3 if has_explanation else 0
        score += 0.4 if addresses_feedback else 0
        
        return score
        
    def _evaluate_feedback_addressing(self, response: str) -> float:
        """Evaluate how well the revision addresses feedback."""
        # Look for references to original review points
        if self.review_comments:
            last_review = self.review_comments[-1].lower()
            response_lower = response.lower()
            
            # Count overlapping concepts
            review_words = set(last_review.split())
            response_words = set(response_lower.split())
            overlap = len(review_words & response_words)
            
            return min(overlap / 20.0, 1.0)
        return 0.0
        
    def _evaluate_code_improvement(self) -> float:
        """Evaluate improvement between code versions."""
        if len(self.code_versions) < 2:
            return 0.0
            
        # Simple heuristic: longer, more structured code is generally better
        original = self.code_versions[0]
        revised = self.code_versions[-1]
        
        improvement_factors = [
            len(revised) > len(original),  # More comprehensive
            revised.count('\n') > original.count('\n'),  # More structured
            revised.count('#') > original.count('#'),  # More comments
            'def ' in revised and 'def ' not in original,  # Added functions
        ]
        
        return sum(improvement_factors) / len(improvement_factors)


# ==================== Iterative Problem Solving Scenario ====================

@register_scenario("iterative_problem_solving")
class IterativeProblemSolvingScenario(MultiTurnScenario):
    """
    Multi-turn iterative problem solving scenario.
    
    Simulates a problem-solving session with initial approach,
    feedback/testing, refinement, and convergence to solution.
    """
    
    def __init__(self, config: ScenarioConfig):
        super().__init__(config)
        self.solution_attempts = []
        self.feedback_history = []
        
    def generate_initial_prompt(self, problem_data: Dict[str, Any]) -> str:
        """Generate initial problem-solving prompt."""
        problem = problem_data.get("problem", "")
        constraints = problem_data.get("constraints", [])
        examples = problem_data.get("examples", [])
        
        prompt = f"""# Problem Solving Session

**Problem**: {problem}

"""
        
        if constraints:
            prompt += "**Constraints**:\n"
            for constraint in constraints:
                prompt += f"- {constraint}\n"
            prompt += "\n"
            
        if examples:
            prompt += "**Examples**:\n"
            for i, example in enumerate(examples, 1):
                prompt += f"Example {i}: {example}\n"
            prompt += "\n"
            
        prompt += """Please provide your initial approach to solving this problem. 
Include:
1. Your understanding of the problem
2. Your proposed solution strategy
3. Step-by-step implementation plan
4. Any assumptions you're making"""
        
        self.add_to_history("user", prompt)
        return prompt
        
    def process_turn_response(self, turn_id: str, response: str, 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Process problem solving turn response."""
        self.add_to_history("assistant", response, {"turn_id": turn_id})
        
        if turn_id.startswith("solution_"):
            # Process solution attempt
            solution = self._extract_solution_from_response(response)
            if solution:
                self.solution_attempts.append(solution)
                
            result = {
                "response": response,
                "solution_completeness": self._evaluate_solution_completeness(response),
                "approach_clarity": self._evaluate_approach_clarity(response),
                "implementation_detail": self._evaluate_implementation_detail(response),
            }
            
        elif turn_id.startswith("refinement_"):
            # Process refinement attempt
            result = {
                "response": response,
                "addresses_feedback": self._evaluate_feedback_addressing(response),
                "improvement_quality": self._evaluate_improvement_quality(response),
                "convergence_progress": self._evaluate_convergence_progress(),
            }
            
        else:
            result = {"response": response}
            
        self.turn_results[turn_id] = result
        return result
        
    def generate_next_prompt(self, turn_id: str, 
                           previous_responses: List[str]) -> Optional[str]:
        """Generate next problem solving prompt."""
        if turn_id.startswith("feedback_"):
            # Generate feedback based on previous solution
            if self.solution_attempts:
                latest_solution = self.solution_attempts[-1]
                feedback = self._generate_feedback(latest_solution)
                self.feedback_history.append(feedback)
                
                prompt = f"""Here's feedback on your solution:

{feedback}

Please refine your approach based on this feedback. Provide an improved solution."""
                
                self.add_to_history("user", prompt)
                return prompt
                
        elif turn_id.startswith("test_"):
            # Generate test scenario
            prompt = """Let's test your solution with some edge cases:

1. What happens with empty input?
2. How does it handle very large inputs?
3. Are there any boundary conditions to consider?

Please walk through how your solution handles these cases."""
            
            self.add_to_history("user", prompt)
            return prompt
            
        return None
        
    def evaluate_scenario(self) -> Dict[str, float]:
        """Evaluate complete problem solving scenario."""
        if not self.turn_results:
            return {"overall_score": 0.0}
            
        # Track improvement over iterations
        solution_scores = []
        refinement_scores = []
        
        for turn_id, result in self.turn_results.items():
            if "solution_completeness" in result:
                solution_scores.append(result["solution_completeness"])
            if "improvement_quality" in result:
                refinement_scores.append(result["improvement_quality"])
                
        metrics = {
            "overall_score": 0.0,
            "total_iterations": len(self.solution_attempts),
            "improvement_trajectory": self._calculate_improvement_trajectory(solution_scores),
            "convergence_achieved": len(solution_scores) > 0 and solution_scores[-1] > 0.8,
        }
        
        if solution_scores:
            metrics["best_solution_score"] = max(solution_scores)
            metrics["average_solution_score"] = sum(solution_scores) / len(solution_scores)
            
        if refinement_scores:
            metrics["average_refinement_quality"] = sum(refinement_scores) / len(refinement_scores)
            
        # Calculate overall score
        components = [
            metrics.get("best_solution_score", 0),
            metrics.get("improvement_trajectory", 0),
            1.0 if metrics["convergence_achieved"] else 0.5,
            min(metrics["total_iterations"] / 3.0, 1.0),  # Reward multiple iterations up to a point
        ]
        
        metrics["overall_score"] = sum(components) / len(components)
        return metrics
        
    def _extract_solution_from_response(self, response: str) -> Optional[str]:
        """Extract solution from response."""
        # Look for code blocks or structured solutions
        code_blocks = re.findall(r"```(?:python)?\n?(.*?)\n?```", response, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
            
        # Look for step-by-step solutions
        steps = re.findall(r"\d+\.\s*(.+)", response)
        if steps:
            return "\n".join(steps)
            
        return response  # Fallback to entire response
        
    def _evaluate_solution_completeness(self, response: str) -> float:
        """Evaluate completeness of solution."""
        completeness_indicators = [
            "step" in response.lower(),
            "algorithm" in response.lower() or "approach" in response.lower(),
            len(response.split()) > 50,  # Sufficient detail
            "```" in response,  # Has code
            any(word in response.lower() for word in ["input", "output", "return"])
        ]
        
        return sum(completeness_indicators) / len(completeness_indicators)
        
    def _evaluate_approach_clarity(self, response: str) -> float:
        """Evaluate clarity of approach."""
        clarity_indicators = [
            response.count(".") > 3,  # Multiple sentences
            "first" in response.lower() or "then" in response.lower(),  # Sequential steps
            "because" in response.lower() or "since" in response.lower(),  # Reasoning
            len(re.findall(r"\d+\.", response)) > 0,  # Numbered points
        ]
        
        return sum(clarity_indicators) / len(clarity_indicators)
        
    def _evaluate_implementation_detail(self, response: str) -> float:
        """Evaluate level of implementation detail."""
        detail_indicators = [
            "```" in response,  # Code blocks
            len(re.findall(r"def |class |for |while |if ", response)) > 0,  # Code structures
            "variable" in response.lower() or "function" in response.lower(),
            "complexity" in response.lower() or "time" in response.lower(),
        ]
        
        return sum(detail_indicators) / len(detail_indicators)
        
    def _evaluate_feedback_addressing(self, response: str) -> float:
        """Evaluate how well response addresses feedback."""
        if not self.feedback_history:
            return 0.0
            
        latest_feedback = self.feedback_history[-1].lower()
        response_lower = response.lower()
        
        # Look for references to feedback points
        feedback_keywords = ["improve", "fix", "address", "change", "update"]
        addressing_score = sum(1 for word in feedback_keywords if word in response_lower) / len(feedback_keywords)
        
        return min(addressing_score, 1.0)
        
    def _evaluate_improvement_quality(self, response: str) -> float:
        """Evaluate quality of improvement."""
        improvement_indicators = [
            "better" in response.lower() or "improved" in response.lower(),
            "optimization" in response.lower() or "optimize" in response.lower(),
            "efficient" in response.lower(),
            len(response) > 100,  # Sufficient explanation
        ]
        
        return sum(improvement_indicators) / len(improvement_indicators)
        
    def _evaluate_convergence_progress(self) -> float:
        """Evaluate progress towards convergence."""
        if len(self.solution_attempts) < 2:
            return 0.0
            
        # Simple heuristic: later solutions should be more comprehensive
        lengths = [len(attempt) for attempt in self.solution_attempts]
        if len(lengths) > 1:
            return 1.0 if lengths[-1] > lengths[0] else 0.5
            
        return 0.5
        
    def _generate_feedback(self, solution: str) -> str:
        """Generate feedback for solution (simplified)."""
        feedback_points = []
        
        if "```" not in solution:
            feedback_points.append("Consider providing code implementation")
            
        if len(solution) < 50:
            feedback_points.append("Please provide more detailed explanation")
            
        if "test" not in solution.lower():
            feedback_points.append("Consider adding test cases or examples")
            
        if not feedback_points:
            feedback_points.append("Good solution! Consider edge cases and optimizations")
            
        return "\n".join(f"- {point}" for point in feedback_points)
        
    def _calculate_improvement_trajectory(self, scores: List[float]) -> float:
        """Calculate improvement trajectory over iterations."""
        if len(scores) < 2:
            return 0.5
            
        improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        positive_improvements = sum(1 for imp in improvements if imp > 0)
        
        return positive_improvements / len(improvements)


# ==================== Teaching Dialogue Scenario ====================

@register_scenario("teaching_dialogue") 
class TeachingDialogueScenario(ConversationalScenario):
    """
    Teaching/instructional dialogue scenario.
    
    Simulates a teaching session with explanation, questions,
    clarification, and knowledge assessment.
    """
    
    def __init__(self, config: ScenarioConfig):
        super().__init__(config)
        self.learning_objectives = []
        self.student_questions = []
        self.assessment_results = []
        
    def generate_initial_prompt(self, problem_data: Dict[str, Any]) -> str:
        """Generate initial teaching prompt."""
        topic = problem_data.get("topic", "")
        level = problem_data.get("level", "beginner")
        objectives = problem_data.get("objectives", [])
        
        self.learning_objectives = objectives
        
        prompt = f"""# Teaching Session: {topic}

**Student Level**: {level}

**Learning Objectives**: {', '.join(objectives)}

I'd like to learn about {topic}. Please start by explaining the fundamental concepts 
in a way that's appropriate for a {level} level student. 

Use examples and make it engaging!"""
        
        self.add_to_history("user", prompt)
        return prompt
        
    def process_turn_response(self, turn_id: str, response: str, 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Process teaching dialogue turn response."""
        result = super().process_turn_response(turn_id, response, context)
        
        # Add teaching-specific evaluations
        if turn_id.startswith("explanation_"):
            result.update({
                "pedagogical_clarity": self._evaluate_pedagogical_clarity(response),
                "example_quality": self._evaluate_example_quality(response),
                "engagement_level": self._evaluate_engagement_level(response),
            })
            
        elif turn_id.startswith("question_response_"):
            result.update({
                "question_addressing": self._evaluate_question_addressing(response),
                "clarification_quality": self._evaluate_clarification_quality(response),
            })
            
        elif turn_id.startswith("assessment_"):
            assessment_score = self._evaluate_assessment_response(response)
            self.assessment_results.append(assessment_score)
            result["assessment_score"] = assessment_score
            
        return result
        
    def generate_next_prompt(self, turn_id: str, 
                           previous_responses: List[str]) -> Optional[str]:
        """Generate next teaching prompt."""
        if turn_id.startswith("student_question_"):
            # Generate a student question based on the topic
            questions = [
                "Can you give me another example?",
                "What if we have a different scenario?",
                "I'm confused about one part - could you clarify?",
                "How does this relate to real-world applications?",
                "What are the most common mistakes people make with this?"
            ]
            
            import random
            question = random.choice(questions)
            self.student_questions.append(question)
            
            self.add_to_history("user", question)
            return question
            
        elif turn_id.startswith("assessment_question_"):
            # Generate assessment question
            prompt = f"""Now I'd like to test my understanding. 

Please give me a practical problem or question related to {self.config.global_context.get('topic', 'the topic')} 
that I can solve to demonstrate my learning."""
            
            self.add_to_history("user", prompt)
            return prompt
            
        return None
        
    def evaluate_scenario(self) -> Dict[str, float]:
        """Evaluate teaching dialogue scenario."""
        base_metrics = super().evaluate_scenario()
        
        # Add teaching-specific metrics
        teaching_metrics = {
            "pedagogical_effectiveness": self._calculate_pedagogical_effectiveness(),
            "student_engagement": self._calculate_student_engagement(),
            "learning_progression": self._calculate_learning_progression(),
            "assessment_success": sum(self.assessment_results) / len(self.assessment_results) if self.assessment_results else 0.0,
        }
        
        # Combine metrics
        all_metrics = {**base_metrics, **teaching_metrics}
        
        # Recalculate overall score with teaching components
        teaching_components = [
            teaching_metrics["pedagogical_effectiveness"],
            teaching_metrics["student_engagement"],
            teaching_metrics["learning_progression"],
            base_metrics.get("average_coherence", 0),
        ]
        
        all_metrics["overall_score"] = sum(teaching_components) / len(teaching_components)
        return all_metrics
        
    def _evaluate_pedagogical_clarity(self, response: str) -> float:
        """Evaluate pedagogical clarity of explanation."""
        clarity_indicators = [
            "for example" in response.lower() or "example" in response.lower(),
            "first" in response.lower() or "step" in response.lower(),
            "simply" in response.lower() or "basically" in response.lower(),
            len(response.split(".")) > 2,  # Multiple sentences
            "because" in response.lower() or "reason" in response.lower(),
        ]
        
        return sum(clarity_indicators) / len(clarity_indicators)
        
    def _evaluate_example_quality(self, response: str) -> float:
        """Evaluate quality of examples provided."""
        example_indicators = [
            "example" in response.lower(),
            "imagine" in response.lower() or "suppose" in response.lower(),
            "like" in response.lower() and ":" in response,  # Analogies
            len(re.findall(r"\d+", response)) > 0,  # Numerical examples
        ]
        
        return sum(example_indicators) / len(example_indicators)
        
    def _evaluate_engagement_level(self, response: str) -> float:
        """Evaluate engagement level of teaching response."""
        engagement_indicators = [
            "you" in response.lower(),  # Direct address
            "?" in response,  # Questions to student
            "!" in response,  # Enthusiasm
            "interesting" in response.lower() or "cool" in response.lower(),
            len(response) > 100,  # Comprehensive response
        ]
        
        return sum(engagement_indicators) / len(engagement_indicators)
        
    def _evaluate_question_addressing(self, response: str) -> float:
        """Evaluate how well the response addresses student questions."""
        if not self.student_questions:
            return 0.5
            
        latest_question = self.student_questions[-1].lower()
        response_lower = response.lower()
        
        # Check if response references the question
        if "question" in response_lower or "ask" in response_lower:
            return 1.0
        elif len(response) > 50:  # Comprehensive response
            return 0.8
        else:
            return 0.5
            
    def _evaluate_clarification_quality(self, response: str) -> float:
        """Evaluate quality of clarification."""
        clarification_indicators = [
            "clarify" in response.lower() or "explain" in response.lower(),
            "other words" in response.lower() or "differently" in response.lower(),
            "simpler" in response.lower() or "easier" in response.lower(),
            len(response.split()) > 30,  # Detailed clarification
        ]
        
        return sum(clarification_indicators) / len(clarification_indicators)
        
    def _evaluate_assessment_response(self, response: str) -> float:
        """Evaluate assessment response (simplified)."""
        # This would typically involve checking against correct answers
        assessment_indicators = [
            len(response) > 20,  # Sufficient length
            "because" in response.lower() or "reason" in response.lower(),  # Reasoning
            any(obj.lower() in response.lower() for obj in self.learning_objectives),  # References objectives
        ]
        
        return sum(assessment_indicators) / len(assessment_indicators)
        
    def _calculate_pedagogical_effectiveness(self) -> float:
        """Calculate overall pedagogical effectiveness."""
        clarity_scores = []
        example_scores = []
        
        for result in self.turn_results.values():
            if "pedagogical_clarity" in result:
                clarity_scores.append(result["pedagogical_clarity"])
            if "example_quality" in result:
                example_scores.append(result["example_quality"])
                
        components = []
        if clarity_scores:
            components.append(sum(clarity_scores) / len(clarity_scores))
        if example_scores:
            components.append(sum(example_scores) / len(example_scores))
            
        return sum(components) / len(components) if components else 0.0
        
    def _calculate_student_engagement(self) -> float:
        """Calculate student engagement level."""
        engagement_scores = [
            result.get("engagement_level", 0)
            for result in self.turn_results.values()
            if "engagement_level" in result
        ]
        
        return sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0.0
        
    def _calculate_learning_progression(self) -> float:
        """Calculate learning progression throughout the dialogue."""
        # Simple heuristic: more questions asked indicates engagement and learning
        question_progression = len(self.student_questions) / max(len(self.conversation_history), 1)
        
        # Assessment improvement
        assessment_progression = 0.0
        if len(self.assessment_results) > 1:
            assessment_progression = (self.assessment_results[-1] - self.assessment_results[0])
            
        return (question_progression + max(assessment_progression, 0)) / 2