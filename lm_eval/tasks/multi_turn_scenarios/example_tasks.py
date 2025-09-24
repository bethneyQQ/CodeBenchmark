"""
Example Multi-Turn Scenario Task Configurations.

This module demonstrates how to create lm-eval compatible tasks
using the new multi-turn scenarios framework.
"""

from .base_scenario import TurnConfig, ScenarioConfig, TurnType, ScenarioType
from .scenario_registry import register_scenario
from .concrete_scenarios import CodeReviewScenario, IterativeProblemSolvingScenario, TeachingDialogueScenario


def create_code_review_config() -> ScenarioConfig:
    """Create configuration for code review scenario."""
    turns = [
        TurnConfig(
            turn_id="initial_review",
            turn_type=TurnType.ASSISTANT_RESPONSE,
            role="assistant",
            prompt_template="Review the provided code and give detailed feedback.",
            evaluation_metrics=["review_quality", "specificity_score", "actionability_score"],
            temperature=0.1,
            max_tokens=1500
        ),
        TurnConfig(
            turn_id="code_revision",
            turn_type=TurnType.ASSISTANT_RESPONSE, 
            role="assistant",
            prompt_template="Revise the code based on the review feedback.",
            depends_on=["initial_review"],
            evaluation_metrics=["revision_quality", "addresses_feedback", "code_improvement"],
            temperature=0.2,
            max_tokens=2000
        ),
        TurnConfig(
            turn_id="final_evaluation",
            turn_type=TurnType.ASSISTANT_RESPONSE,
            role="assistant", 
            prompt_template="Provide final evaluation of the code revision.",
            depends_on=["initial_review", "code_revision"],
            evaluation_metrics=["evaluation_completeness"],
            temperature=0.1,
            max_tokens=800
        )
    ]
    
    return ScenarioConfig(
        scenario_id="code_review_3_turn",
        scenario_type=ScenarioType.CODE_REVIEW,
        name="Code Review with Revision",
        description="Multi-turn code review scenario with initial review, revision, and evaluation",
        turns=turns,
        chat_template_required=True,
        system_message="You are an experienced code reviewer. Provide constructive, specific feedback.",
        success_criteria=[
            "Provides specific, actionable review feedback",
            "Addresses review comments in revision", 
            "Shows measurable code improvement"
        ],
        evaluation_strategy="cumulative"
    )


def create_iterative_problem_solving_config() -> ScenarioConfig:
    """Create configuration for iterative problem solving scenario.""" 
    turns = [
        TurnConfig(
            turn_id="solution_attempt_1",
            turn_type=TurnType.ASSISTANT_RESPONSE,
            role="assistant",
            prompt_template="Provide your initial solution approach.",
            evaluation_metrics=["solution_completeness", "approach_clarity"],
            temperature=0.3,
            max_tokens=2000
        ),
        TurnConfig(
            turn_id="refinement_1",
            turn_type=TurnType.ASSISTANT_RESPONSE,
            role="assistant", 
            prompt_template="Refine your solution based on feedback.",
            depends_on=["solution_attempt_1"],
            evaluation_metrics=["addresses_feedback", "improvement_quality"],
            temperature=0.2,
            max_tokens=1800
        ),
        TurnConfig(
            turn_id="solution_attempt_2", 
            turn_type=TurnType.ASSISTANT_RESPONSE,
            role="assistant",
            prompt_template="Provide your improved solution.",
            depends_on=["refinement_1"],
            evaluation_metrics=["solution_completeness", "convergence_progress"],
            temperature=0.1,
            max_tokens=2000
        )
    ]
    
    return ScenarioConfig(
        scenario_id="iterative_problem_solving",
        scenario_type=ScenarioType.ITERATIVE,
        name="Iterative Problem Solving",
        description="Multi-turn iterative problem solving with refinement cycles",
        turns=turns,
        chat_template_required=True,
        system_message="You are solving a complex problem. Iterate and improve your solution based on feedback.",
        success_criteria=[
            "Shows clear improvement trajectory",
            "Addresses feedback constructively",
            "Converges to quality solution"
        ],
        evaluation_strategy="cumulative"
    )


def create_teaching_dialogue_config() -> ScenarioConfig:
    """Create configuration for teaching dialogue scenario."""
    turns = [
        TurnConfig(
            turn_id="explanation_1",
            turn_type=TurnType.ASSISTANT_RESPONSE,
            role="assistant",
            prompt_template="Explain the fundamental concepts clearly.",
            evaluation_metrics=["pedagogical_clarity", "example_quality", "engagement_level"],
            temperature=0.4,
            max_tokens=1500
        ),
        TurnConfig(
            turn_id="question_response_1",
            turn_type=TurnType.ASSISTANT_RESPONSE,
            role="assistant",
            prompt_template="Address student's follow-up questions.",
            depends_on=["explanation_1"],
            evaluation_metrics=["question_addressing", "clarification_quality"],
            temperature=0.3,
            max_tokens=1200
        ),
        TurnConfig(
            turn_id="assessment_1",
            turn_type=TurnType.ASSISTANT_RESPONSE,
            role="assistant",
            prompt_template="Provide assessment questions to test understanding.",
            depends_on=["explanation_1", "question_response_1"],
            evaluation_metrics=["assessment_score"],
            temperature=0.2,
            max_tokens=1000
        )
    ]
    
    return ScenarioConfig(
        scenario_id="teaching_dialogue", 
        scenario_type=ScenarioType.INSTRUCTIONAL,
        name="Teaching Dialogue Session",
        description="Multi-turn teaching session with explanation, Q&A, and assessment",
        turns=turns,
        chat_template_required=True,
        system_message="You are a patient, knowledgeable teacher. Explain concepts clearly with examples.",
        success_criteria=[
            "Clear, engaging explanations",
            "Responsive to student questions",
            "Effective knowledge assessment"
        ],
        evaluation_strategy="per_turn"
    )


# Register scenarios with their configurations
register_scenario("code_review_3_turn", create_code_review_config())(CodeReviewScenario)
register_scenario("iterative_problem_solving", create_iterative_problem_solving_config())(IterativeProblemSolvingScenario)  
register_scenario("teaching_dialogue", create_teaching_dialogue_config())(TeachingDialogueScenario)


def create_conversational_config() -> ScenarioConfig:
    """Create configuration for general conversational scenario."""
    turns = [
        TurnConfig(
            turn_id="initial_response",
            turn_type=TurnType.ASSISTANT_RESPONSE,
            role="assistant",
            prompt_template="Respond to the user's initial query naturally.",
            evaluation_metrics=["coherence_score", "relevance_score"],
            temperature=0.7,
            max_tokens=1200
        ),
        TurnConfig(
            turn_id="follow_up_1",
            turn_type=TurnType.ASSISTANT_RESPONSE,
            role="assistant",
            prompt_template="Continue the conversation based on context.",
            depends_on=["initial_response"],
            evaluation_metrics=["coherence_score", "relevance_score"],
            context_window=3,
            temperature=0.6,
            max_tokens=1000
        ),
        TurnConfig(
            turn_id="follow_up_2", 
            turn_type=TurnType.ASSISTANT_RESPONSE,
            role="assistant",
            prompt_template="Maintain conversational flow.",
            depends_on=["follow_up_1"],
            evaluation_metrics=["coherence_score", "relevance_score"],
            context_window=5,
            temperature=0.6,
            max_tokens=1000
        )
    ]
    
    return ScenarioConfig(
        scenario_id="conversational_3_turn",
        scenario_type=ScenarioType.CONVERSATIONAL,
        name="Natural Conversation",
        description="Multi-turn natural conversation evaluation", 
        turns=turns,
        chat_template_required=True,
        system_message="You are engaging in a natural conversation. Be helpful and conversational.",
        success_criteria=[
            "Maintains conversation coherence",
            "Responses are relevant and engaging",
            "Natural dialogue flow"
        ],
        evaluation_strategy="cumulative"
    )


# Register conversational scenario
from .base_scenario import ConversationalScenario
register_scenario("conversational_3_turn", create_conversational_config())(ConversationalScenario)


def get_all_example_scenarios():
    """Get list of all example scenario configurations."""
    return [
        ("code_review_3_turn", create_code_review_config()),
        ("iterative_problem_solving", create_iterative_problem_solving_config()),
        ("teaching_dialogue", create_teaching_dialogue_config()),
        ("conversational_3_turn", create_conversational_config())
    ]


def create_sample_datasets():
    """Create sample datasets for each scenario type."""
    
    datasets = {
        "code_review_3_turn": [
            {
                "problem_id": "code_review_001",
                "code": '''def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)''',
                "context": "Review this recursive factorial implementation",
                "language": "python",
                "complexity": "beginner"
            },
            {
                "problem_id": "code_review_002", 
                "code": '''class UserManager:
    def __init__(self):
        self.users = []
    
    def add_user(self, user):
        self.users.append(user)
        
    def get_user(self, user_id):
        for user in self.users:
            if user.id == user_id:
                return user''',
                "context": "Review this user management class for potential improvements",
                "language": "python",
                "complexity": "intermediate"
            }
        ],
        
        "iterative_problem_solving": [
            {
                "problem_id": "optimization_001",
                "problem": "Find the most efficient way to sort a list of 1 million integers",
                "constraints": [
                    "Memory usage should be minimized",
                    "Must handle duplicate values",
                    "Should be stable sort"
                ],
                "examples": [
                    {"input": "[3, 1, 4, 1, 5]", "output": "[1, 1, 3, 4, 5]"}
                ]
            },
            {
                "problem_id": "algorithm_001",
                "problem": "Design an algorithm to find the shortest path in a weighted graph",
                "constraints": [
                    "Handle negative weights",
                    "Detect negative cycles", 
                    "Work with directed graphs"
                ],
                "examples": [
                    {"description": "Graph with 4 nodes, edges: (A,B,1), (B,C,2), (A,C,4)"}
                ]
            }
        ],
        
        "teaching_dialogue": [
            {
                "problem_id": "teach_001",
                "topic": "Binary Search Algorithm",
                "level": "beginner",
                "objectives": [
                    "Understand the concept of divide and conquer",
                    "Learn when to use binary search",
                    "Implement binary search correctly"
                ]
            },
            {
                "problem_id": "teach_002",
                "topic": "Object-Oriented Programming Principles",
                "level": "intermediate", 
                "objectives": [
                    "Understand encapsulation, inheritance, polymorphism",
                    "Apply OOP principles in practice",
                    "Design class hierarchies effectively"
                ]
            }
        ],
        
        "conversational_3_turn": [
            {
                "problem_id": "conversation_001",
                "initial_query": "I'm interested in learning about machine learning. Where should I start?",
                "context": "Help-seeking conversation about ML",
                "user_background": "Programming experience but new to ML"
            },
            {
                "problem_id": "conversation_002",
                "initial_query": "What are the pros and cons of remote work?",
                "context": "Discussion about work arrangements",
                "user_background": "Professional considering remote work"
            }
        ]
    }
    
    return datasets


if __name__ == "__main__":
    # Example usage
    scenarios = get_all_example_scenarios()
    datasets = create_sample_datasets()
    
    print("Available scenarios:")
    for scenario_id, config in scenarios:
        print(f"- {scenario_id}: {config.name}")
        print(f"  Type: {config.scenario_type.value}")
        print(f"  Turns: {len(config.turns)}")
        print(f"  Chat template required: {config.chat_template_required}")
        print()
        
    print("Sample datasets created for each scenario type.")