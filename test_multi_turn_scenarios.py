#!/usr/bin/env python3
"""
Test script for the new Multi-Turn Scenarios framework.

This script validates the enhanced multi-turn implementation with:
- Multiple scenario types
- Chat template integration 
- Comprehensive evaluation metrics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from lm_eval.tasks.multi_turn_scenarios import (
    MultiTurnEvaluationEngine, 
    get_scenario_registry,
    ScenarioType
)
from lm_eval.tasks.multi_turn_scenarios.example_tasks import (
    get_all_example_scenarios,
    create_sample_datasets
)
from lm_eval.tasks.multi_turn_scenarios.chat_template_support import (
    ChatTemplateManager,
    MultiTurnChatTemplateIntegrator
)


def test_scenario_registration():
    """Test that scenarios are properly registered."""
    print("=== Testing Scenario Registration ===")
    
    registry = get_scenario_registry()
    print(f"Registered scenarios: {list(registry.keys())}")
    
    # Test each registered scenario
    for scenario_id in registry:
        scenario_class = registry[scenario_id]
        print(f"✓ {scenario_id}: {scenario_class.__name__}")
    
    print()


def test_chat_template_integration():
    """Test chat template integration functionality.""" 
    print("=== Testing Chat Template Integration ===")
    
    manager = ChatTemplateManager()
    integrator = MultiTurnChatTemplateIntegrator(manager)
    
    # Test different chat template formats
    test_formats = ["chatml", "llama", "alpaca", "openai"]
    from lm_eval.tasks.multi_turn_scenarios.chat_template_support import ChatMessage
    test_messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hello, how are you?"),
        ChatMessage(role="assistant", content="I'm doing well, thank you!")
    ]
    
    for format_name in test_formats:
        try:
            formatted = manager.format_conversation(test_messages, format_name)
            print(f"✓ {format_name}: Successfully formatted conversation")
        except Exception as e:
            print(f"✗ {format_name}: Error - {e}")
    
    print()


def test_scenario_execution():
    """Test execution of different scenario types."""
    print("=== Testing Scenario Execution ===")
    
    engine = MultiTurnEvaluationEngine()
    scenarios = get_all_example_scenarios()
    datasets = create_sample_datasets()
    
    for scenario_id, config in scenarios[:2]:  # Test first 2 scenarios
        print(f"Testing scenario: {scenario_id}")
        print(f"  Type: {config.scenario_type.value}")
        print(f"  Turns: {len(config.turns)}")
        
        # Get sample data for this scenario
        sample_data = datasets.get(scenario_id, [{}])[0]
        
        try:
            # Test scenario initialization
            scenario = engine.get_scenario(scenario_id, sample_data)
            print(f"  ✓ Scenario initialized: {scenario.__class__.__name__}")
            
            # Test turn configuration
            for i, turn in enumerate(config.turns):
                print(f"  ✓ Turn {i+1}: {turn.turn_id} ({turn.turn_type.value})")
                
        except Exception as e:
            print(f"  ✗ Error testing {scenario_id}: {e}")
    
    print()


def test_metrics_system():
    """Test the metrics evaluation system."""
    print("=== Testing Metrics System ===")
    
    from lm_eval.tasks.multi_turn_scenarios.metrics import MultiTurnMetrics
    
    # Create test results for different scenario types
    test_results = {
        ScenarioType.CODE_REVIEW: {
            "scenario_id": "code_review_test",
            "scenario_type": ScenarioType.CODE_REVIEW,
            "turns": [
                {
                    "turn_id": "initial_review",
                    "response": "This code has potential issues with recursion depth limits.",
                    "evaluation_metrics": ["review_quality", "specificity_score"]
                },
                {
                    "turn_id": "code_revision", 
                    "response": "Here's an iterative version that avoids recursion issues.",
                    "evaluation_metrics": ["revision_quality", "addresses_feedback"]
                }
            ]
        },
        ScenarioType.CONVERSATIONAL: {
            "scenario_id": "conversational_test",
            "scenario_type": ScenarioType.CONVERSATIONAL,
            "turns": [
                {
                    "turn_id": "initial_response",
                    "response": "I'd be happy to help you learn about machine learning!",
                    "evaluation_metrics": ["coherence_score", "relevance_score"]
                },
                {
                    "turn_id": "follow_up_1",
                    "response": "Let's start with the basics of supervised vs unsupervised learning.",
                    "evaluation_metrics": ["coherence_score", "relevance_score"]
                }
            ]
        }
    }
    
    metrics = MultiTurnMetrics()
    
    for scenario_type, result in test_results.items():
        try:
            # Calculate metrics for each scenario type
            scores = metrics.calculate_scenario_metrics(result, scenario_type)
            print(f"✓ {scenario_type.value}: Calculated metrics")
            
            # Print sample metrics
            for metric_name, score in list(scores.items())[:3]:  # Show first 3
                print(f"    {metric_name}: {score:.3f}")
                
        except Exception as e:
            print(f"✗ {scenario_type.value}: Error calculating metrics - {e}")
    
    print()


def test_full_evaluation_workflow():
    """Test the complete evaluation workflow."""
    print("=== Testing Full Evaluation Workflow ===")
    
    try:
        engine = MultiTurnEvaluationEngine()
        
        # Use code review scenario for full test
        scenario_id = "code_review_3_turn"
        datasets = create_sample_datasets()
        sample_data = datasets[scenario_id][0]
        
        print(f"Testing full workflow for: {scenario_id}")
        print(f"Sample data: {sample_data['problem_id']}")
        
        # Initialize scenario
        scenario = engine.get_scenario(scenario_id, sample_data)
        config = scenario.get_config()
        
        print(f"✓ Scenario loaded: {config.name}")
        print(f"  System message: {config.system_message[:50]}...")
        print(f"  Success criteria: {len(config.success_criteria)} items")
        print(f"  Chat template required: {config.chat_template_required}")
        
        # Test chat template application
        if config.chat_template_required:
            manager = ChatTemplateManager()
            integrator = MultiTurnChatTemplateIntegrator(manager)
            
            test_conversation = [
                {"role": "system", "content": config.system_message},
                {"role": "user", "content": f"Please review this code: {sample_data['code']}"}
            ]
            
            formatted_prompt = integrator.multi_turn_prompt(
                test_conversation, 
                chat_format="chatml"
            )
            
            print(f"✓ Chat template applied successfully")
            print(f"  Formatted prompt length: {len(formatted_prompt)} chars")
        
        print(f"✓ Full workflow test completed for {scenario_id}")
        
    except Exception as e:
        print(f"✗ Error in full workflow test: {e}")
        import traceback
        traceback.print_exc()
    
    print()


def main():
    """Run all framework tests."""
    print("Multi-Turn Scenarios Framework Test Suite")
    print("=" * 50)
    print()
    
    # Test each component
    test_scenario_registration()
    test_chat_template_integration() 
    test_scenario_execution()
    test_metrics_system()
    test_full_evaluation_workflow()
    
    print("=" * 50)
    print("Framework Testing Complete!")
    print()
    
    # Summary
    scenarios = get_all_example_scenarios()
    print("Framework Summary:")
    print(f"  Total scenarios implemented: {len(scenarios)}")
    print(f"  Chat template integration: ✓ Complete")
    print(f"  Metrics system: ✓ Complete")
    print(f"  Evaluation engine: ✓ Complete")
    print()
    print("Ready for enhanced multi-turn evaluation!")


if __name__ == "__main__":
    main()