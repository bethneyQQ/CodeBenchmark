"""
Chat Template Support for Multi-Turn Scenarios.

This module provides chat template integration for multi-turn evaluation,
enabling proper formatting for instruction-tuned and chat models.
"""

import re
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass


@dataclass
class ChatMessage:
    """Represents a single message in a chat conversation."""
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class ChatTemplateManager:
    """
    Manages chat template formatting for multi-turn scenarios.
    
    This class provides integration with lm-eval's chat template system
    and supports various conversation formats.
    """
    
    def __init__(self, template_name: str = "default"):
        self.template_name = template_name
        self.supported_roles = {"user", "assistant", "system", "function"}
        self.default_templates = self._load_default_templates()
        
    def _load_default_templates(self) -> Dict[str, str]:
        """Load default chat templates for common formats."""
        return {
            "default": "{role}: {content}",
            "chatml": "<|im_start|>{role}\n{content}<|im_end|>",
            "llama": "[INST] {content} [/INST]",  # For user messages
            "alpaca": "### {role}:\n{content}\n\n",
            "vicuna": "{role}: {content}",
            "openai": '{"role": "{role}", "content": "{content}"}',
            "anthropic": "\n\nHuman: {content}\n\nAssistant:",  # Special handling needed
        }
        
    def format_conversation(self, 
                          messages: List[ChatMessage], 
                          template: Optional[str] = None,
                          model_template_fn: Optional[Callable] = None) -> str:
        """
        Format a conversation using specified template.
        
        Args:
            messages: List of chat messages
            template: Template string or name
            model_template_fn: Optional model-specific template function
            
        Returns:
            Formatted conversation string
        """
        if model_template_fn:
            # Use model's built-in chat template function
            chat_history = [{"role": msg.role, "content": msg.content} for msg in messages]
            return model_template_fn(chat_history)
            
        # Use internal template
        template_str = self._get_template_string(template)
        formatted_parts = []
        
        for msg in messages:
            if msg.role in self.supported_roles:
                formatted_part = template_str.format(
                    role=msg.role.title(),
                    content=msg.content.strip()
                )
                formatted_parts.append(formatted_part)
                
        return "\n".join(formatted_parts)
        
    def _get_template_string(self, template: Optional[str]) -> str:
        """Get template string from name or return the string itself."""
        if not template:
            return self.default_templates["default"]
            
        if template in self.default_templates:
            return self.default_templates[template]
            
        # Assume it's a custom template string
        return template
        
    def create_multi_turn_prompt(self,
                                system_message: Optional[str],
                                conversation_history: List[Dict[str, str]],
                                current_prompt: str,
                                template: Optional[str] = None) -> str:
        """
        Create a complete multi-turn prompt with chat formatting.
        
        Args:
            system_message: Optional system instruction
            conversation_history: Previous conversation turns
            current_prompt: Current user query/prompt
            template: Chat template to use
            
        Returns:
            Formatted multi-turn prompt
        """
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append(ChatMessage("system", system_message))
            
        # Add conversation history
        for entry in conversation_history:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            if role and content:
                messages.append(ChatMessage(role, content))
                
        # Add current prompt
        if current_prompt:
            messages.append(ChatMessage("user", current_prompt))
            
        return self.format_conversation(messages, template)
        
    def extract_assistant_responses(self, 
                                  full_response: str,
                                  template: Optional[str] = None) -> List[str]:
        """
        Extract assistant responses from a formatted conversation.
        
        Args:
            full_response: Complete formatted response
            template: Template used for formatting
            
        Returns:
            List of assistant response strings
        """
        template_str = self._get_template_string(template)
        
        # Create pattern to match assistant responses
        if "{role}" in template_str and "{content}" in template_str:
            # Generic pattern based on template
            pattern = template_str.replace("{role}", "Assistant").replace("{content}", r"(.*?)")
            pattern = re.escape(pattern).replace(r"\\(\\.\*\\?\\)", r"(.*?)")
        else:
            # Fallback patterns
            patterns = [
                r"Assistant: (.*?)(?=User:|$)",
                r"assistant: (.*?)(?=user:|$)",  
                r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>",
                r"### Assistant:\n(.*?)(?=###|$)",
            ]
            pattern = "|".join(f"({p})" for p in patterns)
            
        matches = re.findall(pattern, full_response, re.DOTALL | re.IGNORECASE)
        
        # Flatten and clean matches
        responses = []
        for match in matches:
            if isinstance(match, tuple):
                for group in match:
                    if group and group.strip():
                        responses.append(group.strip())
                        break
            elif match and match.strip():
                responses.append(match.strip())
                
        return responses
        
    def validate_chat_format(self, 
                           conversation: str, 
                           template: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate that a conversation follows the expected chat format.
        
        Args:
            conversation: Formatted conversation string
            template: Expected template format
            
        Returns:
            Validation results dictionary
        """
        template_str = self._get_template_string(template)
        
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "message_count": 0,
            "role_distribution": {}
        }
        
        try:
            # Count role occurrences
            for role in self.supported_roles:
                role_pattern = template_str.replace("{role}", role.title()).replace("{content}", r".*?")
                matches = re.findall(role_pattern, conversation, re.IGNORECASE)
                if matches:
                    validation_result["role_distribution"][role] = len(matches)
                    
            validation_result["message_count"] = sum(validation_result["role_distribution"].values())
            
            # Basic validation checks
            if validation_result["message_count"] == 0:
                validation_result["is_valid"] = False
                validation_result["errors"].append("No messages detected in conversation")
                
            if "user" not in validation_result["role_distribution"]:
                validation_result["warnings"].append("No user messages detected")
                
            if "assistant" not in validation_result["role_distribution"]:
                validation_result["warnings"].append("No assistant responses detected")
                
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Validation error: {e}")
            
        return validation_result
        
    def convert_to_lm_eval_format(self, 
                                 messages: List[ChatMessage]) -> List[Dict[str, str]]:
        """
        Convert messages to lm-eval compatible format.
        
        Args:
            messages: List of ChatMessage objects
            
        Returns:
            List of dictionaries with 'role' and 'content' keys
        """
        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role in self.supported_roles
        ]
        
    def support_fewshot_as_multiturn(self, 
                                    fewshot_examples: List[Dict[str, str]],
                                    current_query: str,
                                    system_message: Optional[str] = None) -> List[ChatMessage]:
        """
        Convert few-shot examples to multi-turn conversation format.
        
        This supports lm-eval's --fewshot_as_multiturn functionality.
        
        Args:
            fewshot_examples: List of few-shot examples with input/output
            current_query: Current query to evaluate
            system_message: Optional system instruction
            
        Returns:
            List of ChatMessage objects representing the conversation
        """
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append(ChatMessage("system", system_message))
            
        # Convert few-shot examples to conversation turns
        for example in fewshot_examples:
            if "input" in example:
                messages.append(ChatMessage("user", example["input"]))
            if "output" in example:
                messages.append(ChatMessage("assistant", example["output"]))
                
        # Add current query
        if current_query:
            messages.append(ChatMessage("user", current_query))
            
        return messages


class MultiTurnChatTemplateIntegrator:
    """
    Integrates chat templates with multi-turn scenario evaluation.
    
    This class bridges the gap between lm-eval's chat template system
    and multi-turn scenario requirements.
    """
    
    def __init__(self, chat_manager: ChatTemplateManager):
        self.chat_manager = chat_manager
        
    def prepare_scenario_with_chat_template(self,
                                          scenario_config: Dict[str, Any],
                                          model_template_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Prepare a multi-turn scenario with chat template integration.
        
        Args:
            scenario_config: Original scenario configuration
            model_template_fn: Model's chat template function
            
        Returns:
            Updated configuration with chat template support
        """
        updated_config = scenario_config.copy()
        
        # Mark as chat-template enabled
        updated_config["chat_template_enabled"] = True
        updated_config["model_template_fn"] = model_template_fn
        
        # Update prompt generation functions to use chat templates
        if "turns" in updated_config:
            for turn in updated_config["turns"]:
                if "prompt_template" in turn:
                    turn["original_prompt_template"] = turn["prompt_template"]
                    turn["use_chat_template"] = True
                    
        return updated_config
        
    def format_turn_with_chat_template(self,
                                     turn_config: Dict[str, Any],
                                     conversation_history: List[Dict[str, str]],
                                     context: Dict[str, Any],
                                     model_template_fn: Optional[Callable] = None) -> str:
        """
        Format a single turn using chat template.
        
        Args:
            turn_config: Configuration for the current turn
            conversation_history: Previous conversation history
            context: Additional context for prompt generation
            model_template_fn: Model's chat template function
            
        Returns:
            Formatted prompt string
        """
        # Generate the content for this turn
        prompt_template = turn_config.get("original_prompt_template", turn_config.get("prompt_template", ""))
        
        try:
            turn_content = prompt_template.format(**context)
        except KeyError as e:
            turn_content = f"Error in prompt template: {e}"
            
        # Create chat messages
        messages = []
        
        # Add system message if present in context
        if context.get("system_message"):
            messages.append(ChatMessage("system", context["system_message"]))
            
        # Add conversation history
        for entry in conversation_history:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            if role and content:
                messages.append(ChatMessage(role, content))
                
        # Add current turn
        current_role = turn_config.get("role", "user")
        messages.append(ChatMessage(current_role, turn_content))
        
        # Format using chat template
        if model_template_fn:
            chat_history = self.chat_manager.convert_to_lm_eval_format(messages)
            return model_template_fn(chat_history)
        else:
            template_name = turn_config.get("chat_template", "default")
            return self.chat_manager.format_conversation(messages, template_name)
            
    def extract_and_evaluate_responses(self,
                                     full_response: str,
                                     expected_turns: List[str],
                                     template: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract responses from chat-formatted output and prepare for evaluation.
        
        Args:
            full_response: Complete model response
            expected_turns: Expected turn IDs
            template: Chat template used
            
        Returns:
            Dictionary mapping turn IDs to extracted responses
        """
        assistant_responses = self.chat_manager.extract_assistant_responses(full_response, template)
        
        # Map responses to turn IDs
        turn_responses = {}
        for i, turn_id in enumerate(expected_turns):
            if i < len(assistant_responses):
                turn_responses[turn_id] = assistant_responses[i]
            else:
                turn_responses[turn_id] = ""  # Missing response
                
        return {
            "turn_responses": turn_responses,
            "raw_response": full_response,
            "extraction_success": len(assistant_responses) == len(expected_turns),
            "response_count": len(assistant_responses),
            "expected_count": len(expected_turns)
        }
    
    def multi_turn_prompt(self, 
                         conversation: List[Dict[str, str]], 
                         chat_format: str = "chatml") -> str:
        """
        Create a formatted multi-turn prompt from conversation history.
        
        Args:
            conversation: List of message dictionaries with 'role' and 'content'
            chat_format: Chat template format to use
            
        Returns:
            Formatted prompt string ready for model input
        """
        # Convert dictionaries to ChatMessage objects if needed
        chat_messages = []
        for msg in conversation:
            if isinstance(msg, dict):
                chat_messages.append(ChatMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", "")
                ))
            else:
                chat_messages.append(msg)
        
        return self.chat_manager.format_conversation(chat_messages, chat_format)