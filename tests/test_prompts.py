"""Tests for prompt templates in prompts.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from prompts import (
    get_intent_classification_prompt,
    get_chat_prompt_template,
    QA_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    CALCULATION_SYSTEM_PROMPT,
    MEMORY_SUMMARY_PROMPT,
)


class TestIntentClassificationPrompt:
    """Tests for the intent classification prompt."""

    def test_returns_prompt_template(self):
        """Test that the function returns a PromptTemplate."""
        prompt = get_intent_classification_prompt()
        assert prompt is not None

    def test_has_required_input_variables(self):
        """Test that the prompt has the required input variables."""
        prompt = get_intent_classification_prompt()
        assert "user_input" in prompt.input_variables
        assert "conversation_history" in prompt.input_variables

    def test_format_prompt(self):
        """Test formatting the prompt with values."""
        prompt = get_intent_classification_prompt()
        formatted = prompt.format(
            user_input="What is the total?",
            conversation_history="No prior conversation."
        )
        assert "What is the total?" in formatted
        assert "No prior conversation." in formatted

    def test_prompt_contains_intent_categories(self):
        """Test that the prompt mentions all intent categories."""
        prompt = get_intent_classification_prompt()
        formatted = prompt.format(
            user_input="test",
            conversation_history="none"
        )
        assert "qa" in formatted.lower()
        assert "summarization" in formatted.lower()
        assert "calculation" in formatted.lower()


class TestChatPromptTemplate:
    """Tests for the get_chat_prompt_template function."""

    def test_qa_prompt(self):
        """Test getting the QA prompt template."""
        template = get_chat_prompt_template("qa")
        assert template is not None
        messages = template.messages
        assert len(messages) >= 2

    def test_summarization_prompt(self):
        """Test getting the summarization prompt template."""
        template = get_chat_prompt_template("summarization")
        assert template is not None

    def test_calculation_prompt(self):
        """Test getting the calculation prompt template."""
        template = get_chat_prompt_template("calculation")
        assert template is not None

    def test_unknown_intent_defaults_to_qa(self):
        """Test that unknown intent type defaults to QA prompt."""
        template_qa = get_chat_prompt_template("qa")
        template_unknown = get_chat_prompt_template("unknown")
        assert template_qa is not None
        assert template_unknown is not None

    def test_prompt_can_be_invoked(self):
        """Test that the prompt template can be invoked with input."""
        template = get_chat_prompt_template("qa")
        result = template.invoke({
            "input": "What is the total?",
            "chat_history": []
        })
        assert result is not None

    def test_prompt_has_system_message(self):
        """Test that each prompt contains a system message."""
        for intent_type in ["qa", "summarization", "calculation"]:
            template = get_chat_prompt_template(intent_type)
            messages = template.messages
            first_msg_type = type(messages[0]).__name__
            assert "System" in first_msg_type

    def test_prompt_has_human_message_placeholder(self):
        """Test that prompts include a human message placeholder."""
        template = get_chat_prompt_template("qa")
        messages = template.messages
        msg_types = [type(m).__name__ for m in messages]
        has_human = any("Human" in t for t in msg_types)
        assert has_human


class TestPromptConstants:
    """Tests for prompt constants."""

    def test_qa_system_prompt_not_empty(self):
        """Test that QA_SYSTEM_PROMPT is defined and non-empty."""
        assert QA_SYSTEM_PROMPT
        assert len(QA_SYSTEM_PROMPT) > 0

    def test_summarization_system_prompt_not_empty(self):
        """Test that SUMMARIZATION_SYSTEM_PROMPT is defined and non-empty."""
        assert SUMMARIZATION_SYSTEM_PROMPT
        assert len(SUMMARIZATION_SYSTEM_PROMPT) > 0

    def test_calculation_system_prompt_not_empty(self):
        """Test that CALCULATION_SYSTEM_PROMPT is defined and non-empty."""
        assert CALCULATION_SYSTEM_PROMPT
        assert len(CALCULATION_SYSTEM_PROMPT) > 0

    def test_memory_summary_prompt_not_empty(self):
        """Test that MEMORY_SUMMARY_PROMPT is defined and non-empty."""
        assert MEMORY_SUMMARY_PROMPT
        assert len(MEMORY_SUMMARY_PROMPT) > 0

    def test_calculation_prompt_mentions_calculator_tool(self):
        """Test that the calculation prompt instructs using the calculator tool."""
        assert "calculator" in CALCULATION_SYSTEM_PROMPT.lower()
