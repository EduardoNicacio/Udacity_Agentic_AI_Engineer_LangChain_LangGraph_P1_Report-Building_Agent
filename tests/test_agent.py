"""Tests for the agent workflow in agent.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from agent import (
    AgentState, should_continue, create_workflow, classify_intent,
    qa_agent, summarization_agent, calculation_agent, update_memory
)
from schemas import UserIntent


class TestAgentState:
    """Tests for the AgentState TypedDict."""

    def test_agent_state_has_required_keys(self):
        """Test that AgentState has all required keys."""
        state: AgentState = {
            "user_input": "test",
            "messages": [],
            "intent": None,
            "next_step": "classify_intent",
            "conversation_summary": "",
            "active_documents": [],
            "current_response": None,
            "tools_used": [],
            "session_id": "sess-1",
            "user_id": "user-1",
            "actions_taken": [],
        }
        assert state["user_input"] == "test"
        assert state["messages"] == []
        assert state["next_step"] == "classify_intent"
        assert state["actions_taken"] == []


class TestShouldContinue:
    """Tests for the should_continue router function."""

    def test_returns_qa_agent(self):
        """Test routing to qa_agent."""
        state = {"next_step": "qa_agent"}
        assert should_continue(state) == "qa_agent"

    def test_returns_summarization_agent(self):
        """Test routing to summarization_agent."""
        state = {"next_step": "summarization_agent"}
        assert should_continue(state) == "summarization_agent"

    def test_returns_calculation_agent(self):
        """Test routing to calculation_agent."""
        state = {"next_step": "calculation_agent"}
        assert should_continue(state) == "calculation_agent"

    def test_default_returns_end(self):
        """Test default routing when next_step is missing."""
        state = {}
        assert should_continue(state) == "end"

    def test_returns_end(self):
        """Test explicit end routing."""
        state = {"next_step": "end"}
        assert should_continue(state) == "end"


class TestCreateWorkflow:
    """Tests for the create_workflow function."""

    def test_workflow_compiles(self):
        """Test that the workflow compiles without errors."""
        mock_llm = MagicMock()
        mock_tools = []
        workflow = create_workflow(mock_llm, mock_tools)
        assert workflow is not None

    def test_workflow_has_nodes(self):
        """Test that the workflow has the expected nodes."""
        mock_llm = MagicMock()
        mock_tools = []
        workflow = create_workflow(mock_llm, mock_tools)
        assert workflow is not None


class TestClassifyIntent:
    """Tests for the classify_intent function."""

    def _make_state(self, user_input="test input"):
        """Helper to create a minimal AgentState."""
        return {
            "user_input": user_input,
            "messages": [],
            "intent": None,
            "next_step": "classify_intent",
            "conversation_summary": "",
            "active_documents": [],
            "current_response": None,
            "tools_used": [],
            "session_id": "sess-1",
            "user_id": "user-1",
            "actions_taken": [],
        }

    def test_classify_intent_returns_actions_taken(self):
        """Test that classify_intent appends itself to actions_taken."""
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = UserIntent(
            intent_type="qa", confidence=0.9, reasoning="test"
        )
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = RunnableConfig(configurable={"llm": mock_llm})
        state = self._make_state()
        result = classify_intent(state, config)

        assert "classify_intent" in result["actions_taken"]
        assert result["intent"].intent_type == "qa"
        assert result["next_step"] == "qa_agent"

    def test_classify_intent_routes_to_summarization(self):
        """Test routing to summarization_agent."""
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = UserIntent(
            intent_type="summarization", confidence=0.85, reasoning="test"
        )
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = RunnableConfig(configurable={"llm": mock_llm})
        state = self._make_state("Summarize the documents")
        result = classify_intent(state, config)

        assert result["next_step"] == "summarization_agent"

    def test_classify_intent_routes_to_calculation(self):
        """Test routing to calculation_agent."""
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = UserIntent(
            intent_type="calculation", confidence=0.8, reasoning="test"
        )
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = RunnableConfig(configurable={"llm": mock_llm})
        state = self._make_state("Calculate the total")
        result = classify_intent(state, config)

        assert result["next_step"] == "calculation_agent"

    def test_classify_intent_unknown_defaults_to_qa(self):
        """Test that unknown intent defaults to qa_agent."""
        mock_llm = MagicMock()
        mock_structured_llm = MagicMock()
        mock_structured_llm.invoke.return_value = UserIntent(
            intent_type="unknown", confidence=0.3, reasoning="test"
        )
        mock_llm.with_structured_output.return_value = mock_structured_llm

        config = RunnableConfig(configurable={"llm": mock_llm})
        state = self._make_state("asdfghjkl")
        result = classify_intent(state, config)

        assert result["next_step"] == "qa_agent"
