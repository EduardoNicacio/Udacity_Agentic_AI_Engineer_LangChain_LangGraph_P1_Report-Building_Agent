"""Tests for Pydantic schema definitions in schemas.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from datetime import datetime
from pydantic import ValidationError
from schemas import (
    AnswerResponse, UserIntent, SummarizationResponse,
    CalculationResponse, UpdateMemoryResponse, DocumentChunk, SessionState
)


class TestAnswerResponse:
    """Tests for the AnswerResponse schema."""

    def test_valid_answer_response(self):
        """Test creating a valid AnswerResponse with all fields."""
        response = AnswerResponse(
            question="What is the total?",
            answer="The total is $50,000.",
            sources=["INV-001", "INV-002"],
            confidence=0.95,
            timestamp=datetime.now()
        )
        assert response.question == "What is the total?"
        assert response.answer == "The total is $50,000."
        assert response.sources == ["INV-001", "INV-002"]
        assert response.confidence == 0.95
        assert isinstance(response.timestamp, datetime)

    def test_answer_response_defaults(self):
        """Test AnswerResponse with only required fields uses defaults."""
        response = AnswerResponse(
            question="What is this?",
            answer="This is a document."
        )
        assert response.sources == []
        assert response.confidence == 0.0
        assert isinstance(response.timestamp, datetime)

    def test_answer_response_confidence_bounds(self):
        """Test that confidence is constrained between 0 and 1."""
        response = AnswerResponse(
            question="Q", answer="A", confidence=0.5
        )
        assert response.confidence == 0.5

        with pytest.raises(ValidationError):
            AnswerResponse(question="Q", answer="A", confidence=1.5)

        with pytest.raises(ValidationError):
            AnswerResponse(question="Q", answer="A", confidence=-0.1)

    def test_answer_response_confidence_boundary_values(self):
        """Test confidence at exact boundary values 0 and 1."""
        resp_min = AnswerResponse(question="Q", answer="A", confidence=0.0)
        assert resp_min.confidence == 0.0

        resp_max = AnswerResponse(question="Q", answer="A", confidence=1.0)
        assert resp_max.confidence == 1.0


class TestUserIntent:
    """Tests for the UserIntent schema."""

    def test_valid_user_intent_qa(self):
        """Test creating a valid UserIntent with qa intent type."""
        intent = UserIntent(
            intent_type="qa",
            confidence=0.9,
            reasoning="User is asking a question about documents."
        )
        assert intent.intent_type == "qa"
        assert intent.confidence == 0.9
        assert intent.reasoning == "User is asking a question about documents."

    def test_valid_user_intent_summarization(self):
        """Test creating a valid UserIntent with summarization intent type."""
        intent = UserIntent(
            intent_type="summarization",
            confidence=0.85,
            reasoning="User wants a summary."
        )
        assert intent.intent_type == "summarization"

    def test_valid_user_intent_calculation(self):
        """Test creating a valid UserIntent with calculation intent type."""
        intent = UserIntent(
            intent_type="calculation",
            confidence=0.8,
            reasoning="User needs a calculation."
        )
        assert intent.intent_type == "calculation"

    def test_valid_user_intent_unknown(self):
        """Test creating a valid UserIntent with unknown intent type."""
        intent = UserIntent(
            intent_type="unknown",
            confidence=0.3,
            reasoning="Cannot determine intent."
        )
        assert intent.intent_type == "unknown"

    def test_invalid_intent_type(self):
        """Test that invalid intent_type values are rejected."""
        with pytest.raises(ValidationError):
            UserIntent(intent_type="invalid_type", confidence=0.5, reasoning="test")

    def test_user_intent_confidence_bounds(self):
        """Test confidence constraints on UserIntent."""
        intent = UserIntent(intent_type="qa", confidence=0.0)
        assert intent.confidence == 0.0

        intent = UserIntent(intent_type="qa", confidence=1.0)
        assert intent.confidence == 1.0

        with pytest.raises(ValidationError):
            UserIntent(intent_type="qa", confidence=2.0)

    def test_user_intent_defaults(self):
        """Test UserIntent with only required field uses defaults."""
        intent = UserIntent(intent_type="qa")
        assert intent.confidence == 0.0
        assert intent.reasoning == ""


class TestDocumentChunk:
    """Tests for the DocumentChunk schema."""

    def test_valid_document_chunk(self):
        """Test creating a valid DocumentChunk."""
        chunk = DocumentChunk(
            doc_id="INV-001",
            content="Invoice content here",
            metadata={"client": "Acme"},
            relevance_score=0.95
        )
        assert chunk.doc_id == "INV-001"
        assert chunk.content == "Invoice content here"
        assert chunk.metadata == {"client": "Acme"}
        assert chunk.relevance_score == 0.95

    def test_document_chunk_defaults(self):
        """Test DocumentChunk with defaults."""
        chunk = DocumentChunk(doc_id="DOC-001", content="Some content")
        assert chunk.metadata is not None
        assert chunk.relevance_score == 0.0


class TestSummarizationResponse:
    """Tests for the SummarizationResponse schema."""

    def test_valid_summarization_response(self):
        """Test creating a valid SummarizationResponse."""
        response = SummarizationResponse(
            original_length=1000,
            summary="This is a summary.",
            key_points=["Point 1", "Point 2"],
            document_ids=["DOC-001"]
        )
        assert response.original_length == 1000
        assert response.summary == "This is a summary."
        assert len(response.key_points) == 2
        assert isinstance(response.timestamp, datetime)


class TestCalculationResponse:
    """Tests for the CalculationResponse schema."""

    def test_valid_calculation_response(self):
        """Test creating a valid CalculationResponse."""
        response = CalculationResponse(
            expression="2 + 3",
            result=5.0,
            explanation="Added 2 and 3 to get 5."
        )
        assert response.expression == "2 + 3"
        assert response.result == 5.0
        assert response.units is None
        assert isinstance(response.timestamp, datetime)


class TestUpdateMemoryResponse:
    """Tests for the UpdateMemoryResponse schema."""

    def test_valid_update_memory_response(self):
        """Test creating a valid UpdateMemoryResponse."""
        response = UpdateMemoryResponse(
            summary="Conversation about invoices.",
            document_ids=["INV-001", "INV-002"]
        )
        assert response.summary == "Conversation about invoices."
        assert len(response.document_ids) == 2

    def test_update_memory_response_defaults(self):
        """Test UpdateMemoryResponse with default document_ids."""
        response = UpdateMemoryResponse(summary="Test summary")
        assert response.document_ids is not None


class TestSessionState:
    """Tests for the SessionState schema."""

    def test_valid_session_state(self):
        """Test creating a valid SessionState."""
        session = SessionState(
            session_id="sess-123",
            user_id="user-456"
        )
        assert session.session_id == "sess-123"
        assert session.user_id == "user-456"
        assert session.conversation_history is not None
        assert session.document_context is not None
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.last_updated, datetime)
