"""Tests for document retrieval in retrieval.py."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from retrieval import SimulatedRetriever, Document


@pytest.fixture
def retriever():
    """Create a SimulatedRetriever instance for testing."""
    return SimulatedRetriever()


class TestSimulatedRetriever:
    """Tests for the SimulatedRetriever class."""

    def test_initialization(self, retriever):
        """Test that the retriever initializes with sample documents."""
        assert len(retriever.documents) > 0

    def test_sample_documents_loaded(self, retriever):
        """Test that sample documents are loaded correctly."""
        assert "INV-001" in retriever.documents
        assert "CON-001" in retriever.documents
        assert "CLM-001" in retriever.documents

    def test_retrieve_all(self, retriever):
        """Test retrieving all documents."""
        results = retriever.retrieve_all()
        assert len(results) == len(retriever.documents)
        for chunk in results:
            assert chunk.doc_id is not None
            assert chunk.content is not None

    def test_retrieve_by_keyword(self, retriever):
        """Test retrieving documents by keyword."""
        results = retriever.retrieve_by_keyword("invoice")
        assert len(results) > 0
        assert any("INV" in r.doc_id for r in results)

    def test_retrieve_by_keyword_no_results(self, retriever):
        """Test keyword search with no matching results."""
        results = retriever.retrieve_by_keyword("xyznonexistent")
        assert len(results) == 0

    def test_retrieve_by_type_invoice(self, retriever):
        """Test retrieving documents by type - invoice."""
        results = retriever.retrieve_by_type("invoice")
        assert len(results) > 0
        for r in results:
            assert r.metadata.get("doc_type") == "invoice"

    def test_retrieve_by_type_contract(self, retriever):
        """Test retrieving documents by type - contract."""
        results = retriever.retrieve_by_type("contract")
        assert len(results) > 0
        for r in results:
            assert r.metadata.get("doc_type") == "contract"

    def test_retrieve_by_type_claim(self, retriever):
        """Test retrieving documents by type - claim."""
        results = retriever.retrieve_by_type("claim")
        assert len(results) > 0
        for r in results:
            assert r.metadata.get("doc_type") == "claim"

    def test_retrieve_by_amount_range(self, retriever):
        """Test retrieving documents within an amount range."""
        results = retriever.retrieve_by_amount_range(min_amount=50000, max_amount=100000)
        assert len(results) > 0

    def test_retrieve_by_amount_range_no_min(self, retriever):
        """Test retrieving documents above a minimum amount."""
        results = retriever.retrieve_by_amount_range(min_amount=100000)
        assert len(results) > 0

    def test_retrieve_by_amount_range_no_max(self, retriever):
        """Test retrieving documents below a maximum amount."""
        results = retriever.retrieve_by_amount_range(max_amount=5000)
        assert len(results) > 0

    def test_retrieve_by_exact_amount(self, retriever):
        """Test retrieving documents with an exact amount."""
        results = retriever.retrieve_by_exact_amount(2450)
        assert len(results) > 0
        assert any("CLM" in r.doc_id for r in results)

    def test_retrieve_by_approximate_amount(self, retriever):
        """Test retrieving documents with approximately matching amounts."""
        results = retriever.retrieve_by_approximate_amount(2500, percentage=10)
        assert len(results) > 0

    def test_get_document_by_id(self, retriever):
        """Test retrieving a specific document by ID."""
        doc = retriever.get_document_by_id("INV-001")
        assert doc is not None
        assert doc.doc_id == "INV-001"
        assert doc.content is not None

    def test_get_document_by_id_not_found(self, retriever):
        """Test retrieving a document with non-existent ID."""
        doc = retriever.get_document_by_id("NON-EXISTENT")
        assert doc is None

    def test_get_statistics(self, retriever):
        """Test getting document collection statistics."""
        stats = retriever.get_statistics()
        assert "total_documents" in stats
        assert "document_types" in stats
        assert stats["total_documents"] > 0

    def test_add_document(self, retriever):
        """Test adding a new document to the retriever."""
        new_doc = Document(
            doc_id="TEST-001",
            title="Test Document",
            content="Test content",
            doc_type="test",
            metadata={"amount": 1000}
        )
        retriever.add_document(new_doc)
        assert "TEST-001" in retriever.documents

    def test_retrieve_by_keyword_top_k(self, retriever):
        """Test that keyword retrieval respects top_k parameter."""
        results = retriever.retrieve_by_keyword("invoice", top_k=2)
        assert len(results) <= 2
