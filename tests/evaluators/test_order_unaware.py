import pytest

from evaluators.binary_relevance_order_unaware import Metrics


class TestOrderUnaware:
    @pytest.fixture
    def order_unaware_metrics(self):
        relevant_docs = ["doc1", "doc2", "doc3"]
        return Metrics(relevant_docs)


class TestPrecisionAtK(TestOrderUnaware):
    def test_precision_at_k_all_relevant(self, order_unaware_metrics):
        retrieved_docs = ["doc1", "doc2", "doc3"]
        k = 3
        assert order_unaware_metrics.precision_at_k(retrieved_docs, k) == 1.0

    def test_precision_at_k_some_relevant(self, order_unaware_metrics):
        retrieved_docs = ["doc1", "doc4", "doc3"]
        k = 3
        assert order_unaware_metrics.precision_at_k(retrieved_docs, k) == 2 / 3

    def test_precision_at_k_none_relevant(self, order_unaware_metrics):
        retrieved_docs = ["doc4", "doc5", "doc6"]
        k = 3
        assert order_unaware_metrics.precision_at_k(retrieved_docs, k) == 0.0

    def test_precision_at_k_more_than_k(self, order_unaware_metrics):
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        k = 3
        assert order_unaware_metrics.precision_at_k(retrieved_docs, k) == 1.0

    def test_precision_at_k_less_than_k(self, order_unaware_metrics):
        retrieved_docs = ["doc1", "doc4"]
        k = 3
        assert order_unaware_metrics.precision_at_k(retrieved_docs, k) == 1 / 3

    def test_precision_at_k_empty_retrieval(self, order_unaware_metrics):
        retrieved_docs = []
        k = 3
        assert order_unaware_metrics.precision_at_k(retrieved_docs, k) == 0.0


class TestRecallAtK(TestOrderUnaware):
    def test_recall_at_k_all_relevant(self, order_unaware_metrics):
        retrieved_docs = ["doc1", "doc2", "doc3"]
        k = 3
        assert order_unaware_metrics.recall_at_k(retrieved_docs, k) == 1.0

    def test_recall_at_k_some_relevant(self, order_unaware_metrics):
        retrieved_docs = ["doc1", "doc4", "doc3"]
        k = 3
        assert order_unaware_metrics.recall_at_k(retrieved_docs, k) == 2 / 3

    def test_recall_at_k_none_relevant(self, order_unaware_metrics):
        retrieved_docs = ["doc4", "doc5", "doc6"]
        k = 3
        assert order_unaware_metrics.recall_at_k(retrieved_docs, k) == 0.0

    def test_recall_at_k_more_than_k(self, order_unaware_metrics):
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        k = 3
        assert order_unaware_metrics.recall_at_k(retrieved_docs, k) == 1.0

    def test_recall_at_k_less_than_k(self, order_unaware_metrics):
        retrieved_docs = ["doc1", "doc4"]
        k = 3
        assert order_unaware_metrics.recall_at_k(retrieved_docs, k) == 1 / 3

    def test_recall_at_k_k_greater_than_retrieved(self, order_unaware_metrics):
        retrieved_docs = ["doc1", "doc2"]
        k = 5
        assert order_unaware_metrics.recall_at_k(retrieved_docs, k) == 2 / 3

    def test_recall_at_k_empty_retrieval(self, order_unaware_metrics):
        retrieved_docs = []
        k = 3
        assert order_unaware_metrics.recall_at_k(retrieved_docs, k) == 0.0


class TestF1AtK(TestOrderUnaware):
    def test_f1_at_k_all_relevant(self, order_unaware_metrics):
        retrieved_docs = ["doc1", "doc2", "doc3"]
        k = 3
        assert order_unaware_metrics.f1_at_k(retrieved_docs, k) == 1.0

    def test_f1_at_k_some_relevant(self, order_unaware_metrics):
        retrieved_docs = ["doc1", "doc4", "doc3"]
        k = 3
        precision = order_unaware_metrics.precision_at_k(retrieved_docs, k)
        recall = order_unaware_metrics.recall_at_k(retrieved_docs, k)
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        assert order_unaware_metrics.f1_at_k(retrieved_docs, k) == expected_f1

    def test_f1_at_k_none_relevant(self, order_unaware_metrics):
        retrieved_docs = ["doc4", "doc5", "doc6"]
        k = 3
        assert order_unaware_metrics.f1_at_k(retrieved_docs, k) == 0.0

    def test_f1_at_k_more_than_k(self, order_unaware_metrics):
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        k = 3
        assert order_unaware_metrics.f1_at_k(retrieved_docs, k) == 1.0

    def test_f1_at_k_less_than_k(self, order_unaware_metrics):
        retrieved_docs = ["doc1", "doc4"]
        k = 3
        precision = order_unaware_metrics.precision_at_k(retrieved_docs, k)
        recall = order_unaware_metrics.recall_at_k(retrieved_docs, k)
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        assert order_unaware_metrics.f1_at_k(retrieved_docs, k) == expected_f1

    def test_f1_at_k_k_greater_than_retrieved(self, order_unaware_metrics):
        retrieved_docs = ["doc1", "doc2"]
        k = 5
        precision = order_unaware_metrics.precision_at_k(retrieved_docs, k)
        recall = order_unaware_metrics.recall_at_k(retrieved_docs, k)
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        assert order_unaware_metrics.f1_at_k(retrieved_docs, k) == expected_f1

    def test_f1_at_k_division_zero_attempt(self, order_unaware_metrics):
        retrieved_docs = []
        k = 2
        expected_f1 = 0.0
        assert order_unaware_metrics.f1_at_k(retrieved_docs, k) == expected_f1
