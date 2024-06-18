import math

from evaluators.graded_relevance import Metrics


class TestDiscountedCumulativeGainAtK:
    def test_discounted_cumulative_gain_at_k_basic(self):
        relevant_docs_all_queries = [
            [{"doc": "doc1", "relevance": 3}, {"doc": "doc2", "relevance": 2}]
        ]
        evaluator = Metrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc1", "doc2", "doc3"]]

        expected_dcg = (3 / math.log2(1 + 1)) + (2 / math.log2(2 + 1))
        assert (
            evaluator.discounted_cumulative_gain_at_k(retrieved_docs_all_queries, 0, 2)
            == expected_dcg
        )

    def test_discounted_cumulative_gain_at_k_no_relevant_docs(self):
        relevant_docs_all_queries = [
            [{"doc": "doc1", "relevance": 3}, {"doc": "doc2", "relevance": 2}]
        ]
        evaluator = Metrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc3", "doc4", "doc5"]]

        assert (
            evaluator.discounted_cumulative_gain_at_k(retrieved_docs_all_queries, 0, 2)
            == 0.0
        )

    def test_discounted_cumulative_gain_at_k_partial_relevance(self):
        relevant_docs_all_queries = [
            [{"doc": "doc1", "relevance": 3}, {"doc": "doc2", "relevance": 2}]
        ]
        evaluator = Metrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc2", "doc3", "doc1"]]

        expected_dcg = (2 / math.log2(1 + 1)) + (3 / math.log2(3 + 1))
        assert (
            evaluator.discounted_cumulative_gain_at_k(retrieved_docs_all_queries, 0, 3)
            == expected_dcg
        )

    def test_discounted_cumulative_gain_at_k_k_greater_than_retrieved_docs(self):
        relevant_docs_all_queries = [
            [{"doc": "doc1", "relevance": 3}, {"doc": "doc2", "relevance": 2}]
        ]
        evaluator = Metrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc1", "doc2"]]

        expected_dcg = (3 / math.log2(1 + 1)) + (2 / math.log2(2 + 1))
        assert (
            evaluator.discounted_cumulative_gain_at_k(retrieved_docs_all_queries, 0, 5)
            == expected_dcg
        )

    def test_discounted_cumulative_gain_at_k_k_equal_zero(self):
        relevant_docs_all_queries = [
            [{"doc": "doc1", "relevance": 3}, {"doc": "doc2", "relevance": 2}]
        ]
        evaluator = Metrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc1", "doc2", "doc3"]]

        assert (
            evaluator.discounted_cumulative_gain_at_k(retrieved_docs_all_queries, 0, 0)
            == 0.0
        )


class TestNormalizedDiscountedCumulativeGainAtK:
    def test_discounted_cumulative_gain_at_k_basic(self):
        relevant_docs_all_queries = [
            [{"doc": "doc1", "relevance": 3}, {"doc": "doc2", "relevance": 2}]
        ]
        evaluator = Metrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc1", "doc2", "doc3"]]

        expected_dcg = (3 / math.log2(1 + 1)) + (2 / math.log2(2 + 1))
        assert (
            evaluator.discounted_cumulative_gain_at_k(retrieved_docs_all_queries, 0, 2)
            == expected_dcg
        )

    def test_discounted_cumulative_gain_at_k_no_relevant_docs(self):
        relevant_docs_all_queries = [
            [{"doc": "doc1", "relevance": 3}, {"doc": "doc2", "relevance": 2}]
        ]
        evaluator = Metrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc3", "doc4", "doc5"]]

        assert (
            evaluator.discounted_cumulative_gain_at_k(retrieved_docs_all_queries, 0, 2)
            == 0.0
        )

    def test_discounted_cumulative_gain_at_k_partial_relevance(self):
        relevant_docs_all_queries = [
            [{"doc": "doc1", "relevance": 3}, {"doc": "doc2", "relevance": 2}]
        ]
        evaluator = Metrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc2", "doc3", "doc1"]]

        expected_dcg = (2 / math.log2(1 + 1)) + (3 / math.log2(3 + 1))
        assert (
            evaluator.discounted_cumulative_gain_at_k(retrieved_docs_all_queries, 0, 3)
            == expected_dcg
        )

    def test_discounted_cumulative_gain_at_k_k_greater_than_retrieved_docs(self):
        relevant_docs_all_queries = [
            [{"doc": "doc1", "relevance": 3}, {"doc": "doc2", "relevance": 2}]
        ]
        evaluator = Metrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc1", "doc2"]]

        expected_dcg = (3 / math.log2(1 + 1)) + (2 / math.log2(2 + 1))
        assert (
            evaluator.discounted_cumulative_gain_at_k(retrieved_docs_all_queries, 0, 5)
            == expected_dcg
        )

    def test_discounted_cumulative_gain_at_k_k_equal_zero(self):
        relevant_docs_all_queries = [
            [{"doc": "doc1", "relevance": 3}, {"doc": "doc2", "relevance": 2}]
        ]
        evaluator = Metrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc1", "doc2", "doc3"]]

        assert (
            evaluator.discounted_cumulative_gain_at_k(retrieved_docs_all_queries, 0, 0)
            == 0.0
        )

    def test_normalized_discounted_cumulative_gain_at_k_basic(self):
        relevant_docs_all_queries = [
            [{"doc": "doc1", "relevance": 3}, {"doc": "doc2", "relevance": 2}]
        ]
        evaluator = Metrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc1", "doc2", "doc3"]]

        expected_dcg = (3 / math.log2(1 + 1)) + (2 / math.log2(2 + 1))
        expected_idcg = (3 / math.log2(1 + 1)) + (2 / math.log2(2 + 1))
        expected_ndcg = expected_dcg / expected_idcg
        assert (
            evaluator.normalized_discounted_cumulative_gain_at_k(
                retrieved_docs_all_queries, 0, 2
            )
            == expected_ndcg
        )

    def test_normalized_discounted_cumulative_gain_at_k_no_relevant_docs(self):
        relevant_docs_all_queries = [
            [{"doc": "doc1", "relevance": 3}, {"doc": "doc2", "relevance": 2}]
        ]
        evaluator = Metrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc3", "doc4", "doc5"]]

        assert (
            evaluator.normalized_discounted_cumulative_gain_at_k(
                retrieved_docs_all_queries, 0, 2
            )
            == 0.0
        )

    def test_normalized_discounted_cumulative_gain_at_k_partial_relevance(self):
        relevant_docs_all_queries = [
            [{"doc": "doc1", "relevance": 3}, {"doc": "doc2", "relevance": 2}]
        ]
        evaluator = Metrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc2", "doc3", "doc1"]]

        expected_dcg = (2 / math.log2(1 + 1)) + (3 / math.log2(3 + 1))
        expected_idcg = (3 / math.log2(1 + 1)) + (2 / math.log2(2 + 1))
        expected_ndcg = expected_dcg / expected_idcg
        assert (
            evaluator.normalized_discounted_cumulative_gain_at_k(
                retrieved_docs_all_queries, 0, 3
            )
            == expected_ndcg
        )

    def test_normalized_discounted_cumulative_gain_at_k_k_greater_than_retrieved_docs(
        self,
    ):
        relevant_docs_all_queries = [
            [{"doc": "doc1", "relevance": 3}, {"doc": "doc2", "relevance": 2}]
        ]
        evaluator = Metrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc1", "doc2"]]

        expected_dcg = (3 / math.log2(1 + 1)) + (2 / math.log2(2 + 1))
        expected_idcg = (3 / math.log2(1 + 1)) + (2 / math.log2(2 + 1))
        expected_ndcg = expected_dcg / expected_idcg
        assert (
            evaluator.normalized_discounted_cumulative_gain_at_k(
                retrieved_docs_all_queries, 0, 5
            )
            == expected_ndcg
        )

    def test_normalized_discounted_cumulative_gain_at_k_k_equal_zero(self):
        relevant_docs_all_queries = [
            [{"doc": "doc1", "relevance": 3}, {"doc": "doc2", "relevance": 2}]
        ]
        evaluator = Metrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc1", "doc2", "doc3"]]

        assert (
            evaluator.normalized_discounted_cumulative_gain_at_k(
                retrieved_docs_all_queries, 0, 0
            )
            == 0.0
        )
