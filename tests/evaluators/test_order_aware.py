import pytest

from evaluators.binary_relevance import OrderAwareMetrics


class TestOrderAware:
    @pytest.fixture
    def order_aware_metrics(self):
        relevant_docs = [["doc1", "doc2", "doc3"], ["doc4", "doc5"]]
        return OrderAwareMetrics(relevant_docs)


class TestReciprocalRank(TestOrderAware):
    def test_reciprocal_rank_no_relevant_docs(self, order_aware_metrics):
        retrieved_docs_all_queries = [["doc4", "doc5", "doc6"]]
        assert order_aware_metrics.reciprocal_rank(retrieved_docs_all_queries, 0) == 0

    def test_reciprocal_rank_empty_retrieved_list(self, order_aware_metrics):
        retrieved_docs_all_queries = [[]]

        assert order_aware_metrics.reciprocal_rank(retrieved_docs_all_queries, 0) == 0

    def test_reciprocal_rank_first_relevant_at_start(self, order_aware_metrics):
        retrieved_docs_all_queries = [["doc1", "doc4", "doc2"]]

        assert order_aware_metrics.reciprocal_rank(retrieved_docs_all_queries, 0) == 1


class TestMeanReciprocalRank(TestOrderAware):
    def test_mean_reciprocal_rank(self, order_aware_metrics):
        retrieved_docs_all_queries = [
            ["doc3", "doc1", "doc4"],  # RR = 1/3
            ["doc5", "doc6", "doc4"],  # RR = 1/1
        ]

        assert (
            order_aware_metrics.mean_reciprocal_rank(retrieved_docs_all_queries)
            == (1 / 3 + 1) / 2
        )

    def test_mean_reciprocal_rank_no_relevant_docs(self, order_aware_metrics):
        retrieved_docs_all_queries = [
            ["doc6", "doc7", "doc8"],  # RR = 0
            ["doc9", "doc10", "doc11"],  # RR = 0
        ]

        assert order_aware_metrics.mean_reciprocal_rank(retrieved_docs_all_queries) == 0

    def test_mean_reciprocal_rank_mixed_relevance(self):
        evaluator = OrderAwareMetrics([["doc1"], ["doc2"], ["doc3"]])
        retrieved_docs_all_queries = [
            ["doc1"],  # RR = 1
            ["doc3", "doc2"],  # RR = 1/2
            ["doc5", "doc3"],  # RR = 1/2
        ]
        assert (
            evaluator.mean_reciprocal_rank(retrieved_docs_all_queries)
            == (1 + 1 / 2 + 1 / 2) / 3
        )

    def test_mean_reciprocal_rank_empty_retrieved_lists(self):
        evaluator = OrderAwareMetrics([["doc1", "doc2"], ["doc3"]])

        retrieved_docs_all_queries = [[], []]  # RR = 0  # RR = 0

        assert evaluator.mean_reciprocal_rank(retrieved_docs_all_queries) == 0

    def test_mean_reciprocal_rank_partial_relevance(self):
        evaluator = OrderAwareMetrics([["doc1", "doc2"], ["doc3"]])

        retrieved_docs_all_queries = [
            ["doc4", "doc1"],  # RR = 1/2
            ["doc3", "doc5"],  # RR = 1
        ]

        assert (
            evaluator.mean_reciprocal_rank(retrieved_docs_all_queries)
            == (1 / 2 + 1) / 2
        )


class TestAveragePrecision(TestOrderAware):
    def test_average_precision_basic(self):
        evaluator = OrderAwareMetrics([["doc1", "doc2", "doc3"]])

        retrieved_docs_all_queries = [["doc3", "doc1", "doc4", "doc2"]]

        assert (
            evaluator.average_precision(retrieved_docs_all_queries, 0)
            == ((1 / 3) + (2 / 4) + (3 / 6)) / 3
        )

    def test_average_precision_no_relevant_docs(self):
        relevant_docs_all_queries = [["doc1", "doc2", "doc3"]]
        evaluator = OrderAwareMetrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc4", "doc5", "doc6"]]

        assert evaluator.average_precision(retrieved_docs_all_queries, 0) == 0.0

    def test_average_precision_all_relevant_docs_first(self):
        relevant_docs_all_queries = [["doc1", "doc2"]]
        evaluator = OrderAwareMetrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc1", "doc2", "doc3"]]

        assert (
            evaluator.average_precision(retrieved_docs_all_queries, 0)
            == (1 / 1 + 2 / 2) / 2
        )

    def test_average_precision_some_relevant_docs(self):
        relevant_docs_all_queries = [["doc1", "doc2", "doc3"]]
        evaluator = OrderAwareMetrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc1", "doc4", "doc2", "doc3"]]

        assert (
            evaluator.average_precision(retrieved_docs_all_queries, 0)
            == (1 / 1 + 2 / 3 + 3 / 4) / 3
        )

    def test_average_precision_empty_retrieved_list(self):
        relevant_docs_all_queries = [["doc1", "doc2"]]
        evaluator = OrderAwareMetrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [[]]

        assert evaluator.average_precision(retrieved_docs_all_queries, 0) == 0.0


class TestMeanAveragePrecision:
    def test_mean_average_precision_basic(self):
        relevant_docs_all_queries = [["doc1", "doc2"], ["doc3", "doc4"]]
        evaluator = OrderAwareMetrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [
            ["doc2", "doc1", "doc3"],  # AP = (1/2 + 2/2) / 2
            ["doc4", "doc3", "doc5"],  # AP = (1/1 + 2/2) / 2
        ]

        expected_map = ((1 / 2 + 2 / 2) / 2 + (1 / 1 + 2 / 2) / 2) / 2
        assert (
            evaluator.mean_average_precision(retrieved_docs_all_queries) == expected_map
        )

    def test_mean_average_precision_no_relevant_docs(self):
        relevant_docs_all_queries = [["doc1"], ["doc2"]]
        evaluator = OrderAwareMetrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [["doc3", "doc4"], ["doc5", "doc6"]]

        assert evaluator.mean_average_precision(retrieved_docs_all_queries) == 0.0

    def test_mean_average_precision_mixed_relevance(self):
        relevant_docs_all_queries = [["doc1", "doc2"], ["doc3"]]
        evaluator = OrderAwareMetrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [
            ["doc1", "doc4", "doc2"],  # AP = (1/1 + 2/3) / 2
            ["doc3", "doc5", "doc6"],  # AP = 1/1
        ]

        expected_map = ((1 / 1 + 2 / 3) / 2 + 1) / 2
        assert (
            evaluator.mean_average_precision(retrieved_docs_all_queries) == expected_map
        )

    def test_mean_average_precision_empty_retrieved_lists(self):
        relevant_docs_all_queries = [["doc1"], ["doc2"]]
        evaluator = OrderAwareMetrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [[], []]

        assert evaluator.mean_average_precision(retrieved_docs_all_queries) == 0.0

    def test_mean_average_precision_partial_relevance(self):
        relevant_docs_all_queries = [["doc1", "doc2"], ["doc3"]]
        evaluator = OrderAwareMetrics(relevant_docs_all_queries)

        retrieved_docs_all_queries = [
            ["doc1", "doc2", "doc3"],  # AP = (1/1 + 2/2) / 2
            ["doc5", "doc3", "doc6"],  # AP = 1/2
        ]

        expected_map = ((1 / 1 + 2 / 2) / 2 + 1 / 2) / 2
        assert (
            evaluator.mean_average_precision(retrieved_docs_all_queries) == expected_map
        )
