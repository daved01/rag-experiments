from dataclasses import replace

from shared.models import (
    ExperimentResults,
    QueryResult,
)
from shared.constants import InputConstants
from evaluators.base_evaluator import BaseEvaluator


class Metrics:
    """Order Aware Metrics.

    Supported metrics are:
    - RR, MRR
    - AP, MAP

    Attributes:
        relevant_docs_all_queries: List of list of strings representing the
            ground-truth of relevant documents of evaluation.
            The outer list corresponds to the list of queries, while the innner
            list is the number of relevant documents for that query.
    """

    def __init__(self, relevant_docs_all_queries: list[list[str]]):
        self.relevant_docs_all_queries = relevant_docs_all_queries

    def reciprocal_rank(
        self, retrieved_docs_all_queries: list[list[str]], query_index: int
    ) -> float:
        """Caluclates RR.

        Args:
            retrieved_docs_all_queries: list[list[str]]: A list of list of strings,
                                        where each outer list represents a query, and
                                        the inner list the relevant documents for
                                        that query.
            query_index: The index of the query to process in the list of all queries.
        Returns:
            reciprocal_rank: The RR as a float.
        """
        relevant_docs_lookup = set(self.relevant_docs_all_queries[query_index])
        for ind, retrieved_doc in enumerate(retrieved_docs_all_queries[query_index]):
            if retrieved_doc in relevant_docs_lookup:
                return 1 / (1 + ind)
        return 0

    def mean_reciprocal_rank(
        self, retrieved_docs_all_queries: list[list[str]]
    ) -> float:
        """Calculates MRR."""

        num_queries = len(self.relevant_docs_all_queries)

        mrr = 0.0
        for ind in range(num_queries):
            mrr += self.reciprocal_rank(retrieved_docs_all_queries, ind)

        return (1 / num_queries) * mrr

    def average_precision(
        self, retrieved_docs_all_queries: list[list[str]], query_index: int
    ) -> float:
        """Calculates AP."""
        relevant_docs_lookup = set(self.relevant_docs_all_queries[query_index])

        score = 0.0
        num_hits = 0.0
        for ind, retrieved_doc in enumerate(retrieved_docs_all_queries[query_index]):
            if retrieved_doc in relevant_docs_lookup:
                num_hits += 1.0
                score += num_hits / (ind + 1)

        return float(score / len(self.relevant_docs_all_queries[query_index]))

    def mean_average_precision(
        self, retrieved_docs_all_queries: list[list[str]]
    ) -> float:
        """Calculates MAP."""
        num_queries = len(self.relevant_docs_all_queries)

        nominator = 0
        for ind in range(num_queries):
            nominator += self.average_precision(retrieved_docs_all_queries, ind)

        return float(nominator / num_queries)


class Evaluator(BaseEvaluator):
    """Evaluator for evaluation with order aware metrics."""

    def __init__(self, config, queries: list[dict]):
        self.relevant_docs: list[list[str]] = [
            [q.get("doc") for q in query.get(InputConstants.KEY_RELEVANT_DOCS)]
            for query in queries
        ]

    def run(self, experiment_results: ExperimentResults) -> ExperimentResults:
        """Runs evaluation on order aware metrics."""

        avg_evals = experiment_results.evaluations

        query_results_obj: list[QueryResult] = experiment_results.results
        retrieved_documents_list: list[list[str]] = [
            q.contexts for q in query_results_obj
        ]
        query_evals: list[dict] = [q.evaluations for q in query_results_obj]
        num_queries = len(query_results_obj)

        # Calculate metrics
        order_aware_metrics = Metrics(self.relevant_docs)

        # Query-level metrics
        query_results_with_evals = []

        for i in range(num_queries):
            rr = order_aware_metrics.reciprocal_rank(retrieved_documents_list, i)
            ap = order_aware_metrics.average_precision(retrieved_documents_list, i)
            query_evals[i]["RR"] = rr
            query_evals[i]["AP"] = ap
            updated_query_results_obj = replace(
                query_results_obj[i], evaluations=query_evals[i]
            )
            query_results_with_evals.append(updated_query_results_obj)

        # Overall metrics
        mean_reciprocal_rank = order_aware_metrics.mean_reciprocal_rank(
            retrieved_documents_list
        )
        mean_average_precision = order_aware_metrics.mean_average_precision(
            retrieved_documents_list
        )
        avg_evals["MRR"] = mean_reciprocal_rank
        avg_evals["MAP"] = mean_average_precision

        # Update ExperimentResults
        results_with_metrics = replace(
            experiment_results,
            results=query_results_with_evals,
            evaluations=avg_evals,
        )

        return results_with_metrics
