from dataclasses import replace

from shared.models import (
    ExperimentResults,
    QueryResult,
)
from shared.constants import ConfigConstants, InputConstants
from evaluators.base_evaluator import BaseEvaluator


class Metrics:
    """Order Unaware Metrics.

    Supported metrics are:
    - precision@k
    - recall@k
    - f1@k

    Attributes:
        relevant_docs: List of strings representing the ground-truth of
        relevant documents of evaluation.
    """

    def __init__(self, relevant_docs: list[str]):
        self.relevant_docs = relevant_docs

    def precision_at_k(self, retrieved_docs: list[str], k: int) -> float:
        """Calculates Precision@k."""
        retrieved_docs_at_k = retrieved_docs[:k]
        relevant_count = sum(
            [1 for doc in retrieved_docs_at_k if doc in self.relevant_docs]
        )
        return float(relevant_count / k)

    def recall_at_k(self, retrieved_docs: list[str], k: int) -> float:
        """Calculates Recall@k."""
        retrieved_docs_at_k = retrieved_docs[:k]
        relevant_count = sum(
            [1 for doc in retrieved_docs_at_k if doc in self.relevant_docs]
        )
        return float(relevant_count / len(self.relevant_docs))

    def f1_at_k(self, retrieved_docs: list[str], k: int) -> float:
        """Calculates F1@k."""
        precision = self.precision_at_k(retrieved_docs, k)
        recall = self.recall_at_k(retrieved_docs, k)
        if precision + recall == 0:
            return 0.0
        return float(2 * (precision * recall) / (precision + recall))


class Evaluator(BaseEvaluator):
    """Evaluator for evaluation with order unaware metrics."""

    def __init__(self, config, queries: list[dict]):
        self.relevant_docs: list[list[str]] = [
            [
                q.get(InputConstants.KEY_DOC)
                for q in query.get(InputConstants.KEY_RELEVANT_DOCS)
            ]
            for query in queries
        ]
        self.k = None
        self.config = self._load_config(config)

    def _load_config(self, config: dict) -> None:
        eval_config = config.get(ConfigConstants.KEY_EVALUATORS).get(
            ConfigConstants.KEY_EVALUATORS_ORDER_UNAWARE
        )
        self.k = eval_config.get(ConfigConstants.KEY_EVALUATORS_ORDER_UNAWARE_K)

    def run(self, experiment_results: ExperimentResults) -> ExperimentResults:
        """Runs evaluation on order unaware metrics."""

        query_results: list[QueryResult] = experiment_results.results
        query_results_with_evals = []

        # Calculate metrics for each query
        for relevant_doc, query_result in zip(self.relevant_docs, query_results):
            order_unaware_metrics = Metrics(relevant_doc)
            query_result_with_eval = query_result
            if not query_result_with_eval.evaluations:
                query_result_with_eval.evaluations = {}

            retrieved_documents = query_result.contexts
            query_result_with_eval.evaluations[f"precision@{str(self.k)}"] = (
                order_unaware_metrics.precision_at_k(retrieved_documents, self.k)
            )
            query_result_with_eval.evaluations[f"recall@{str(self.k)}"] = (
                order_unaware_metrics.recall_at_k(retrieved_documents, self.k)
            )
            query_result_with_eval.evaluations[f"f1@{str(self.k)}"] = (
                order_unaware_metrics.f1_at_k(retrieved_documents, self.k)
            )
            query_results_with_evals.append(query_result_with_eval)

        # Calculate average metrics over all queries.
        n = len(query_results_with_evals)

        avg_precision = (
            sum(
                [
                    q.evaluations.get(f"precision@{str(self.k)}")
                    for q in query_results_with_evals
                ]
            )
            / n
        )
        avg_recall = (
            sum(
                [
                    q.evaluations.get(f"recall@{str(self.k)}")
                    for q in query_results_with_evals
                ]
            )
            / n
        )
        avg_f1 = (
            sum(
                [
                    q.evaluations.get(f"f1@{str(self.k)}")
                    for q in query_results_with_evals
                ]
            )
            / n
        )

        avg_evals = {
            f"avg_precision@{str(self.k)}": avg_precision,
            f"avg_recall@{str(self.k)}": avg_recall,
            f"avg_f1@{str(self.k)}": avg_f1,
        }

        results_with_metrics = replace(
            experiment_results,
            results=query_results_with_evals,
            evaluations=avg_evals,
        )

        return results_with_metrics
