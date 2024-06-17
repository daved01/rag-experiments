from abc import ABC
from dataclasses import replace

from evaluators import binary_relevance, graded_relevance
from shared.models import (
    ExperimentResults,
    QueryResult,
)
from shared.constants import ConfigConstants, InputConstants


class BaseEvaluator(ABC):
    def run(self) -> ExperimentResults:
        """Runs the evaluators."""


class OrderUnawareEvaluators(BaseEvaluator):
    """Evaluator for evaluation with order unaware metrics."""

    def __init__(self, config, queries: list[dict]):
        self.relevant_docs: list[list[str]] = [
            [q.get("doc") for q in query.get(InputConstants.KEY_RELEVANT_DOCS)]
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
            order_unaware_metrics = binary_relevance.OrderUnawareMetrics(relevant_doc)
            query_result_with_eval = query_result

            retrieved_documents = query_result.contexts
            precision = order_unaware_metrics.precision_at_k(
                retrieved_documents, self.k
            )
            recall = order_unaware_metrics.recall_at_k(retrieved_documents, self.k)
            f1 = order_unaware_metrics.f1_at_k(retrieved_documents, self.k)

            query_result_with_eval.evaluations = {
                f"precision@{str(self.k)}": precision,
                f"recall@{str(self.k)}": recall,
                f"f1@{str(self.k)}": f1,
            }
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


class OrderAwareEvaluators(BaseEvaluator):
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
        order_aware_metrics = binary_relevance.OrderAwareMetrics(self.relevant_docs)

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


class GradedRelevanceEvaluators(BaseEvaluator):
    """Evaluator for evaluation graded relevance metrics.
    Supported metrics are:

    DCG@K
    NDCG@K

    Attributes:
        config:  Configuration
        queries: A list of query object, one object per query. Each object contains
                 the key `relevant_docs`, which is the ground truth as a list of
                 relevant docs for that query. The relevant docs contain a score
                 `relevance`.
    """

    def __init__(self, config, queries: list[dict]):
        self.relevant_docs: list[list[dict]] = [
            [
                relevant_doc_obj
                for relevant_doc_obj in query.get(InputConstants.KEY_RELEVANT_DOCS)
            ]
            for query in queries
        ]  # relevant_doc_obj has keys `doc`, `relevance`
        self.k = None
        self.config = self._load_config(config)

    def _load_config(self, config: dict) -> None:
        eval_config = config.get(ConfigConstants.KEY_EVALUATORS).get(
            ConfigConstants.KEY_EVALUATORS_ORDER_UNAWARE
        )
        self.k = eval_config.get(ConfigConstants.KEY_EVALUATORS_ORDER_UNAWARE_K)

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
        # order_aware_metrics = binary_relevance.OrderAwareMetrics(self.relevant_docs)
        graded_relevance_metrics = graded_relevance.Metrics(self.relevant_docs)

        # Query-level metrics
        query_results_with_evals = []

        total_dcg_at_k = 0.0
        total_ndcg_at_k = 0.0

        for i in range(num_queries):
            dcg_at_k = graded_relevance_metrics.discounted_cumulutive_gain_at_k(
                retrieved_documents_list, i, self.k
            )
            total_dcg_at_k += dcg_at_k
            ndccg_at_k = (
                graded_relevance_metrics.normalized_discounted_cumulutive_gain_at_k(
                    retrieved_documents_list, i, self.k
                )
            )
            total_ndcg_at_k += ndccg_at_k
            query_evals[i][f"DCG@{str(self.k)}"] = dcg_at_k
            query_evals[i][f"NDCG@{str(self.k)}"] = ndccg_at_k
            updated_query_results_obj = replace(
                query_results_obj[i], evaluations=query_evals[i]
            )
            query_results_with_evals.append(updated_query_results_obj)

        # Overall metrics

        avg_evals[f"avg_DCG@{str(self.k)}"] = total_dcg_at_k / num_queries
        avg_evals[f"avg_NDCG@{str(self.k)}"] = total_ndcg_at_k / num_queries

        # Update ExperimentResults
        results_with_metrics = replace(
            experiment_results,
            results=query_results_with_evals,
            evaluations=avg_evals,
        )

        return results_with_metrics
