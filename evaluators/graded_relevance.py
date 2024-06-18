import math
from dataclasses import replace

from shared.models import (
    ExperimentResults,
    QueryResult,
)
from shared.constants import ConfigConstants, InputConstants
from evaluators.base_evaluator import BaseEvaluator


class Metrics:
    """Graded Relevance Metrics.

    Supported metrics are:
    - DCG@K
    - NDCG@K

    Attributes:
        relevant_docs_all_queries: List of list of dicts representing the queries, including the
        ground-truths `relevant_docs` with a relevance score.
    """

    def __init__(self, relevant_docs_all_queries: list[list[dict]]):
        self.relevant_docs = relevant_docs_all_queries
        self.num_relevant_docs = len(self.relevant_docs)

    def discounted_cumulative_gain_at_k(
        self, retrieved_docs_all_queries: list[list[str]], query_index: int, k: int
    ):
        """Calculates DCG@k.

        Args:
            retrieved_docs_all_queries: A list of list of query stings.
            query_index: The index of the query for which DCG@k should be
                         calculated.
            k: Number up to which DCG@k should be calculated.
        Returns:
            DCG@k
        """
        retrieved_docs: list[str] = retrieved_docs_all_queries[query_index]
        relevant_docs: list[dict] = self.relevant_docs[query_index]

        dcg = 0.0
        for i in range(min(k, len(retrieved_docs))):
            retrieved_doc = retrieved_docs[i]
            relevance_score = 0
            for rel_doc in relevant_docs:
                if rel_doc[InputConstants.KEY_DOC] == retrieved_doc:
                    relevance_score = rel_doc[InputConstants.KEY_RELEVANCE]
                    break
            dcg += relevance_score / math.log2(i + 2)

        return dcg

    def normalized_discounted_cumulative_gain_at_k(
        self, retrieved_docs_all_queries: list[list[str]], query_index: int, k: int
    ):
        """Calculated NDCG@k.

        Args:
            retrieved_docs_all_queries: A list of list of query stings.
            query_index: The index of the query for which DCG@k should be
                         calculated.
            k: Number up to which DCG@k should be calculated.

        Returns:
            NDCG@k
        """

        relevant_docs: list[dict] = self.relevant_docs[query_index]

        # Sort relevance scores
        relevance_scores = [doc[InputConstants.KEY_RELEVANCE] for doc in relevant_docs]
        relevance_scores.sort(reverse=True)

        # Calculate IDCG@k
        idcg_at_k = 0.0
        for i in range(min(k, len(relevance_scores))):
            idcg_at_k += relevance_scores[i] / math.log2(i + 2)

        # Calculate DCG@k
        dcg_at_k = self.discounted_cumulative_gain_at_k(
            retrieved_docs_all_queries, query_index, k
        )

        # Handle cases where IDCG is zero to avoid division by zero
        if idcg_at_k == 0:
            return 0.0

        return dcg_at_k / idcg_at_k


class Evaluator(BaseEvaluator):
    """Evaluator for evaluation graded relevance metrics.
    Supported metrics are:

    - DCG@K
    - NDCG@K

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
        graded_relevance_metrics = Metrics(self.relevant_docs)

        # Query-level metrics
        query_results_with_evals = []

        total_dcg_at_k = 0.0
        total_ndcg_at_k = 0.0

        for i in range(num_queries):
            dcg_at_k = graded_relevance_metrics.discounted_cumulative_gain_at_k(
                retrieved_documents_list, i, self.k
            )
            total_dcg_at_k += dcg_at_k
            ndccg_at_k = (
                graded_relevance_metrics.normalized_discounted_cumulative_gain_at_k(
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
