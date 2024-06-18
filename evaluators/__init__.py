from shared.models import (
    ExperimentResults,
)
from shared.constants import InputConstants
from evaluators import (
    binary_relevance_order_unaware,
    binary_relevance_order_aware,
    graded_relevance,
)


class RetrievalEvaluator:
    """Defines evaluators to run for evaluating the retriever component.

    Example usage:
        ```
        evaluators = RetrievalEvaluator(config, prompts_queries)
        results_with_evals = evaluators.run(results_without_or_with_evals)
        ```

    Attributes:
        config:         The configuration as a dict.
        prompt_queries: The experiment inputs, including the ground-truths for each query.
    """

    def __init__(self, config: dict, prompts_queries: dict):
        self.config = config
        self.prompts_queries = prompts_queries
        self.evaluator_order_unaware_evaluators = (
            binary_relevance_order_unaware.Evaluator(
                self.config, self.prompts_queries.get(InputConstants.KEY_QUERIES)
            )
        )
        self.evaluator_order_aware_evaluators = binary_relevance_order_aware.Evaluator(
            self.config, self.prompts_queries.get(InputConstants.KEY_QUERIES)
        )
        self.graded_relevance_evaluators = graded_relevance.Evaluator(
            self.config, self.prompts_queries.get(InputConstants.KEY_QUERIES)
        )

    def run(self, results: ExperimentResults) -> ExperimentResults:
        """Runs the evaluators."""
        results_with_eval = self.evaluator_order_unaware_evaluators.run(results)
        results_with_eval = self.evaluator_order_aware_evaluators.run(results_with_eval)
        results_with_eval = self.graded_relevance_evaluators.run(results_with_eval)
        return results_with_eval
