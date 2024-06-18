from abc import ABC

from shared.models import ExperimentResults


class BaseEvaluator(ABC):
    """Abstract Base Class for evaluators."""

    def run(self, results: ExperimentResults) -> ExperimentResults:
        """Runs the evaluators and adds the results to the ExperimentResults object."""
