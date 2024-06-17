from openai_pipeline import OpenAIPipeline
from local_pipeline import LocalPipeline
from shared.models import ExperimentResults
from shared.utils import (
    load_config,
    load_prompt_queries,
    save_experiments_results_to_json,
    setup_logging,
)
from shared.constants import ConfigConstants, InputConstants
from evaluators import (
    OrderUnawareEvaluators,
    OrderAwareEvaluators,
    GradedRelevanceEvaluators,
)


PROMPT_QUERIES_FILE = "prompts_queries.json"


def main():
    print("Running experiments ...")
    setup_logging()
    config = load_config(ConfigConstants.DEFAULT_CONFIG_FILE)
    prompts_queries = load_prompt_queries(PROMPT_QUERIES_FILE)

    # OpenAI
    print("Running OpenAI pipeline ...")
    openai_pipeline = OpenAIPipeline(config, prompts_queries)
    results_openai: ExperimentResults = openai_pipeline.run_queries()

    print("Evaluating results ...")
    evaluator_order_unaware_evaluators = OrderUnawareEvaluators(
        config, prompts_queries.get(InputConstants.KEY_QUERIES)
    )
    # TODO: Modify so that they accept existing evals
    results_with_eval_openai = evaluator_order_unaware_evaluators.run(results_openai)

    evaluator_order_aware_evaluators = OrderAwareEvaluators(
        config, prompts_queries.get(InputConstants.KEY_QUERIES)
    )
    results_with_eval_openai = evaluator_order_aware_evaluators.run(
        results_with_eval_openai
    )

    graded_relevance_evaluators = GradedRelevanceEvaluators(
        config, prompts_queries.get(InputConstants.KEY_QUERIES)
    )
    results_with_eval_openai = graded_relevance_evaluators.run(results_with_eval_openai)

    # Run with local
    print("Running local pipeline ...")
    # local_pipeline = LocalPipeline(config, prompts_queries)
    # results_local: ExperimentResults = local_pipeline.run_queries()
    # results_with_eval_local = evaluator_order_unaware.run(results_local)

    # Save results to a file
    print("Saving results ...")
    save_experiments_results_to_json(
        [results_with_eval_openai],
        config["output"]["directory"],
    )
    print("Done!")


if __name__ == "__main__":
    main()
