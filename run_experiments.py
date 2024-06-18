from openai_pipeline import OpenAIPipeline
from local_pipeline import LocalPipeline
from shared.models import ExperimentResults
from shared.utils import (
    load_config,
    load_prompt_queries,
    save_experiments_results_to_json,
    setup_logging,
)
from shared.constants import ConfigConstants
from evaluators import RetrievalEvaluator


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
    retrieval_evaluators = RetrievalEvaluator(config, prompts_queries)
    results_with_evals_openai = retrieval_evaluators.run(results_openai)

    # Run with local
    print("Running local pipeline ...")
    # local_pipeline = LocalPipeline(config, prompts_queries)
    # results_local: ExperimentResults = local_pipeline.run_queries()
    # results_with_eval_local = evaluator_order_unaware.run(results_local)

    # Save results to a file
    print("Saving results ...")
    save_experiments_results_to_json(
        [results_with_evals_openai],
        config["output"]["directory"],
    )
    print("Done!")


if __name__ == "__main__":
    main()
