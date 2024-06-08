from openai_pipeline import OpenAIPipeline
from local_pipeline import LocalPipeline
from shared.models import ExperimentResults
from shared.utils import (
    load_config,
    load_prompt_queries,
    save_experiment_results_to_json,
    setup_logging,
)
from shared.constants import ConfigConstants


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

    # Run with local
    print("Running local pipeline ...")
    local_pipeline = LocalPipeline(config, prompts_queries)
    results_local: ExperimentResults = local_pipeline.run_queries()

    # Save results to a file
    print("Saving results ...")
    save_experiment_results_to_json(results_openai, config["output"]["directory"])
    save_experiment_results_to_json(results_local, config["output"]["directory"])
    print("Done!")


if __name__ == "__main__":
    main()
