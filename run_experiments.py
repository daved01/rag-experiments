from openai_pipeline import OpenAIPipeline
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
    setup_logging()
    config = load_config(ConfigConstants.DEFAULT_CONFIG_FILE)
    prompts_queries = load_prompt_queries(PROMPT_QUERIES_FILE)

    # OpenAI
    openai_pipeline = OpenAIPipeline(config, prompts_queries)
    results_openai: ExperimentResults = openai_pipeline.run_queries()

    # Run with local
    # results_local: Optional[list[str]] = run_local(config, prompt, queries)

    # Save results to a file
    save_experiment_results_to_json(results_openai, config["output"]["directory"])


if __name__ == "__main__":
    main()
