from openai_pipeline import OpenAIPipeline
from shared.models import ExperimentResults
from shared.utils import load_config, load_prompt_queries
from shared.constants import ConfigConstants


PROMPT_QUERIES_FILE = "prompts_queries.json"


def main():
    config = load_config(ConfigConstants.DEFAULT_CONFIG_FILE)
    prompts_queries = load_prompt_queries(PROMPT_QUERIES_FILE)

    queries: list[dict] = prompts_queries.get(ConfigConstants.KEY_QUERIES)

    # OpenAI
    openai_pipeline = OpenAIPipeline(config, prompts_queries)
    # results_openai: Optional[list[str]] = run_openai(config, prompt, queries)
    results_openai: ExperimentResults = openai_pipeline.run_queries()
    print(f"\nResults OpenAI run:\n{results_openai}")

    # Run with local
    # results_local: Optional[list[str]] = run_local(config, prompt, queries)

    # Save results to a file

    # save_results(config["output"])


if __name__ == "__main__":
    main()
