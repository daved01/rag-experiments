from datetime import datetime
from typing import Optional

from openai_models.embedding import OpenAIEmbeddings
from openai_models.llm import OpenAILLM
from shared.database import ChromaDB
from shared.models import ExperimentResults, QueryResult
from shared.utils import load_config, load_prompt_queries, save_results, create_prompt


def run_openai(
    config: dict, prompt_template: str, queries: list[dict]
) -> Optional[list[str]]:
    """Runs queries against the OpenAI-based pipeline."""

    results = ExperimentResults(
        results=[],
        model=config["pipelines"]["openai"]["llm"],
        parameters={},
        timestamp_end=None,
    )

    embedder_openai = OpenAIEmbeddings(config["pipelines"]["openai"])

    embeddings_openai: list[list[float]] = embedder_openai.get_embeddings(
        [query.get("text") for query in queries]
    )

    method = config["splitter"]["method"]
    chunk_size = config["splitter"]["chunk_size"]
    chunk_overlap = config["splitter"]["chunk_overlap"]
    database_openai = ChromaDB(config, f"openai_{method}_{chunk_size}_{chunk_overlap}")

    contexts: list[Optional[list[str]]] = database_openai.query(embeddings_openai)

    llm_openai = OpenAILLM(config["pipelines"]["openai"])

    for ind, query in enumerate(queries):
        query_text = query.get("text")

        # Create prompt
        prompt = create_prompt(prompt_template, query_text, contexts[ind])
        chat_response = llm_openai.chat_request(prompt)
        results.results.append(
            QueryResult(
                query=query_text,
                contexts=contexts[ind],
                prompt=prompt,
                response=chat_response,
            )
        )

    results.timestamp_end = datetime.now()
    return results


def run_local(config: dict, prompt: str, queries: list[dict]) -> Optional[list[str]]:
    return []


def main():
    config = load_config("config.yaml")
    prompts_queries = load_prompt_queries("prompts_queries.json")
    prompt = prompts_queries.get("prompt")
    queries: list[dict] = prompts_queries.get("queries")

    # OpenAI
    results_openai: Optional[list[str]] = run_openai(config, prompt, queries)
    print(f"\nResults OpenAI run:\n{results_openai}")

    # Run with local
    # results_local: Optional[list[str]] = run_local(config, prompt, queries)

    # Save results to a file

    # save_results(config["output"])


if __name__ == "__main__":
    main()
