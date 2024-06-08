from datetime import datetime
from typing import Optional
import logging
from logging import Logger

from local_pipeline.embedding import LocalEmbeddings
from local_pipeline.llm import LLAMA3
from shared.database import ChromaDB
from shared.models import ExperimentResults, QueryResult
from shared.utils import create_prompt
from shared.constants import ConfigConstants, EmbeddingConstants


class LocalPipeline:
    def __init__(self, config: dict, prompts_queries: dict):
        self.logger: Logger = logging.getLogger(self.__class__.__name__)
        self.config: dict = config
        self.prompt_template: str = prompts_queries.get(ConfigConstants.KEY_PROMPT)
        self.queries: list[dict] = prompts_queries.get(ConfigConstants.KEY_QUERIES, [])

        self.model: str = config[ConfigConstants.KEY_PIPELINES][
            ConfigConstants.KEY_LOCAL
        ][ConfigConstants.KEY_LLM]

        self.embedder_local: LocalEmbeddings = LocalEmbeddings(
            config[ConfigConstants.KEY_PIPELINES][ConfigConstants.KEY_LOCAL]
        )
        self.splitter_method: str = config[ConfigConstants.KEY_SPLITTER][
            ConfigConstants.KEY_METHOD
        ]
        self.chunk_size: int = config[ConfigConstants.KEY_SPLITTER][
            ConfigConstants.KEY_CHUNK_SIZE
        ]
        self.chunk_overlap: int = config[ConfigConstants.KEY_SPLITTER][
            ConfigConstants.KEY_CHUNK_OVERLAP
        ]
        self.database: ChromaDB = ChromaDB(
            config,
            f"local_{self.splitter_method}_{self.chunk_size}_{self.chunk_overlap}",
        )
        self.llm = LLAMA3(
            config[ConfigConstants.KEY_PIPELINES][ConfigConstants.KEY_LOCAL]
        )

    def run_queries(self) -> ExperimentResults:
        """Runs queries against the local pipeline."""

        results = ExperimentResults(
            results=[],
            model=self.model,
            parameters=[
                self.config[ConfigConstants.KEY_SPLITTER],
                self.config[ConfigConstants.KEY_PIPELINES][ConfigConstants.KEY_LOCAL],
            ],
            timestamp_end=None,
        )

        embeddings_local: list[list[float]] = self.embedder_local.get_embeddings(
            [query.get(EmbeddingConstants.KEY_TEXT) for query in self.queries]
        )

        contexts: list[Optional[list[str]]] = self.database.query(embeddings_local)

        for ind, query in enumerate(self.queries):
            query_text = query.get(EmbeddingConstants.KEY_TEXT)

            prompt = create_prompt(self.prompt_template, query_text, contexts[ind])

            chat_response = self.llm.chat_request(prompt)

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
