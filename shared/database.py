import chromadb
import logging
from typing import Optional
import uuid

from shared.constants import ConfigConstants, DatabaseConstants
from shared.models import Document


class ChromaDB:
    """Chroma database methods."""

    def __init__(self, config: dict, collection_name: str) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = chromadb.PersistentClient(
            path=config[ConfigConstants.KEY_CONFIG_DATABASE][
                ConfigConstants.KEY_CONFIG_PATH
            ]
        )
        self.collection = self._get_or_create_collection(collection_name)
        self.n_results = 5

    def _get_or_create_collection(self, name: str) -> None:
        """Creates a new collection."""
        try:
            collection = self.client.get_collection(name=name)
            self.logger.info("Loaded collection `%s`!", name)
            return collection
        except ValueError:
            collection = self.client.create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
            self.logger.info("Created collection `%s`!", name)
            return collection

    def add_chunks(self, chunks: list[Document]) -> None:
        """Adds documents to database."""
        n = len(chunks)
        self.logger.info("Adding %s chunks ...", n)

        self.collection.add(
            embeddings=[chunk.embedding for chunk in chunks],
            documents=[chunk.page_content for chunk in chunks],
            ids=[str(uuid.uuid4()) for i in range(n)],
        )

    def query(self, query_embeddings: list[list[float]]) -> list[Optional[list[str]]]:
        """Queries the database.

        Takes in a list of embeddings, typically one embedding per query to run
        those in a batch.

        Returns a list of list of string, where each string is a relevent context.
        The outer list corresponds to the queries.

        """
        response_obj = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=self.n_results,
        )

        docs = response_obj.get(DatabaseConstants.KEY_DATABASE_DOCUMENTS, [])
        assert len(docs) == len(query_embeddings)

        return docs
