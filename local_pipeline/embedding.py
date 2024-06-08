from sentence_transformers import SentenceTransformer
from typing import Any
import logging

from local_pipeline.tokenizer import SentenceTransformerTokenizer
from shared.models import Document
from shared.constants import ConfigConstants


class LocalEmbeddings:
    """Creates embeddings"""

    def __init__(self, config_local: dict) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = config_local["embedding"]
        self.model = SentenceTransformer(self.model_name)
        self.tokenizer = SentenceTransformerTokenizer(
            config_local[ConfigConstants.KEY_EMBEDDING],
            config_local[ConfigConstants.KEY_MAX_TOKENS],
        )

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Gets the embeddings for a list of texts."""

        self.logger.info("Creating embeddings ...")
        texts_cleaned = [text.replace("\n", " ") for text in texts]

        if self.tokenizer.check_tokenlimit_exceeded(texts_cleaned):
            self.logger.warning(
                "Number of tokens exceeds the limit. Text will be truncated."
            )

        return self.model.encode(texts_cleaned).tolist()

    def add_embeddings_to_docs(self, documents: list[Document]) -> list[Document]:
        """Adds embeddings to Document objects."""

        embeddings: list[list[float]] = self.get_embeddings(
            [doc.page_content for doc in documents]
        )
        assert len(embeddings) == len(documents)

        docs_with_embeddings: list[Document] | None = []
        for ind, embedding in enumerate(embeddings):
            doc = documents[ind]
            doc.embedding = embedding
            docs_with_embeddings.append(doc)

        return docs_with_embeddings
