from typing import Any
from openai import OpenAI
import logging

from shared.models import Document


class OpenAIEmbeddings:
    """Creates embeddings"""

    def __init__(self, config_openai: dict) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = config_openai["embedding"]
        self.client = OpenAI()

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Gets the embeddings for a list of texts."""

        self.logger.info("Creating embeddings ...")
        texts_cleaned = [text.replace("\n", " ") for text in texts]

        responses = self.client.embeddings.create(input=texts_cleaned, model=self.model)

        assert len(responses.data) == len(texts)

        return [response.embedding for response in responses.data]

    def add_embeddings_to_docs(self, documents: list[Document]) -> list[Document]:
        """Adds embeddings to Document objects."""

        embeddings: list[float] = self.get_embeddings(
            [doc.page_content for doc in documents]
        )
        assert len(embeddings) == len(documents)

        docs_with_embeddings: list[Document] | None = []
        for ind, embedding in enumerate(embeddings):
            doc = documents[ind]
            doc.embedding = embedding
            docs_with_embeddings.append(doc)

        return docs_with_embeddings
