from openai import OpenAI
import logging

from openai_pipeline.tokenizer import OpenAITokenizer
from shared.models import Document
from shared.constants import ConfigConstants


class OpenAIEmbeddings:
    """Creates embeddings"""

    def __init__(self, config_openai: dict) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = config_openai[ConfigConstants.KEY_EMBEDDING]
        self.tokenizer = OpenAITokenizer(
            config_openai[ConfigConstants.KEY_EMBEDDING],
            config_openai[ConfigConstants.KEY_MAX_TOKENS],
        )
        self.client = OpenAI()

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Gets the embeddings for a list of texts."""

        self.logger.info("Creating embeddings ...")
        texts_cleaned = [text.replace("\n", " ") for text in texts]

        if self.tokenizer.check_tokenlimit_exceeded(texts_cleaned):
            self.logger.warning(
                "Number of tokens exceeds the limit. Text will be truncated."
            )
        responses = self.client.embeddings.create(
            input=texts_cleaned, model=self.model_name
        )

        assert len(responses.data) == len(texts)

        return [response.embedding for response in responses.data]

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
