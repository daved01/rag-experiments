from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

from shared.constants import ModelConstants, ConfigConstants
from shared.models import Document


class TextSplitter:
    def __init__(self, config: dict):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.chunk_size = config.get(ConfigConstants.KEY_CHUNK_SIZE)
        self.chunk_overlap = config.get(ConfigConstants.KEY_CHUNK_OVERLAP)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Splits a list of documents into chunks.

        Args:
            documents: A list of ``Document``.

        Returns:
            A list of split documents.

        """

        return [
            Document(
                page_content=split.page_content,
                title=split.metadata.get(ModelConstants.KEY_TITLE),
                metadata=split.metadata,
            )
            for split in self.splitter.split_documents(documents=documents)
        ]
