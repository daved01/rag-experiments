from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.models import Document


class TextSplitter:
    def __init__(self, config: dict):
        self.chunk_size = config.get("chunk_size")
        self.chunk_overlap = config.get("chunk_overlap")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def split_documents(self, documents: list[Document]) -> list[dict]:
        """Splits a list of documents into chunks.

        Args:
            documents: A list of ``Document``.

        Returns:
            A list of split documents.

        """

        return [
            Document(
                page_content=split.page_content,
                title=split.metadata.get("title"),
                metadata=split.metadata,
            )
            for split in self.splitter.split_documents(documents=documents)
        ]
