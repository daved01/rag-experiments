import logging
from pypdf import PdfReader
import re

from shared.models import Document
from shared.constants import ModelConstants

filename_pattern = re.compile(r"([^/]+)(?=\.[^.]+$)")


class Loader:
    """Loads and chunks documents from file."""

    def __init__(self, paths: list[str]):
        self.paths = paths
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_pdf(self) -> list[Document]:
        """Loads a list of PDF files into a list of `Document` objects.

        Returns:
            A list of `Documents`, one for each page.
        """
        self.logger.info(
            "Trying to load %d PDF%s ...",
            len(self.paths),
            "s" if len(self.paths) != 1 else "",
        )
        documents = []

        for path in self.paths:
            try:
                title = self._extract_filename(path)
                with open(path, "rb") as file:
                    reader = PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        content = page.extract_text()
                        documents.append(
                            Document(
                                page_content=content,
                                title=title,
                                metadata={
                                    ModelConstants.KEY_PAGE: page_num + 1,
                                    ModelConstants.KEY_SOURCE: path,
                                    ModelConstants.KEY_TITLE: title,
                                },
                            )
                        )
            except ValueError:
                self.logger.warning("Failed to load file.")

            self.logger.info("Loaded PDF from %s!", path)
        return documents

    @staticmethod
    def _extract_filename(path: str) -> str:
        """Extract the filename from a path."""
        match = filename_pattern.search(path)
        if match:
            return match.group(1)
        else:
            raise ValueError("No valid filename found in the path")
