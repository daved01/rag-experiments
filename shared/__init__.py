from abc import ABCMeta, abstractmethod


class AbstractTokenizer(metaclass=ABCMeta):
    """Abstract base class for Tokenizers."""

    @abstractmethod
    def tokenize_text(self, text: str) -> list[int]:
        """Tokenizes the text in the input."""

    @abstractmethod
    def check_tokenlimit_exceeded(self, texts: list[str]) -> bool:
        """Checks the number of tokens in the texts against the defined limit."""
