from abc import ABCMeta, abstractmethod


class AbstractTokenizer(metaclass=ABCMeta):
    """Abstract base class for Tokenizers."""

    @abstractmethod
    def tokenize_text(self, text: str) -> list[int]:
        """Tokenizes the text in the input."""

    @abstractmethod
    def check_tokenlimit_exceeded(self, texts: list[str]) -> bool:
        """Checks the number of tokens in the texts against the defined limit."""

    def check_tokenlimit_exceeded(self, texts: list[str]) -> bool:
        """Checks the number of tokens in the texts against the defined limit."""
        self.logger.info(
            "Checking if number of tokens exceeds the maximum for embedding model %s...",
            self.model,
        )
        max_count_tokens = 0
        for text in texts:
            count_tokens = len(self.tokenize_text(text))
            max_count_tokens = max(count_tokens, max_count_tokens)
            if count_tokens > self.max_tokens:
                self.logger.warning(
                    "Number of %s tokens in input exceeds limit of %s tokens!",
                    count_tokens,
                    self.max_tokens,
                )
                return True
        self.logger.info("Highest token count for embedding: %s", max_count_tokens)
        return False
