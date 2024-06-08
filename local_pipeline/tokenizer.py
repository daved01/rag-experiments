from transformers import AutoTokenizer
import logging

from shared import AbstractTokenizer


class SentenceTransformerTokenizer(AbstractTokenizer):
    def __init__(self, model: str, max_tokens: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_tokens = max_tokens
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def tokenize_text(self, text: str) -> list[int]:
        """Tokenizes the text in the input."""
        tokens = self.tokenizer(text)["input_ids"]
        return tokens

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


class LLAMA3Tokenizer(AbstractTokenizer):
    def __init__(self, model: str, max_tokens: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_tokens = max_tokens
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def tokenize_text(self, text: str) -> list[int]:
        """Tokenizes the text in the input."""
        tokens = self.tokenizer(text)["input_ids"]
        return tokens

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
