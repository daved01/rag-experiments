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


class LLAMA3Tokenizer(AbstractTokenizer):
    def __init__(self, model: str, max_tokens: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_tokens = max_tokens
        self.model = model
        self.tokenizer = None  # TODO: Add tokenizer

    def tokenize_text(self, text: str) -> list[int]:
        """Tokenizes the text in the input."""
        tokens = self.tokenizer(text)["input_ids"]
        return tokens
