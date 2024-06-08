import logging
import tiktoken

from shared import AbstractTokenizer


class OpenAITokenizer(AbstractTokenizer):
    def __init__(self, model: str, max_tokens: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_tokens = max_tokens
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(self.model)

    def tokenize_text(self, text: str) -> list[int]:
        """Tokenizes the text in the input."""
        return self.tokenizer.encode(text)
