import logging
from langchain_community.llms import Ollama

from local_pipeline.tokenizer import LLAMA3Tokenizer
from shared.constants import ConfigConstants


class LLAMA3:
    def __init__(self, config_local: dict) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = config_local[ConfigConstants.KEY_LLM]
        # TODO: Add tokenizer for LLAMA3
        self.client = Ollama(model=self.model)

    def chat_request(self, text: str) -> str:
        """Returns a chat message."""
        self.logger.info("Sending request to local model %s...", self.model)

        # TODO: Add token number checker here
        # self.tokenizer.check_tokenlimit_exceeded([text])

        return self.client.invoke(text)
