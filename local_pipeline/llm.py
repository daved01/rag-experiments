import logging
from langchain_community.llms import Ollama

from local_pipeline.tokenizer import LLAMA3Tokenizer
from shared.constants import ConfigConstants


class LLAMA3:
    def __init__(self, config_local: dict) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = config_local[ConfigConstants.KEY_LLM]

        # TODO: Update this tokenizer with the one used by LLAMA3
        # self.tokenizer = LLAMA3Tokenizer(
        #     self.model,
        #     config_local[ConfigConstants.KEY_MAX_TOKENS],
        # )
        self.client = Ollama(model=self.model)

    def chat_request(self, text: str) -> str:
        """Returns a chat message."""
        self.logger.info("Sending request to local model %s...", self.model)

        # TODO: Add this check later
        # self.tokenizer.check_tokenlimit_exceeded([text])

        # TODO: Update this with LLAMA3
        return self.client.invoke(text)
