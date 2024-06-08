import logging
from openai import OpenAI

from openai_pipeline.tokenizer import OpenAITokenizer
from shared.constants import ConfigConstants


class OpenAILLM:
    def __init__(self, config_openai: dict) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = config_openai[ConfigConstants.KEY_LLM]
        self.tokenizer = OpenAITokenizer(
            self.model,
            config_openai[ConfigConstants.KEY_MAX_TOKENS],
        )
        self.client = OpenAI()

    def chat_request(self, text: str) -> str:
        """Returns a chat message."""
        self.logger.info("Sending request to OpenAI LLM %s...", self.model)

        self.tokenizer.check_tokenlimit_exceeded([text])

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        )
        return response.choices[0].message.content
