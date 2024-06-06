from openai import OpenAI

from shared.constants import ConfigConstants


class OpenAILLM:
    def __init__(self, config: dict) -> None:
        self.model = config[ConfigConstants.KEY_LLM]
        self.client = OpenAI()

    def chat_request(self, text: str) -> str:
        """Returns a chat message."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        )
        return response.choices[0].message.content
