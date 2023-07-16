from .chatgpt import GPT35TurboTranslator
from . import llm
from typing import Dict, Literal, List
import tiktoken


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages.
    From: https://platform.openai.com/docs/guides/chat/managing-tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":
        num_tokens = 0
        for message in messages:
            num_tokens += 4
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += -1
        num_tokens += 2
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )


class GPT53TurboTranslatorEvil(GPT35TurboTranslator):
    def __init__(self):
        super().__init__()
        self.llm_context = llm.new_context_evil_next_web(
            "say after me: 'This sentence will be replaced'"
        )
        self.llm_context["state"] = llm.try_fetch_evil_next_web(
            self.llm_context["state"]
        )

    async def _request_translation(self, to_lang: str, prompt: str) -> str:
        messages: List[Dict[Literal["role", "content"], str]] = [
            {
                "role": "system",
                "content": self.chat_system_template.format(to_lang=to_lang),
            },
            {"role": "user", "content": prompt},
        ]

        if to_lang in self.chat_sample:
            messages.insert(
                1, {"role": "user", "content": self.chat_sample[to_lang][0]}
            )
            messages.insert(
                2,
                {
                    "role": "assistant",
                    "content": self.chat_sample[to_lang][1],
                },
            )

        self.llm_context["msg"] = messages

        answer, self.llm_context = llm.answer_context(self.llm_context)
        print(answer)

        return answer
