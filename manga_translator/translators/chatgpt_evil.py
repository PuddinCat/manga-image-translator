from .chatgpt import GPT35TurboTranslator
from . import llm
from .common import MissingAPIKeyException
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
        try:
            super().__init__()
        except MissingAPIKeyException:  # 没想到吧，我不需要KEY！
            # 你妈的，整个项目的设计简直就是一坨狗屎
            # 这个exception如果不捕获，会在一个随着web后台启动的线程中被捕获
            # 然后那个线程就会以为这个类无法使用
            # 而且那个线程还通过他妈的HTTP请求告知web后台这个类无法使用
            # 再然后WEB后台就会去他妈的 改 写 HTML 模板！还是用的正则！
            # 最后虽然web后台的HTML中写死类一大堆translator，但是最终用户根本看不到我加上的类
            # 我找了半天才知道这个exception有这么个鬼用
            # 喵了个咪的 ---by puddin
            pass
        # 帮父类擦屁股！
        self.token_count = 0
        self.token_count_last = 0
        # 初始化咱的LLM
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
        tokens = num_tokens_from_messages(messages)
        self.token_count += tokens
        self.token_count_last = tokens
        print(answer)

        return answer
