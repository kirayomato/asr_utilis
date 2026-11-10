import re
from typing import Callable, Optional


class TextNormalizer:
    """文本标准化器"""

    def __init__(
        self,
        remove_punctuation: bool = True,
        lower_case: bool = True,
        separate_chinese_english: bool = True,
        custom_fn: Optional[Callable[[str], str]] = None,
    ):
        self.remove_punctuation = remove_punctuation
        self.lower_case = lower_case
        self.separate_chinese_english = separate_chinese_english
        self.custom_fn = custom_fn

    def normalize(self, text: str) -> str:
        """标准化文本"""
        if self.custom_fn:
            return self.custom_fn(text)

        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)

        if self.separate_chinese_english:
            parts = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", text)
            result = []
            for word in parts:
                if word.encode("UTF-8").isalpha():
                    result.append(word.lower() if self.lower_case else word)
                else:
                    result.extend(list(word))
            text = " ".join(result)
        elif self.lower_case:
            text = text.lower()

        return text.strip()
