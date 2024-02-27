# 版权所有 (c) Microsoft Corporation。
# 根据 MIT 许可证授权。

import abc
from pyrit.memory import MemoryInterface

class PromptTarget(abc.ABC):  # 定义一个名为 PromptTarget 的抽象基类
    memory: MemoryInterface  # 定义一个 MemoryInterface 类型的变量 memory

    """
    支持的转换器列表。如果列表为空，则表示 PromptTarget 支持所有转换器。
    """
    supported_transformers: list  # 定义一个列表，用来存储支持的转换器

    def __init__(self, memory: MemoryInterface) -> None:  # 类的初始化函数
        self.memory = memory  # 初始化 memory 属性

    @abc.abstractmethod
    def set_system_prompt(self, prompt: str, conversation_id: str, normalizer_id: str) -> None:
        """
        为 PromptTarget 设置系统提示
        """

    @abc.abstractmethod
    def send_prompt(self, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        """
        向 PromptTarget 发送一个规范化的提示。
        """
