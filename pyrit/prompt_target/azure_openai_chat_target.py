# 版权所有 (c) Microsoft Corporation。
# 根据MIT许可证授权。

from pyrit.chat.azure_openai_chat import AzureOpenAIChat
from pyrit.memory import FileMemory, MemoryInterface
from pyrit.models import ChatMessage
from pyrit.prompt_target import PromptTarget

# 定义一个继承自AzureOpenAIChat和PromptTarget的类，用于处理与Azure OpenAI聊天的逻辑
class AzureOpenAIChatTarget(AzureOpenAIChat, PromptTarget):
    def __init__(
        self,
        *,
        deployment_name: str, # 部署名称
        endpoint: str, # API端点
        api_key: str, # API密钥
        memory: MemoryInterface = None, # 内存接口，默认为None
        api_version: str = "2023-08-01-preview", # API版本，默认为2023-08-01-preview
        temperature: float = 1.0, # 生成文本的温度参数，默认为1.0
    ) -> None:
        super().__init__(deployment_name=deployment_name, endpoint=endpoint, api_key=api_key, api_version=api_version)

        self.memory = memory if memory else FileMemory() # 如果提供了memory参数，则使用该参数，否则默认使用FileMemory
        self.temperature = temperature # 设置温度参数

    # 设置系统提示信息
    def set_system_prompt(self, prompt: str, conversation_id: str, normalizer_id: str) -> None:
        # 根据会话ID获取记忆中的消息
        messages = self.memory.get_memories_with_conversation_id(conversation_id=conversation_id)

        # 如果消息存在，则抛出异常，因为系统提示需要在对话开始时设置
        if messages:
            raise RuntimeError("Conversation already exists, system prompt needs to be set at the beginning")

        # 向内存中添加系统提示消息
        self.memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="system", content=prompt),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

    # 发送提示信息并获取响应
    def send_prompt(self, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        # 根据会话ID获取聊天消息
        messages = self.memory.get_chat_messages_with_conversation_id(conversation_id=conversation_id)

        # 创建用户角色的聊天消息
        msg = ChatMessage(role="user", content=normalized_prompt)

        # 将消息添加到消息列表中
        messages.append(msg)

        # 将聊天消息添加到内存中
        self.memory.add_chat_message_to_memory(
            conversation=msg, conversation_id=conversation_id, normalizer_id=normalizer_id
        )

        # 使用父类的complete_chat方法发送消息并获取响应
        resp = super().complete_chat(messages=messages, temperature=self.temperature)

        # 将响应作为助手角色的聊天消息添加到内存中
        self.memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="assistant", content=resp),
            conversation_id=conversation_id,
            normalizer_id=normalizer_id,
        )

        # 返回响应
        return resp
