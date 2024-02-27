# 版权所有 (c) Microsoft Corporation。
# 根据MIT许可证授权。

from openai import AsyncAzureOpenAI, AzureOpenAI
from openai.types.chat import ChatCompletion
from pyrit.common import default_values

from pyrit.interfaces import ChatSupport
from pyrit.models import ChatMessage

class AzureOpenAIChat(ChatSupport):
    # 定义环境变量名称
    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_API_KEY"  # Azure OpenAI API密钥的环境变量名
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_ENDPOINT"  # Azure OpenAI终端点的环境变量名

    def __init__(
        self,
        *,
        deployment_name: str,  # 部署名称
        endpoint: str = None,  # 终端点，默认为空
        api_key: str = None,  # API密钥，默认为空
        api_version: str = "2023-08-01-preview",  # API版本，默认为2023-08-01预览版
    ) -> None:
        self._deployment_name = deployment_name

        # 获取必需的终端点和API密钥值
        endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )
        api_key = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )

        # 初始化AzureOpenAI和AsyncAzureOpenAI客户端
        self._client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        self._async_client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )

    def parse_chat_completion(self, response):
        """
        解析聊天消息以获取响应
        参数:
            response (ChatMessage): 包含生成的响应消息的聊天消息对象
        返回:
            str: 生成的响应消息
        """
        try:
            response_message = response.choices[0].message.content
        except KeyError as ex:
            if response.choices[0].finish_reason == "content_filter":
                raise RuntimeError(f"Azure因内容过滤阻止了响应。响应: {response}") from ex
            else:
                raise RuntimeError(f"Azure聊天出错。响应: {response}") from ex
        return response_message

    async def complete_chat_async(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 1024,  # 生成的最大令牌数，默认为1024
        temperature: float = 1.0,  # 控制响应生成的随机性，默认为1.0
        top_p: int = 1,  # 控制响应生成的多样性，默认为1
        frequency_penalty: float = 0.5,  # 控制生成相同文本行的频率，默认为0.5
        presence_penalty: float = 0.5,  # 控制谈论新话题的可能性，默认为0.5
    ) -> str:
        """
        完成异步聊天请求
        解析聊天消息以获取响应
        参数:
            messages (list[ChatMessage]): 包含角色和内容的聊天消息对象。
            其他参数见上方说明。
        返回:
            str: 生成的响应消息
        """
        response: ChatCompletion = await self._async_client.chat.completions.create(
            model=self._deployment_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n=1,
            stream=False,
            messages=[{"role": msg.role, "content": msg.content} for msg in messages],  # 忽略类型
        )
        return self.parse_chat_completion(response)

    def complete_chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: int = 1,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
    ) -> str:
        """
        完成同步聊天请求
        解析聊天消息以获取响应
        参数及返回值同上
        """
        response: ChatCompletion = self._client.chat.completions.create(
            model=self._deployment_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n=1,
            stream=False,
            messages=[{"role": msg.role, "content": msg.content} for msg in messages],  # 忽略类型
        )
        return self.parse_chat_completion(response)
