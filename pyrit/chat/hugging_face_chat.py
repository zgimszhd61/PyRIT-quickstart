# 版权所有 (c) Microsoft Corporation。
# 根据MIT许可证授权。

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from pyrit.common.prompt_template_generator import PromptTemplateGenerator
from pyrit.interfaces import ChatSupport
from pyrit.models import ChatMessage

logger = logging.getLogger(__name__)

class HuggingFaceChat(ChatSupport):
    """HuggingFaceChat与HuggingFace模型交互，特别是用于进行红队活动。

    参数:
        ChatSupport (abc.ABC): 实现与HuggingFace模型交互的方法。
    """

    def __init__(
        self,
        *,
        model_id: str = "cognitivecomputations/WizardLM-7B-Uncensored",
        use_cuda: bool = False,
        tensor_format: str = "pt",
    ) -> None:
        """
        参数:
            model_id: HuggingFace模型ID，可以在模型页面找到。默认为cognitivecomputations/WizardLM-7B-Uncensored
            use_cuda: 标志位，指示是否使用CUDA（GPU）（如果可用）。它允许软件开发者使用CUDA启用的图形处理单元（GPU）进行通用处理。
            tensor_format: 转换模型数据张量格式，默认为"pt"（PyTorch）。"np" -> (Numpy) 和 "tf" ->TensorFlow
        """
        self.model_id: str = model_id
        self.use_cuda: bool = use_cuda

        if self.use_cuda and not torch.cuda.is_available():
            raise RuntimeError("请求CUDA但不可用。")

        # 加载HuggingFace的分词器和模型
        self.tokenizer: AutoTokenizer = None
        self.model: AutoModelForCausalLM = None
        self.load_model_and_tokenizer()

        # 转换模型数据张量格式，默认为"pt"（PyTorch）。"np" -> (Numpy) 和 "tf" ->TensorFlow
        self.tensor_format = tensor_format

        self.prompt_template_generator = PromptTemplateGenerator()

    def is_model_id_valid(self) -> bool:
        """
        检查HuggingFace模型ID是否有效。
        :返回: 如果有效则为True，否则为False。
        """
        try:
            # 尝试加载模型的配置
            PretrainedConfig.from_pretrained(self.model_id)
            return True
        except Exception as e:
            logger.error(f"无效的HuggingFace模型ID {self.model_id}: {e}")
            return False

    def load_model_and_tokenizer(self):
        """
        加载模型和分词器。
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            if self.use_cuda:
                self.model.to("cuda")  # 将模型移动到GPU
            logger.info(f"模型 {self.model_id} 加载成功。")
        except Exception as e:
            logger.error(f"加载模型 {self.model_id} 时出错: {e}")
            raise

    def extract_last_assistant_response(self, text: str) -> str:
        """识别给定文本字符串中“ASSISTANT”最后一次出现的位置，并提取其之后的所有内容。

        参数:
            text (str): 包含对话的字符串，包括系统指令、用户输入和助理响应，格式化有特定的标记
                    并可能包含闭合标记'</s>'。

        返回:
            str: 提供文本中的最后一个助理响应。如果找不到助理响应，则返回空字符串。
        """

        # 查找“ASSISTANT:”的最后一次出现
        last_assistant_index = text.rfind("ASSISTANT:")

        if last_assistant_index == -1:
            return ""

        # 提取“ASSISTANT:”之后的文本
        extracted_text = text[last_assistant_index + len("ASSISTANT:"):]

        # 查找闭合标记</s>并将文本裁剪到该点为止
        closing_token_index = extracted_text.find("</s>")
        if closing_token_index != -1:
            extracted_text = extracted_text[:closing_token_index].strip()

        return extracted_text

    async def complete_chat_async(self, messages: list[ChatMessage]) -> str:
        raise NotImplementedError

    def complete_chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 400,
        temperature: float = 1.0,
        top_p: int = 1,
    ) -> str:
        """通过生成对给定输入提示的响应来完成聊天交互。

        参数:
            messages (list[ChatMessage]): 包含角色和内容的聊天消息对象。
            max_tokens (int, 可选): 生成的最大令牌数。默认为400。
            temperature (float, 可选): 控制响应生成中的随机性。默认为1.0。
            top_p (int, 可选): 控制响应生成的多样性。默认为1。

        返回:
            str: 生成的响应消息。
        """
        prompt_template = self.prompt_template_generator.generate_template(messages)

        try:
            # 对聊天模板进行分词并获取input_ids
            input_ids = self.tokenizer(prompt_template, return_tensors=self.tensor_format).input_ids

            # 如果CUDA可用，将输入移动到GPU上进行推理
            if self.use_cuda:
                input_ids = input_ids.to("cuda")  # 将输入移动到GPU上进行推理

            # 生成响应
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            response_message = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"推理过程中发生错误: {e}")
            raise

        # 清理响应消息以移除特定的令牌（如果有的话）
        extracted_response_message = self.extract_last_assistant_response(response_message)
        return extracted_response_message
