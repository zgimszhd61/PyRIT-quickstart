# 版权所有 (c) Microsoft Corporation。
# 根据MIT许可证授权。

import os
from pyrit.embedding.azure_text_embedding import AzureTextEmbedding
from pyrit.interfaces import EmbeddingSupport
from pyrit.memory.memory_models import ConversationMemoryEntry, EmbeddingMemoryData

class MemoryEmbedding:
    """
    MemoryEmbedding类负责编码记忆嵌入。

    属性:
        embedding_model (EmbeddingSupport): 支持生成嵌入的类的实例。
    """

    def __init__(self, *, embedding_model: EmbeddingSupport):
        if embedding_model is None:
            raise ValueError("embedding_model 必须被设置。")
        self.embedding_model = embedding_model

    def generate_embedding_memory_data(self, *, chat_memory: ConversationMemoryEntry) -> EmbeddingMemoryData:
        """
        为聊天记忆条目生成元数据。

        参数:
            chat_memory (ConversationMemoryEntry): 聊天记忆条目。

        返回:
            ConversationMemoryEntryMetadata: 生成的元数据。
        """
        embedding_data = EmbeddingMemoryData(
            embedding=self.embedding_model.generate_text_embedding(text=chat_memory.content).data[0].embedding,
            embedding_type_name=self.embedding_model.__class__.__name__,
            uuid=chat_memory.uuid,
        )
        return embedding_data

def default_memory_embedding_factory(embedding_model: EmbeddingSupport = None) -> MemoryEmbedding | None:
    if embedding_model:
        return MemoryEmbedding(embedding_model=embedding_model)

    api_key = os.environ.get("AZURE_OPENAI_EMBEDDING_KEY")
    api_base = os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT")
    deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if api_key and api_base and deployment:
        model = AzureTextEmbedding(api_key=api_key, endpoint=api_base, deployment=deployment)
        return MemoryEmbedding(embedding_model=model)
    else:
        return None
