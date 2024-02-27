# 版权属于 Microsoft Corporation。
# 根据 MIT 许可证授权。

from pathlib import Path
import pathlib

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from pyrit.common.path import RESULTS_PATH  # 引入结果路径
from pyrit.memory.memory_embedding import default_memory_embedding_factory  # 引入默认记忆嵌入工厂

# 引入会话记忆条目、会话记忆条目列表和带有相似度的会话消息
from pyrit.memory.memory_models import (
    ConversationMemoryEntry,
    ConversationMemoryEntryList,
    ConversationMessageWithSimilarity,
)

from pyrit.interfaces import EmbeddingSupport  # 引入嵌入支持接口
from pyrit.memory.memory_interface import MemoryInterface  # 引入记忆接口


class FileMemory(MemoryInterface):
    """
    处理以 JSON 格式存储和检索聊天记忆的类。
    由于许多操作都必须在内存中序列化，因此它不具备可扩展性。

    参数:
        filepath (Union[Path, str]): 记忆文件的路径。

    异常:
        ValueError: 如果选择了无效的记忆文件。

    属性:
        filepath (Path): 记忆文件的路径。
    """

    file_extension = ".json.memory"  # 文件扩展名
    default_memory_file = "default_memory.json.memory"  # 默认记忆文件名

    def __init__(self, *, filepath: Path | str = None, embedding_model: EmbeddingSupport = None):
        self.memory_embedding = default_memory_embedding_factory(embedding_model=embedding_model)  # 初始化记忆嵌入

        if filepath is None:  # 如果未提供文件路径，则使用默认路径
            filepath = pathlib.Path(RESULTS_PATH, self.default_memory_file).resolve()
            filepath.touch(exist_ok=True)  # 确保文件存在

        if isinstance(filepath, str):
            filepath = Path(filepath)  # 将字符串路径转换为 Path 对象
        self.filepath = filepath

        if not filepath.suffix:  # 如果文件没有后缀，则添加默认后缀
            self.filepath = self.filepath.with_suffix(self.file_extension)

        if "".join(self.filepath.suffixes) != self.file_extension:
            raise ValueError(
                f"Invalid memory file selected '{self.filepath}'. \
                Memory files must have extension '{self.file_extension}'."
            )  # 检查文件扩展名是否正确

    def get_all_memory(self) -> list[ConversationMemoryEntry]:
        """
        实现 Memory 接口中的 get_all_memory 方法。
        """

        if not self.filepath.exists() or self.filepath.stat().st_size == 0:
            return []  # 如果文件不存在或为空，则返回空列表

        memory_data = self.filepath.read_text(encoding="utf-8")
        return ConversationMemoryEntryList.parse_raw(memory_data).conversations  # 读取并解析记忆数据

    def save_conversation_memory_entries(self, new_entries: list[ConversationMemoryEntry]):
        """
        实现 Memory 接口中的 save_conversation_memory_entries 方法。
        """
        entries = self.get_all_memory()  # 获取当前所有记忆条目
        entries.extend(new_entries)  # 添加新的记忆条目

        entryList = ConversationMemoryEntryList(conversations=entries)
        self.filepath.write_text(entryList.model_dump_json(), encoding="utf-8")  # 将更新后的记忆数据写入文件

    def get_memory_by_exact_match(self, *, memory_entry_content: str) -> list[ConversationMessageWithSimilarity | None]:
        """
        实现 Memory 接口中的 get_memory_by_exact_match 方法。
        """
        msg_matches: list[ConversationMessageWithSimilarity | None] = []
        for memory_entry in self.get_all_memory():
            if memory_entry.content == memory_entry_content:  # 精确匹配内容
                msg_matches.append(
                    ConversationMessageWithSimilarity(
                        score=1.0,
                        role=memory_entry.role,
                        content=memory_entry.content,
                        metric="exact",
                    )
                )
        return msg_matches  # 返回匹配的消息

    def get_memory_by_embedding_similarity(
        self, *, memory_entry_emb: list[float], threshold: float = 0.8
    ) -> list[ConversationMessageWithSimilarity | None]:
        """
        实现 Memory 接口中的 get_memory_by_embedding_similarity 方法。
        """

        matched_conversations: list[ConversationMessageWithSimilarity] = []
        target_memory_emb = np.array(memory_entry_emb).reshape(1, -1)  # 将目标记忆嵌入转换为适合比较的形状

        for curr_memory in self.get_all_memory():
            if not curr_memory.embedding_memory_data or not curr_memory.embedding_memory_data.embedding:
                continue

            curr_memory_emb = np.array(curr_memory.embedding_memory_data.embedding).reshape(1, -1)
            emb_distance = cosine_similarity(target_memory_emb, curr_memory_emb)[0][0]  # 计算嵌入相似度
            if emb_distance >= threshold:
                matched_conversations.append(
                    ConversationMessageWithSimilarity(
                        score=emb_distance,
                        role=curr_memory.role,
                        content=curr_memory.content,
                        metric="embedding",
                    )
                )
        return matched_conversations  # 返回匹配的对话

    def get_memories_with_conversation_id(self, *, conversation_id: str) -> list[ConversationMemoryEntry]:
        """
        实现 Memory 接口中的 get_memories_with_conversation_id 方法。
        """
        memories: list[ConversationMemoryEntry] = []
        for mem_entry in self.get_all_memory():
            if mem_entry.conversation_id == conversation_id:
                memories.append(mem_entry)
        return memories  # 返回具有指定会话 ID 的记忆条目

    def get_memories_with_normalizer_id(self, *, normalizer_id: str) -> list[ConversationMemoryEntry]:
        """
        实现 Memory 接口中的 get_memories_with_normalizer_id 方法。
        """
        memories: list[ConversationMemoryEntry] = []
        for mem_entry in self.get_all_memory():
            if mem_entry.normalizer_id == normalizer_id:
                memories.append(mem_entry)
        return memories  # 返回具有指定规范化器 ID 的记忆条目
