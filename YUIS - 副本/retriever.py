from embedding import embed_texts
from index import FaissIndex
from config import Settings
import numpy as np


def retriever(
    query: str,
    faiss_idx,
    top_k=Settings.TOP_K,
) -> list:
    """
    检索最接近的片段 发给llm 基于教材实现RAG 否则只能从联网获得信息
    """
    q_vec = embed_texts([query])
    result = faiss_idx.search(q_vec, k=top_k)
    return result
