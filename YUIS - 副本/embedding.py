from sentence_transformers import SentenceTransformer
import numpy as np
from config import Settings

model = SentenceTransformer(Settings.EMBADDING_MODEL)  # 使用该模型


def embed_texts(texts: list) -> np.ndarray:
    """
    通过huggingFace的sentence_transformers层的encode把文本转化为词向量 并以numpy数组的形式返回
    """
    embs = model.encode(
        texts, show_progress_bar=False, convert_to_numpy=True
    )  # 禁用进度条
    return embs
