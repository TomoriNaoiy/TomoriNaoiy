import faiss
import numpy as np
import json
from pathlib import Path
from config import Settings


class FaissIndex:
    def __init__(self, dim, index_path=Settings.FAISS_INDEX_PATH):
        """
        dim: 向量维度
        index_path 使用设置好的地址
        id_map: 向量对应的文本信息 便于后面搜索
        """
        self.dim = dim
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(dim)  # L2方法获得索引
        self.id_map = []

    def add(
        self, embeddings: np.ndarray, metadatas: list
    ):  # 把向量添加数据库并保存到磁盘
        self.index.add(embeddings) # type: ignore
        self.id_map.extend(metadatas)
        self.save()

    def search(self, query_vec: np.ndarray, k=5) -> list:
        """
        放入要查询的向量 搜索最近的k个向量 如果数据库有(不超过)就存入list返回
        """
        D, I = self.index.search(query_vec, k) # type: ignore
        results = []
        for idx in I[0]:
            if idx < len(self.id_map):
                results.append(self.id_map[idx])
        return results

    def save(self):
        """
        索引结构（向量、分块信息）保存为二进制文件；
        元数据保存为 JSON 文件。
        """
        Path(self.index_path).parent.mkdir(
            parents=True, exist_ok=True
        )  # 如果没有这个文件夹就先创建
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, ensure_ascii=False)

    def load(self):
        """
        从磁盘加载数据
        """
        if Path(self.index_path).exists():
            self.index = faiss.read_index(self.index_path)
            with open(self.index_path + "meta.json", "r", encoding="utf-8") as f:
                self.id_map = json.load(f)
