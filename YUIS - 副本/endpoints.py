from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from data_loader import extract_text_from_file, split_text
import shutil
from index import FaissIndex
from fastapi import BackgroundTasks
from retriever import retriever
from embedding import embed_texts
from generator import bulid_prompt
from model_local import generate_local
from config import Settings
import global_vars
from tools import search_web, extract_json
from model_api import generate_api
from FlagEmbedding import FlagReranker  # 重排序库
import asyncio
import glob  # 找到删除文件
import os

router = APIRouter()

print("加载重排序模型")
try:
    reranker_model = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
    print("加载成功")
except:
    print("加载失败")


class QueryIn(BaseModel):  # 强制规定question类型
    question: str


class QuizRequest(BaseModel):
    topic: str = "全部内容"
    difficulty: str = "中等"
    num_questions: int = 3  # 题目数量 后期改为试卷一致
    choice_count: int = 3  # 选择题数量
    fill_blank_count: int = 2  # 填空题数量
    short_answer_count: int = 1  # 简答题数量
    include_web_search: bool = True


def llm_generate(prompt: str, system_prompt: str = "") -> str:
    """决定使用哪个配置 (API)"""
    if Settings.USE_API:
        print("调用DS API")
        return generate_api(prompt, system_prompt=system_prompt)
    else:
        print("调用本地")
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        return generate_local(full_prompt)


@router.post("/generate_quiz")
async def generate_quiz(req: QuizRequest) -> dict:
    if (
        global_vars.db_index is None
        or global_vars.db_index.index.ntotal == 0  # 计算向量的总数
    ):  # 还没有上传资料 先上传资料之后再产生卷子
        return {"error": "请先上传复习资料 PDF"}
    # RAG
    # 看是否指定了topic
    search_query = req.topic if req.topic != "全部内容" else "重要概念 总结"
    raw_docs = retriever(
        search_query, global_vars.db_index, top_k=20
    )  # 抽取片段(由于后续进行重排序 这里的top_k取大一点)
    final_docs = []
    if reranker_model and raw_docs:
        print("进行重排序")
        pairs = [(search_query, doc["text"]) for doc in raw_docs]
        scores = reranker_model.compute_score(pairs)
        if isinstance(scores, float):
            scores = [scores]
        combined = list(zip(raw_docs, scores))  # type: ignore
        # 对分数进行排序
        combined.sort(key=lambda x: x[1], reverse=True)  # type:ignore
        final_docs = [doc for doc, score in combined[:5]]  # type:ignore
    else:
        final_docs = raw_docs[:5]

    book_context = "\n".join([d["text"] for d in final_docs])
    web_content = ""
    if req.include_web_search:
        web_keywords = f"{req.topic} 知识点衍生 细节覆盖"
        web_content = search_web(web_keywords)

    def _sync_generate_task(task_name: str, count: int, q_types: list):
        if count <= 0:
            return []
        type_instruction = ",".join(q_types)
        # num1 = req.choice_count
        # num2 = req.fill_blank_count
        # num3 = req.short_answer_count
        # total_q = num1 + num2 + num3
        user_prompt = f"""你是一个出题老师 请基于给的'教材'和'网络咨询'生成一份试卷。"
        [教材]:
        {book_context[::1]}
        [网络咨询]:
        {web_content[:2000] if web_content else None}

        [要求]:1. 生成{count} 道题目。
                2. 题型是{type_instruction}
                3. 绝对不要修改 JSON 结构！不要嵌套！不要使用 "questions" 数组！
                4. 题目类型只能是：'choice' (选择), 'fill_blank' (填空), 'short_answer' (简答)。
                5. 遇到多选也请归类为 'choice'。
                6. 生成完 JSON 的最后一个 }} 后立刻停止。不要写“注意”、“解释”等任何文字
                
        "quiz": [
            {{
            "id": 1,
            "type": "choice",
            "question": "Transformer的核心机制是什么？",
            "options": ["A. 卷积", "B. 自注意力", "C. 循环", "D. 池化"],
            "answer": "B",
            "analysis": "根据教材,Transformer核心是Self-Attention。"
            }},
            {{
            "id": 2,
            "type": "short_answer",
            "question": "结合最新资讯，列举一个深度学习的应用。",
            "answer": "自动驾驶或AI绘画..."
            }}
        ]
        }}    
        
        """
        print("正在生成...")
        raw_res = llm_generate(user_prompt, system_prompt="严格遵守JSON格式")
        data = extract_json(raw_res)
        return data.get("quiz", []) if data else []  # type:ignore

    task = []  # 开始并发任务
    if req.choice_count > 0:
        task.append(
            asyncio.to_thread(
                _sync_generate_task, "选择题", req.choice_count, ["choice"]
            )
        )
    other_count = req.fill_blank_count + req.short_answer_count
    if other_count > 0:
        task.append(
            asyncio.to_thread(
                _sync_generate_task,
                "非选择题",
                other_count,
                ["fill_blank", "short_answer"],
            )
        )
    print(f"启动{len(task)}个协程任务")
    results = await asyncio.gather(*task)
    final_quiz = []
    for batch_questions in results:
        final_quiz.extend(batch_questions)
    for idx, q in enumerate(final_quiz):
        q["id"] = idx + 1
    print(f"共生成{len(final_quiz)}") if final_quiz else print("生成失败")
    return {"status": "success", "data": {"quiz": final_quiz}}


class ExplainRequest(BaseModel):
    question: str
    user_answer: str


@router.post("/ask")
def ask(q: QueryIn) -> dict:
    if global_vars.db_index is None or global_vars.db_index.index.ntotal == 0:
        return {
            "answer": "知识库正在初始化或为空，请稍后重试或上传文档。",
            "source": [],
        }
    docs = retriever(q.question, global_vars.db_index)
    prompt = bulid_prompt(q.question, docs)
    answer = generate_local(prompt)
    return {"answer": answer, "source": [d.get("source") for d in docs]}


@router.post("/explain_question")
def explain_question(req: ExplainRequest) -> dict:
    docs = retriever(req.question, global_vars.db_index, top_k=2)
    source_text = "\n".join([d["text"] for d in docs])
    user_prompt = f"""你是辅导老师 学生做了题需要批改和指导 你根据原文进行点评
【题目】：{req.question}
【学生回答】：{req.user_answer}

[原文]:{source_text[:2000]}

[任务]:
1. 判断学生回答是否正确
2. 如果错误 根据原文进行回答
3. 不够详细 需要拓展后回答
"""
    system_prompt = (
        "你是一位仁慈博学的大学老师 在做完了你的卷子后 学生的错题需要你根据原文进行讲解"
    )
    explanation = llm_generate(user_prompt, system_prompt=system_prompt)
    return {"explanation": explanation, "source": [d.get("source") for d in docs]}


def process_file_task(file_path: str, source_name: str):
    # 1. 解析文本
    text = extract_text_from_file(file_path)
    # 2. 切片
    chunks = split_text(text)
    if not chunks:
        return
    # 3. 向量化
    embeddings = embed_texts(chunks)
    # 4. 准备元数据
    metadatas = [{"source": source_name, "text": chunk} for chunk in chunks]
    # 5. 存入 FAISS
    global_vars.db_index.add(embeddings, metadatas)  # type: ignore
    print(f"File {source_name} processed and indexed.")


@router.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 1. 保存文件到临时目录
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb+") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. 使用后台任务处理耗时操作（解析+向量化）
    background_tasks.add_task(process_file_task, file_location, file.filename)  # type: ignore

    return {"status": "processing", "filename": file.filename}


@router.post("/reset_knowledge_base")
def reset_knowledge_base():
    global_vars.db_index = FaissIndex(dim=384)
    print("已经重置")
    files = glob.glob("temp_*")
    del_count = 0
    for f in files:
        try:
            os.remove(f)
            del_count += 1
        except:
            print("删除失败")
    return {
        "status": "success",
        "message": f"知识库已清空，删除了 {del_count} 个临时文件。",
    }
