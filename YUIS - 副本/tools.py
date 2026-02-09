import json
import re
from ddgs import DDGS
import json_repair


# 进行联网搜索 防止数据过少 以及数据清洗
def search_web(keywords: str, max_results: int = 3) -> str:
    """使用DUCKDuCKGO 进行搜索 返回拼接文本"""
    print(f"正在搜索： {keywords}")
    try:
        results = DDGS().text(keywords, max_results=max_results)
        if not results:
            return ""
        content = ""
        for item in results:
            content += f"[来源：{item['title']}]\n{item['body']}\n\n"
            return content
        return content
    except:
        print("搜索失败")
        return ""


def extract_json(text: str):
    """
    【DeepSeek 专用无敌版】
    集成：自动补全、数学符号转义修复、静默容错
    """
    if not text:
        return None

    # 1. 基础清洗
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # 2. 隐形字符清洗
    text = text.replace("\xa0", " ")  # 不换行空格
    text = text.replace("\u3000", " ")  # 全角空格

    # 3. 数学公式/转义符清洗 防止公式报错

    try:
        text = re.sub(r'\\(?![/bfnrtu"\\])', r"\\\\", text)
    except:
        pass

    # 4. 截取有效范围
    start = text.find("{")
    if start != -1:
        text = text[start:]
        # 如果找不到结尾的 }，说明被截断了，json_repair 会自动修，不用我们管
        end = text.rfind("}")
        if end != -1:
            # 只有当结尾明显是 } 时才截取，否则保留原样让 repair 去补全
            text = text[: end + 1]

    # --- 第一：标准解析 ---
    try:
        # strict=False 允许控制字符
        return json.loads(text, strict=False)
    except:
        pass

    # --- 第二：json_repair ---
    try:
        # 这里的 decoded_object 就是最终修好的数据
        return json_repair.loads(text)
    except RecursionError:
        print(" json_repair 陷入递归，尝试暴力闭合...")
        # 最后的挣扎：强行加括号
        try:
            return json_repair.loads(text + "]}")
        except:
            pass
    except Exception as e:
        print(f" 解析最终失败: {e}")

    # 如果真的到了这一步，打印出来看看
    print("-" * 30)
    print("DEBUG: 无法解析的文本:")
    print(text[:200] + "...")
    print("-" * 30)

    return None
