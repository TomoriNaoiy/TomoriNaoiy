from openai import OpenAI
from config import Settings
import os

client = None


def init_client():
    global client
    if client is None:
        client = OpenAI(
            api_key=Settings.API_KEY,
            base_url=Settings.API_BASE_URL,
        )


def generate_api(
    prompt: str, system_prompt: str = "你是一个喜欢抽象的AI", max_token: int = 2048
) -> str:
    """
    deepseek专用api
    """
    init_client()
    try:
        response = client.chat.completions.create(  # type: ignore
            model=Settings.API_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_token,
            temperature=0.7,  # 控制传造性 出题设置可以比回答低
            stream=False,  # 不用流式
        )
        return response.choices[0].message.content  # type: ignore
    except:
        print("API调用失败")
        return "API调用失败"
