def bulid_prompt(query: str,docs: list) -> str:
    context='\n\n'.join([f"来源：{d.get('source','unknown')}\n内容:{d['text']}" for d in docs])
    prompt=f"你是一个知识问答助理 根据内容进行回复 并简要带上知识来源:\n\n{context}\n\n问题:{query}\n 请回答"
    return prompt