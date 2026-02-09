import fitz
from typing import List
import re
import os
from docx import Document
def extract_text_from_file(path: str) -> str: # 切割PDF
    '''
    独取PDF TEXT WORD里面的文字
    '''
    ext=os.path.splitext(path)[1].lower()
    doc=fitz.open(path)
    text=''
    if ext == '.pdf':
        doc=fitz.open(path)
        for page in doc:
            text += page.get_text() # type: ignore
        doc.close()
    elif ext == '.docx':
        doc=Document(path)
        for para in doc.paragraphs:
            text+=para.text+'\n'
    elif ext=='.text':
        with open(path, 'r', encoding='utf-8') as f:
            text=f.read()
    else:
        print(f"请输入支持的文档类型,当前格式不支持{ext}")
        return ""
    return text
def split_text(text: str,chunk_size=800, overlap=100) -> List[str]: 
    '''
    def: 将文本切割成token
    hunk_size: 区块大小
    over_lap: 重叠大小 防止连续性丢失 让前后两个块有所连接
    '''
    tokens=text.split()
    chunks=[]
    if len(tokens)==0:# 如果长度太短 直接按照字符长度进行切分
        for i in range(0,len(text),chunk_size-overlap):
            chunks.append(text[i:i+chunk_size])
        return chunks
    i=0
    # 否则按正常逻辑切割
    while i<len(tokens):
        chunk=''.join(tokens[i:i+chunk_size])
        chunks.append(chunk)
        i+=chunk_size-overlap
    return chunks
