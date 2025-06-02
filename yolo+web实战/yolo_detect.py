import torch
from pathlib import Path
import os
import cv2
model=torch.hub.load('ultralytics/yolov5','yolov5s',trust_repo=True)
def detect_image(input_path,output_path):
    results=model(input_path)#模型对图片的识别
    filename = os.path.basename(input_path)
    output_file_path = os.path.join(output_path, filename)

    # 渲染结果并保存
    annotated_img = results.render()[0]
    cv2.imwrite(output_file_path, annotated_img)
