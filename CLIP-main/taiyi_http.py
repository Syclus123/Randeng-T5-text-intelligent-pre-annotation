'''
-*- coding: utf-8 -*-
@File  : taiyi_http.py
@Time  : 2023/11/27 13:18
'''
# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import requests
import torch
import numpy as np
import csv
from transformers import BertTokenizer, BertForSequenceClassification, CLIPProcessor, CLIPModel

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源的请求
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头
)

# 用于接收前端发送的URL
class UrlInfo(BaseModel):
    url: str

labels = []
with open('./class_name/imagenet1000.csv', 'r', encoding='GBK') as csv_file:
    csv_reader = csv.reader(csv_file)
    labels = [row[0] for row in csv_reader]
# with open('./class_name/class_cn.csv', 'r', encoding='GBK') as csv_file:
#     csv_reader = csv.reader(csv_file)
#     labels = [row[0] for row in csv_reader]

text_tokenizer = BertTokenizer.from_pretrained("./taiyiclip")
text_encoder = BertForSequenceClassification.from_pretrained("./taiyiclip").eval()
text = text_tokenizer(labels, return_tensors='pt', padding=True)['input_ids']

clip_model = CLIPModel.from_pretrained("clip32/")
processor = CLIPProcessor.from_pretrained("clip32/")

# 修改处理函数为异步函数
async def get_label(url: str):
    response = requests.get(url, proxies=None)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image)
            text_features = text_encoder(text).logits
            # 归一化
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            # 计算余弦相似度 logit_scale是尺度系数
            logit_scale = clip_model.logit_scale.exp()

            logits_per_image = logit_scale * image_features @ text_features.t()
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            return {"label": labels[np.argmax(probs)]}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to download the image from {url}")

# 定义FastAPI端点
@app.post("/img/")
async def process_image(url_info: UrlInfo):
    try:
        result = await get_label(url_info.url)
        return result
    except HTTPException as e:
        return {"error": str(e)}

# @app.post("/img/")
# async def get_label(url: str):
#     try:
#         response = requests.get(url, proxies=None)
#         response.raise_for_status()
#     except requests.RequestException as e:
#         raise HTTPException(status_code=500, detail=f"Failed to download the image from {url}")
#
#     image = Image.open(BytesIO(response.content))
#     image = processor(images=image, return_tensors="pt")
#
#     with torch.no_grad():
#         image_features = clip_model.get_image_features(**image)
#         text_features = text_encoder(text).logits
#         image_features = image_features / image_features.norm(dim=1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=1, keepdim=True)
#         logit_scale = clip_model.logit_scale.exp()
#
#         logits_per_image = logit_scale * image_features @ text_features.t()
#         probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#         result_label = labels[np.argmax(probs)]
#
#     return JSONResponse(content={"url": url, "label": result_label})
#
# # You can run the FastAPI application using the following command:
# # uvicorn your_script_name:app --reload
