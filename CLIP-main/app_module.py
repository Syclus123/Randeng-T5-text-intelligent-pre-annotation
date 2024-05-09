'''
-*- coding: utf-8 -*-
@File  : app_module.py
@Time  : 2023/11/12 17:16
'''
# app_module.py
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import requests
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import csv
import asyncio
import socket

app = FastAPI()

# Serve static files (e.g., HTML, JS) for WebSocket communication
app.mount("/static", StaticFiles(directory="./static"), name="static")

class Item(BaseModel):
    url: str

# labels = ["猫", "狗", '猪', '虎']

# 21841类标签
# with open('./class_name/class_cn.csv', 'r', encoding='GBK') as csv_file:
#     csv_reader = csv.reader(csv_file)
#     # 从CSV文件中读取第一列数据并存储为列表
#     labels = [row[0] for row in csv_reader]
# # print(labels)

with open('./class_name/imagenet1000.csv', 'r', encoding='GBK') as csv_file:
    csv_reader = csv.reader(csv_file)
    labels = [row[0] for row in csv_reader]

text_tokenizer = BertTokenizer.from_pretrained("./taiyiclip")
text_encoder = BertForSequenceClassification.from_pretrained("./taiyiclip").eval()
text = text_tokenizer(labels, return_tensors='pt', padding=True)['input_ids']

clip_model = CLIPModel.from_pretrained("clip32/")
processor = CLIPProcessor.from_pretrained("clip32/")

async def process_url(websocket: WebSocket, item: Item):
    url = item.url
    print(f"Received URL: {url}")  # 添加这行打印语句
    try:
        response = requests.get(url, proxies=None)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = clip_model.get_image_features(**image)
                text_features = text_encoder(text).logits
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                logit_scale = clip_model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                result = url + "的label是" + labels[np.argmax(probs)]
                await websocket.send_text(result)
                await websocket.send_text("Data received and processed successfully")  # 发送确认消息
        else:
            result = f"Failed to download the image from {url}"
            await websocket.send_text(result)
            await websocket.send_text("Failed to process the data")  # 发送确认消息
    except Exception as e:
        result = f"Error processing image: {str(e)}"
        await websocket.send_text(result)
        await websocket.send_text("Failed to process the data")  # 发送确认消息

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        item = Item(url=data)
        await process_url(websocket, item)