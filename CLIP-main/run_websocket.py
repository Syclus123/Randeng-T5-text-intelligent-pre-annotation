'''
-*- coding: utf-8 -*-
@File  : run_websocket.py
@Time  : 2023/11/12 16:40
'''


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
# app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/static", StaticFiles(directory="./static"), name="static")

class Item(BaseModel):
    url: str

# labels = ["猫", "狗", '猪', '虎']

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
        else:
            result = f"Failed to download the image from {url}"
            await websocket.send_text(result)
    except Exception as e:
        result = f"Error processing image: {str(e)}"
        await websocket.send_text(result)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        item = Item(url=data)
        await process_url(websocket, item)

if __name__ == "__main__":
    import uvicorn
    host_name = socket.gethostname()
    ip_address = socket.gethostbyname(host_name)
    print(f"Python server is running at {ip_address}:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from io import BytesIO
# from PIL import Image
# import requests
# import torch
# from transformers import BertForSequenceClassification, BertTokenizer
# from transformers import CLIPProcessor, CLIPModel
# import numpy as np
# import csv
# import socket
#
#
# app = FastAPI()
#
# class Item(BaseModel):
#     url: str
#
# # labels = ["猫", "狗", '猪', '虎']
#
# with open('./class_name/imagenet1000.csv', 'r', encoding='GBK') as csv_file:
#     csv_reader = csv.reader(csv_file)
#     labels = [row[0] for row in csv_reader]
#
# text_tokenizer = BertTokenizer.from_pretrained("./taiyiclip")
# text_encoder = BertForSequenceClassification.from_pretrained("./taiyiclip").eval()
# text = text_tokenizer(labels, return_tensors='pt', padding=True)['input_ids']
#
# clip_model = CLIPModel.from_pretrained("clip32/")
# processor = CLIPProcessor.from_pretrained("clip32/")
#
# def get_label(url):
#     response = requests.get(url, proxies=None)
#     if response.status_code == 200:
#         image = Image.open(BytesIO(response.content))
#         image = processor(images=image, return_tensors="pt")
#         with torch.no_grad():
#             image_features = clip_model.get_image_features(**image)
#             text_features = text_encoder(text).logits
#             image_features = image_features / image_features.norm(dim=1, keepdim=True)
#             text_features = text_features / text_features.norm(dim=1, keepdim=True)
#             logit_scale = clip_model.logit_scale.exp()
#             logits_per_image = logit_scale * image_features @ text_features.t()
#             probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#             print(url + "的label是" + labels[np.argmax(probs)])
#     else:
#         print(f"Failed to download the image from {url}")
#
# @app.post("/process_image")
# async def process_image(item: Item):
#     url = item.url
#     try:
#         get_label(url)
#         return {"message": "Image processed successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
#
# if __name__ == "__main__":
#     import uvicorn
#     host_name = socket.gethostname()
#     ip_address = socket.gethostbyname(host_name)
#     print(f"Python server is running at {ip_address}:8000")
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
