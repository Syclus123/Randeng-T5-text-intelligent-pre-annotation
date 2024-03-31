'''
-*- coding: utf-8 -*-
@File  : run_model.py
@Time  : 2023/11/18 21:42
'''
# -*- coding: utf-8 -*-

import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import socket

# 定义一个模型来接收请求数据
class Item(BaseModel):
    text: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

pretrained_model = "./T5-77M-CH"

special_tokens = ["<extra_id_{}>".format(i) for i in range(100)]  # 添加定义特殊token
tokenizer = T5Tokenizer.from_pretrained(  # T5tokenizer处理文本
    pretrained_model,
    do_lower_case=True,  # 是否转小写
    max_length=512,  # token最大长度
    truncation=True,  # 是否截断输入文本以适应模型的最大长度
    additional_special_tokens=special_tokens,  # 添加额外的特殊token
    legacy=True, #消除T5Tokenizer 的警告
)

config = T5Config.from_pretrained(pretrained_model)  # 加载预训练模型配置
model = T5ForConditionalGeneration.from_pretrained(pretrained_model, config=config)  # T5生成模型，载入配置
model.resize_token_embeddings(len(tokenizer))  # 调整模型的嵌入层，使其包含新的token


@app.post("/predict/")
async def predict(item: Item):
    encode_dict = tokenizer(item.text, max_length=1024, padding='max_length', truncation=True)

    inputs = {
        "input_ids": torch.tensor([encode_dict['input_ids']]).long(),
        "attention_mask": torch.tensor([encode_dict['attention_mask']]).long(),
    }

    logits = model.generate(
        input_ids=inputs['input_ids'],
        max_length=100,  # 生成的最大长度
        early_stopping=True,  # 提前停止生成
    )

    logits = logits[:, 1:]
    predict_label = [tokenizer.decode(i, skip_special_tokens=True) for i in logits]

    print(predict_label)
    return {"prediction": predict_label}

# @app.get("/predict/")
# async def predict(item: Item):
#     encode_dict = tokenizer(item.text, max_length=1024, padding='max_length', truncation=True)
#
#     inputs = {
#         "input_ids": torch.tensor([encode_dict['input_ids']]).long(),
#         "attention_mask": torch.tensor([encode_dict['attention_mask']]).long(),
#     }
#
#     logits = model.generate(
#         input_ids=inputs['input_ids'],
#         max_length=100,  # 生成的最大长度
#         early_stopping=True,  # 提前停止生成
#     )
#
#     logits = logits[:, 1:]
#     predict_label = [tokenizer.decode(i, skip_special_tokens=True) for i in logits]
#     print(predict_label)
#     return {"prediction": predict_label}

#
# if __name__ == "__main__":
#     host_name = socket.gethostname()
#     ip_address = socket.gethostbyname(host_name)
#     print(f"Python server is running at {ip_address}:6666")  # 打印端口
#     uvicorn.run("main:app", host="127.0.0.1", port=6666, reload=True)


