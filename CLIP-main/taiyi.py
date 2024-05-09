'''
-*- coding: utf-8 -*-
@File  : taiyi.py
@Time  : 2023/11/06 17:22
'''
from io import BytesIO

from PIL import Image
import requests
# import clip
import torch
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import csv

# labels = ["猫", "狗",'猪', '虎']  # 这里是输入文本的，可以随意替换。

# 21841类标签
# with open('./class_name/class_cn.csv', 'r', encoding='GBK') as csv_file:
#     csv_reader = csv.reader(csv_file)
#     # 从CSV文件中读取第一列数据并存储为列表
#     labels = [row[0] for row in csv_reader]
# # print(labels)

# 1000类标签
with open('./class_name/imagenet1000.csv', 'r', encoding='GBK') as csv_file:
    csv_reader = csv.reader(csv_file)
    # 从CSV文件中读取第一列数据并存储为列表
    labels = [row[0] for row in csv_reader]
# print(labels)

#################################模型加载########################################

# 加载Taiyi 中文 text encoder
text_tokenizer = BertTokenizer.from_pretrained("./taiyiclip")
text_encoder = BertForSequenceClassification.from_pretrained("./taiyiclip").eval()
text = text_tokenizer(labels, return_tensors='pt', padding=True)['input_ids']


# 加载CLIP的image encoder
clip_model = CLIPModel.from_pretrained("clip32/")
processor = CLIPProcessor.from_pretrained("clip32/")

# #################################本地处理######################################
#
# def get_label(url):
#     image = processor(images=Image.open(url), return_tensors="pt")
#     with torch.no_grad():
#         image_features = clip_model.get_image_features(**image)
#         text_features = text_encoder(text).logits
#         # 归一化
#         image_features = image_features / image_features.norm(dim=1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=1, keepdim=True)
#         # 计算余弦相似度 logit_scale是尺度系数
#         logit_scale = clip_model.logit_scale.exp()
#
#         logits_per_image = logit_scale * image_features @ text_features.t()
#         probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#         print(url+"的label是"+labels[np.argmax(probs)])
# url = "./image_test/cat.jpg"
# get_label(url)
#################################URL处理######################################

def get_label(url):
    response = requests.get(url,proxies=None)
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
            print(url + "的label是" + labels[np.argmax(probs)])
    else:
        print(f"Failed to download the image from {url}")

url = "https://items-storage.oss-cn-beijing.aliyuncs.com/7fab767a6f214935bc9024cc4a723f2c-%E7%89%9B%E5%B0%BE%E8%8A%B1/V1/Snipaste_2023-11-08_16-35-50.png"
get_label(url)

