'''
-*- coding: utf-8 -*-
@File  : translate.py
@Time  : 2023/11/06 22:36
'''

import httplib2
import urllib
import random
import json
from hashlib import md5

appid = '*********'  # 你的appid
secretKey = '********'  # 你的密钥

httpClient = None
myurl = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
q = 'apple'  # 要翻译的词
fromLang = 'en'  # 翻译源语言
toLang = 'zh'  # 译文语言
salt = random.randint(32768, 65536)

# 签名
sign = appid + q + str(salt) + secretKey
m1 = md5()
m1.update(sign.encode(encoding='utf-8'))
sign = m1.hexdigest()
# myurl = myurl+'?appid='+appid+'&q='+urllib.parse.quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign
myurl = myurl + '?q=' + urllib.parse.quote(
    q) + '&from=' + fromLang + '&to=' + toLang + '&appid=' + appid + '&salt=' + str(salt) + '&sign=' + sign
try:
    h = httplib2.Http('.cache')
    response, content = h.request(myurl)
    if response.status == 200:
        print(content.decode('utf-8'))
        print(type(content))
        response = json.loads(content.decode('utf-8'))  # loads将json数据加载为dict格式
        print(type(response))
        print(response["trans_result"][0]['dst'])
except httplib2.ServerNotFoundError:
    print("Site is Down!")
