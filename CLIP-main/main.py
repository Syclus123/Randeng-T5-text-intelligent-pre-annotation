'''
-*- coding: utf-8 -*-
@File  : main.py
@Time  : 2023/11/12 17:18
'''
# main.py

from taiyi_http import app

import uvicorn
import socket

if __name__ == "__main__":
    # 启动服务，因为我们这个文件叫做 main.py，所以需要启动 main.py 里面的 app
    # 然后是 host 和 port 表示监听的 ip 和端口
    host_name = socket.gethostname()
    ip_address = socket.gethostbyname(host_name)
    print(f"Python server is running at {ip_address}:6666") #打印端口
    uvicorn.run("main:app", host="127.0.0.1", port=10010,reload=True)
    # home.itzyc.com: 12349
    # http://home.itzyc.com:12349