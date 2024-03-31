'''
-*- coding: utf-8 -*-
@File  : main.py
@Time  : 2023/11/18 21:54
'''
import uvicorn
import socket
from run_model import app


if __name__ == "__main__":
    host_name = socket.gethostname()
    ip_address = socket.gethostbyname(host_name)
    print(f"Python server is running at {ip_address}:6666")  # 打印端口
    uvicorn.run("main:app", host="127.0.0.1", port=3000, reload=True)