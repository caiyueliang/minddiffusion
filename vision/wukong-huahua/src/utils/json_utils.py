# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
-------------------------------------------------
   Description :
   Author :       caiyueliang
   Date :         2018/11/26
-------------------------------------------------

"""
import json


# 部分需要展示的字段保存在mysql中
def load_json(file_path):
    with open(file_path) as f:
        content = json.load(fp=f)
        return content


# 保存自动调參的超参数，本地保存
def save_json(file_path, content):
    with open(file_path, 'w') as f:
        json.dump(obj=content, fp=f, indent=4)


# 将 Python 对象编码成 JSON 字符串
def dumps_json(data):
    json_data = json.dumps(data)
    return json_data


# 将已编码的 JSON 字符串解码为 Python 对象
def loads_json(str_data):
    json_data = json.dumps(str_data)
    return json_data
