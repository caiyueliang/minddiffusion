#!/usr/bin/env python3
# coding=utf-8
"""
Author: changwanli
since: 2022-11-08 06:25:56
LastTime: 2022-11-10 17:04:53
LastAuthor: changwanli
message:
Copyright (c) 2022 Wuhan Artificial Intelligence Research. All Rights Reserved
"""
import requests
import json
import os


def test(data, host):
    """
    [summary]

    Args:
        line ([type]): [description]
        mention ([type], optional): [description]. Defaults to None.
        port (int, optional): [description]. Defaults to 8080.

    Returns:
        [type]: [description]
    """
    my_url = 'http://{host}/predict'.format(host=host)
    # print(json.dumps(data, ensure_ascii=False, indent=4))
    my_params = json.dumps(data)
    header = {'content-type': 'application/json'}

    r = requests.post(url=my_url, data=my_params, headers=header, timeout=36000)
    print("[test] response: {}".format(r))
    return json.loads(r.text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default="127.0.0.1:8080", type=str, help='host of service')
    parser.add_argument('--text', default="人在太阳地下劳动", type=str)
    args = parser.parse_args()
    data = { "text": args.text}
    res = test(data, host=args.host)
    print(json.dumps(res, ensure_ascii=False, indent=4))
