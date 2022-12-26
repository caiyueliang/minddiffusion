import os
import sys
import argparse
import logging
import json
import uuid
from flask import Flask, request, jsonify
from flask import current_app, make_response, send_file
from flask.globals import session

sys.path.append("./")

from src.wukong import WuKong
from src.log_utils import set_log_file


app = Flask(__name__)


def response(code, **kwargs):
    """
        Generic HTTP JSON response method

        :param code: HTTP code (int)
        :param kwargs: Data structure for response (dict)
        :return: HTTP Json response
    """
    # 添flash的信息
    # flashes = session.get("_flashes", [])

    # flashes.append((category, message))
    # session["_flashes"] = []

    _ret_json = jsonify(kwargs)
    resp = make_response(_ret_json, code)
    flash_json = []
    # for f in flashes:
    #     flash_json.append([f[0], f[1]])
    resp.headers["api_flashes"] = json.dumps(flash_json)
    resp.headers["Content-Type"] = "application/json; charset=utf-8"
    return resp


@app.route('/health', methods=['POST', 'GET'])
def health():
    message = {
        "status": 0,
        "message": "success"
    }
    return response(200, **message)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        req = request.json
        text = req.get('text', None)
        style = req.get('style', None)
        n_samples = req.get('n_samples', 1)
        n_iter = req.get('n_iter', 1)
        h = req.get('h', 512)
        w = req.get('w', 512)

        scale = req.get('scale', 7.5)
        ddim_steps = req.get('ddim_steps', 50)

        my_uuid = str(uuid.uuid1())

        logging.warning("[predict][{}] start text:{}, style:{}, n_iter:{}, n_samples:{}, scale:{}, ddim_steps:{}, size: ({}, {}) ...".format(
            my_uuid, text, style, n_iter, n_samples, scale, ddim_steps, h, w))

        if text is not None:
            if style is not  None:
                prompt = text + " " + style
            else:
                prompt = text

            msg = WuKong().predict(uuid=my_uuid, prompt=prompt, n_iter=n_iter, n_samples=n_samples, H=h, W=w,
                                   scale=scale, ddim_steps=ddim_steps)

            message = {
                "status": 0,
                "message": "success",
                "data": msg
            }
            return response(200, **message)
        else:
            message = {
                "status": 200,
                "message": "failed",
                "data": "parameter [text] is need"
            }
            return response(200, **message)
    except Exception as e:
        logging.exception(e)
        message = {
            "status": 0,
            "message": "failed",
            "data": {
                "exception": str(e)
            }
        }
        return response(500, **message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="狗 绘画 写实风格",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./outputs/"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v1-inference-chinese.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./pretraind_models/wukong-huahua-ms.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument('--log_file', default="./log/server.log", type=str, help='log dir')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    set_log_file(logger_obj=logging.getLogger(), filename=args.log_file)

    logging.warning(args)

    WuKong(args=args)

    app.run(host='0.0.0.0', port=8080)

