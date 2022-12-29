import os
import sys
import argparse
import logging
import json
import uuid
from flask import Flask, request, jsonify
from flask import make_response

sys.path.append("./")

from obs import PutObjectHeader
from src.diffusion import Diffusion
from src.alluxio.hw_obs import cube_bucket, obsClient
from src.utils.log_utils import set_log_file

app = Flask(__name__)

logger = logging.getLogger()
logger.setLevel(level=logging.WARNING)

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
        pics_generated = req.get('pics_generated', 2)

        my_uuid = str(uuid.uuid1())

        logger.warning("[predict][{}] start text:{}, pics_generated: {} ...".format(my_uuid, text, pics_generated))
        # msg = Diffusion().predict(uuid=my_uuid, prompt=text, pics_generated=pics_generated)
        msg = Diffusion().predict_thread(uuid=my_uuid, prompt=text, pics_generated=pics_generated)

        message = {
            "status": 0,
            "message": "success",
            "data": msg
        }

        return response(200, **message)
    except Exception as e:
        logger.exception(e)
        message = {
            "status": 0,
            "message": "failed",
            "data": {
                "exception": str(e)
            }
        }
        return response(500, **message)


@app.route('/test', methods=['POST'])
def test():
    try:
        req = request.json
        text = req.get('text', None)

        logger.info("[test] start text:{} ...".format(text))

        obs_upload_to = "server/text2image/diffusion_glide_mindspore/scripts/run_server_docker.sh"

        local_dir = "/home/server/scripts/run_server_docker.sh"
        # 文件上传到obs/minio
        logger.warning("上传到obs/minio路径: {}".format(obs_upload_to))

        headers = PutObjectHeader()
        headers.contentType = 'text/plain'
        obsClient.putFile(cube_bucket, obs_upload_to, local_dir, metadata={}, headers=headers)

        message = {
            "status": 0,
            "message": "success",
            "data": {
                "obs": obs_upload_to
            }
        }

        return response(200, **message)
    except Exception as e:
        logger.exception(e)
        message = {
            "status": 0,
            "message": "failed",
            "data": {
                "exception": str(e)
            }
        }
        return response(500, **message)


def print_dir(root_dir):
    logger.info("[print_dir] start : {} ...".format(root_dir))
    for parent, _, fileNames in os.walk(root_dir):
            for name in fileNames:
                if name.startswith('.'):  # 去除隐藏文件
                    continue
                else:
                    logger.info("{}, {}".format(parent, name))
    logger.info("[print_dir] end ...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_chinese", default=True, type=bool, help="chinese or not")
    parser.add_argument("--denoise_steps", default=1, type=int, help="denoise steps")
    parser.add_argument("--super_res_step", default=1, type=int, help="super res step")
    parser.add_argument("--guidance_scale", default=5, type=int, help="guidance scale")
    parser.add_argument("--pics_generated", default=8, type=int, help="pic num for each prompt")

    parser.add_argument("--tokenizer_model", default="cog-pretrain.model", help="tokenizer model")
    parser.add_argument("--gen_ckpt", default="glide_gen.ckpt", help="gen ckpt")
    parser.add_argument("--super_ckpt", default="glide_super_res.ckpt", help="super ckpt")
    parser.add_argument("--srgan_ckpt", default="srgan.ckpt", help="srgan ckpt")

    parser.add_argument("--prompts_file", default="prompts.txt", help="prompts file")
    parser.add_argument("--data_path", default="", help="dataset path")
    parser.add_argument("--output_path", default="", help="output path")
    parser.add_argument("--ckpt_path", default="pretraind_models/", type=str, help="ckpt init path")
    parser.add_argument("--model_config_path", default="./configs/model_config.json", help="model_config")

    parser.add_argument('--log_file', default="./log/server.log", type=str, help='log dir')
    parser.add_argument('--thread_pool_size', default=1, type=int, help='thread pool size')

    args = parser.parse_args()

    # logging.basicConfig(level=logging.INFO)
    set_log_file(logger_obj=logger, filename=args.log_file)

    logger.warning(args)

    Diffusion(args=args)

    print_dir(root_dir="/home/server/log/")

    app.run(host='0.0.0.0', port=8080, threaded=False)

