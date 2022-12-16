import sys
import argparse
import logging
import json
from flask import Flask, request, jsonify
from flask import current_app, make_response, send_file
from flask.globals import session

sys.path.append("./")

from src.diffusion import Diffusion

app = Flask(__name__)
logging.getLogger().setLevel(level=logging.DEBUG)

def response(code, **kwargs):
    """
        Generic HTTP JSON response method

        :param code: HTTP code (int)
        :param kwargs: Data structure for response (dict)
        :return: HTTP Json response
    """
    # 添flash的信息
    flashes = session.get("_flashes", [])

    # flashes.append((category, message))
    session["_flashes"] = []

    _ret_json = jsonify(kwargs)
    resp = make_response(_ret_json, code)
    flash_json = []
    for f in flashes:
        flash_json.append([f[0], f[1]])
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
   req = request.json
   text = req.get('text', None)

   # text = "一只可爱的猫坐在草地上"
   logging.info("[predict] start text:{} ...".format(text))
   # result = Diffusion().predict(prompt=text)

   message = {
      "status": 0,
      "message": "success",
      "data": {
         "obs": text
      }
   }

   return response(200, **message)


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

   args = parser.parse_args()
   logging.warning(args)

   Diffusion(args=args)

   app.run(host='0.0.0.0', port=8080)

