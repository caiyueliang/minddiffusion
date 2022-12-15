import sys
import argparse
import logging
from flask import Flask
sys.path.append("./")

from src.diffusion import Diffusion

app = Flask(__name__)
logging.getLogger().setLevel(level=logging.DEBUG)

@app.route('/')
def hello_world():
   return 'Hello World'


@app.route('/predict', methods=['POST'])
def predict():
   text = "一只可爱的猫坐在草地上"
   logging.info("[predict] start text:{} ...".format(text))
   result = Diffusion().predict(prompt=text)
   return result


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

