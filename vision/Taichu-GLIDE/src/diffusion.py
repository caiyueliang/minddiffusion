#!/usr/bin/env python3
# coding=utf-8
"""
Author: changwanli
since: 2022-11-07 09:52:36
LastTime: 2022-11-14 11:12:44
LastAuthor: changwanli
message:
Copyright (c) 2022 Wuhan Artificial Intelligence Research. All Rights Reserved
"""

import os
import json
import logging

import mindspore
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from model.glide_utils.img_utils import save_images
from model.glide_text2im.tokenizer.bpe import get_encoder
from model.glide_text2im.tokenizer.chinese_tokenizer import from_pretrained
from model.glide_text2im.tokenizer.caption_to_tokens import convert_input_to_token_gen, convert_input_to_token_super_res
from model.glide_text2im.default_options import model_and_diffusion_defaults, model_and_diffusion_upsample
from model.glide_text2im.diffusion_creator import init_diffusion_model, init_super_res_model
from model.glide_text2im.main_funcs import gaussian_p_sample_loop, ddim_sample_loop
from model.glide_text2im.model.srgan_util import SRGAN

from threading import RLock

from src.alluxio.s3 import send_directory_to

class Diffusion(object):
    single_lock = RLock()       # 上锁
    init_flag = False

    def __init__(self, args=None):
        if self.init_flag is False:
            logging.warning("[Diffusion] init start. ")
            logging.warning("[Diffusion] args: {}".format(args))
            self.init(args=args)
            logging.warning("[Diffusion] init finish. ")
            self.init_flag = True

        return

    def __new__(cls, *args, **kwargs):
        with Diffusion.single_lock:
            if not hasattr(Diffusion, "_instance"):
                Diffusion._instance = object.__new__(cls)

        return Diffusion._instance

    def load_ckpt(self, net, ckpt_file, model_type="base"):
        if not ckpt_file:
            return
        logging.info(f"start loading ckpt:{ckpt_file}")
        param_dict = load_checkpoint(ckpt_file)
        # logging.info(f"[load_ckpt] param_dict: {param_dict}")
        new_param_dict = {}
        for key, val in param_dict.items():
            keyL = key.split(".")
            new_keyL = []
            for para in keyL:
                if para == "diffusion_with_p_sample":
                    continue
                else:
                    new_keyL.append(para)
                if para == "guider_net" and model_type == "base":
                    new_keyL.append("model")
            new_key = ".".join(new_keyL)
            new_param_dict[new_key] = val
        if param_dict:
            param_not_load = load_param_into_net(net, new_param_dict)
            logging.info("param not load: {}".format(param_not_load))
        logging.info(f"end loading ckpt:{ckpt_file}")

    def read_prompts_file(self, file):
        prompts = []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                txt = line.strip()
                prompts.append(txt)
        return prompts

    def get_random_captions(self, file):
        captions = []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                caption = line.strip().split('\t')[1]
                captions.append(caption)
        logging.info(f"the num of all random captions: {len(captions)}")
        return captions

    def init(self, args):
        mindspore.context.set_context(mode=mindspore.context.GRAPH_MODE, device_id=0)

        # hyper-params
        self.output_path = args.output_path
        chinese = args.is_chinese
        self.denoise_steps = args.denoise_steps
        self.super_res_step = args.super_res_step
        guidance_scale = args.guidance_scale
        self.pics_generated = args.pics_generated
        self.tokenizer = from_pretrained(os.path.join(args.ckpt_path, args.tokenizer_model)) if chinese else get_encoder()
        ckpt_path_gen = os.path.join(args.ckpt_path, args.gen_ckpt)
        ckpt_path_super_res = os.path.join(args.ckpt_path, args.super_ckpt)
        ckpt_path_srgan = os.path.join(args.ckpt_path, args.srgan_ckpt)

        self.options = model_and_diffusion_defaults(timestep_respacing=str(self.denoise_steps), dtype=mindspore.float16)
        options_up = model_and_diffusion_upsample(dtype=mindspore.float16)
        self.input_shape = (self.pics_generated * 2, 3, self.options["image_size"], self.options["image_size"])
        self.up_shape = (self.pics_generated, 3, options_up["image_size"], options_up["image_size"])

        logging.info("Initializing models...")
        # base model for 64*64 generative
        self.diffusion_model = init_diffusion_model(options=self.options,
                                               guidance_scale=guidance_scale,
                                               shape=self.input_shape,
                                               ckpt_path=ckpt_path_gen)
        self.load_ckpt(self.diffusion_model, ckpt_path_gen)

        # super res model 64*64 to 256*256
        self.super_res_model = init_super_res_model(options=options_up, shape=self.up_shape)
        self.load_ckpt(self.super_res_model, ckpt_path_super_res, model_type="supres")

        # super res model 256*256 to 1024*1024
        self.srgan = SRGAN(4, ckpt_path_srgan)

    def predict(self, uuid, prompt):
        logging.info("read prompts_file...")
        # prompts = self.read_prompts_file(args.prompts_file)
        # for prompt in prompts:

        output_dir = os.path.join(self.output_path, uuid)
        os.makedirs(name=output_dir, exist_ok=True)
        obs_upload_to = "server/text2image/diffusion_glide_mindspore/{}/".format(uuid)

        ori_image_path = os.path.join(output_dir, prompt + ".jpg")
        upx4_image_path = os.path.join(output_dir, prompt + "_up256.jpg")
        upx16_image_path = os.path.join(output_dir, prompt + "_up1024.jpg")

        # Sample from the base model.
        token, mask = convert_input_to_token_gen(prompt,
                                                 self.pics_generated,
                                                 self.options['text_ctx'],
                                                 self.tokenizer)
        samples = gaussian_p_sample_loop(diffusion_model=self.diffusion_model,
                                         token=token,
                                         mask=mask,
                                         shape=self.input_shape,
                                         num_timesteps=self.denoise_steps,
                                         tokenizer=self.tokenizer,
                                         text_ctx=self.options['text_ctx'],
                                         progress=True,
                                         dtype=self.options["dtype"],
                                         vocab_len=self.options["n_vocab"])[:self.pics_generated]
        save_images(samples, ori_image_path)

        logging.warning("部分图片生成成功，开始上传到obs/minio路径: {}".format(obs_upload_to))
        send_directory_to(local_directory=output_dir, s3_directory_name=obs_upload_to)

        token, mask = convert_input_to_token_super_res(prompt,
                                                       self.pics_generated,
                                                       self.options['text_ctx'],
                                                       self.tokenizer)

        samples = ddim_sample_loop(super_res_model=self.super_res_model,
                                   samples=samples,
                                   token=token,
                                   mask=mask,
                                   up_shape=self.up_shape,
                                   num_timesteps=self.super_res_step,
                                   progress=True, dtype=self.options["dtype"])
        save_images(samples, upx4_image_path)

        samples = self.srgan.sr_handle(mindspore.ops.Cast()(samples, mindspore.float32))  # use fp32
        save_images(samples, upx16_image_path)

        # 文件上传到obs/minio
        logging.warning("图片生成成功，开始上传到obs/minio路径: {}".format(obs_upload_to))
        send_directory_to(local_directory=output_dir, s3_directory_name=obs_upload_to)

        obs_upload_to = "server/text2image/diffusion_glide_mindspore/{}/".format(uuid)

        obs_ori_image_path = obs_upload_to + prompt + ".jpg"
        obs_upx4_image_path = obs_upload_to + prompt + "_up256.jpg"
        obs_upx16_image_path = obs_upload_to + prompt + "_up1024.jpg"
        logging.warning("obs_ori_image_path: {}".format(obs_ori_image_path))
        logging.warning("obs_upx4_image_path: {}".format(obs_upx4_image_path))
        logging.warning("obs_upx16_image_path: {}".format(obs_upx16_image_path))

        result = {"infer_result": "success",
                  "image_dir": output_dir,
                  "image_list": [
                      {"size": "64*64", "url": obs_ori_image_path},
                      {"size": "256*256", "url": obs_upx4_image_path},
                      {"size": "1024*1024", "url": obs_upx16_image_path}
                  ]}
        result = json.dumps(result)
        # infe_result_path = os.path.join(self.output_path, "infer_result.json")
        # with open(infe_result_path, 'w', encoding='utf-8') as f:
        #     f.write(result)
        return result
