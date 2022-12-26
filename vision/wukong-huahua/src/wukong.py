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
import cv2
import logging
import numpy as np
import mindspore as ms
from itertools import islice
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler


from PIL import Image

from obs import PutObjectHeader

from threading import RLock

# from src.alluxio.s3 import send_directory_to
from src.alluxio.hw_obs import cube_bucket, obsClient


class WuKong(object):
    single_lock = RLock()       # 上锁
    init_flag = False

    def __init__(self, args=None):
        if self.init_flag is False:
            logging.warning("[WuKong] init start. ")
            logging.warning("[WuKong] args: {}".format(args))
            self.opt = args

            self.init(opt=args)

            # 最开始先进行一次预测
            self.predict(uuid="init", prompt="测试", n_iter=1, n_samples=1, H=128, W=128, scale=7.5, ddim_steps=1)

            logging.warning("[WuKong] init finish. ")
            self.init_flag = True

        return

    def __new__(cls, *args, **kwargs):
        with WuKong.single_lock:
            if not hasattr(WuKong, "_instance"):
                WuKong._instance = object.__new__(cls)

        return WuKong._instance

    @staticmethod
    def seed_everything(seed):
        if seed:
            ms.set_seed(seed)
            np.random.seed(seed)

    @staticmethod
    def chunk(it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @staticmethod
    def load_model_from_config(config, ckpt, verbose=False):
        logging.warning(f"Loading model from {ckpt}")
        model = instantiate_from_config(config.model)
        if os.path.exists(ckpt):
            param_dict = ms.load_checkpoint(ckpt)
            if param_dict:
                param_not_load = ms.load_param_into_net(model, param_dict)
                logging.warning("param not load: {}".format(param_not_load))
        else:
            logging.warning(f"{ckpt} not exist:")

        return model

    @staticmethod
    def put_watermark(img, wm_encoder=None):
        if wm_encoder is not None:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = wm_encoder.encode(img, 'dwtDct')
            img = Image.fromarray(img[:, :, ::-1])
        return img

    @staticmethod
    def load_replacement(x):
        try:
            hwc = x.shape
            y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
            y = (np.array(y) / 255.0).astype(x.dtype)
            assert y.shape == x.shape
            return y
        except Exception:
            return x

    @staticmethod
    def image_split(image_file, num=2):
        split_dict = dict()

        img = Image.open(image_file)        # 读入当前图片
        # img = img.convert('RGB')          # 转换成RGB三通道格式
        w = img.size[0]  # 获取图片宽度
        h = img.size[1]  # 获取图片高度

        for i in range(num):
            sub_img = img.crop([w / num * i, 0, w / num * (i + 1), h])
            new_file = image_file.replace(".jpg", "_{}.jpg".format(i))
            sub_img.save(new_file)
            split_dict[i] = new_file

        return split_dict

    def init(self, opt):
        device_id = int(os.getenv("DEVICE_ID", 0))
        ms.context.set_context(
            mode=ms.context.GRAPH_MODE,
            device_target="Ascend",
            device_id=device_id,
            max_device_memory="30GB"
        )

        self.seed_everything(opt.seed)

        config = OmegaConf.load(f"{opt.config}")
        self.model = self.load_model_from_config(config, f"{opt.ckpt}")
        self.sampler = PLMSSampler(self.model)

        os.makedirs(opt.outdir, exist_ok=True)
        self.output_path = opt.outdir

        # self.batch_size = opt.n_samples
        # n_rows = opt.n_rows if opt.n_rows > 0 else self.batch_size

        # if not opt.from_file:
        #     prompt = opt.prompt
        #     assert prompt is not None
        #     data = [self.batch_size * [prompt]]
        # else:
        #     print(f"reading prompts from {opt.from_file}")
        #     with open(opt.from_file, "r") as f:
        #         data = f.read().splitlines()
        #         data = list(self.chunk(data, self.batch_size))

        # sample_path = os.path.join(outpath, "samples")
        # os.makedirs(sample_path, exist_ok=True)
        # base_count = len(os.listdir(sample_path))
        # grid_count = len(os.listdir(outpath)) - 1

    def predict(self, uuid, prompt, n_iter=1, n_samples=4, H=512, W=512, scale=7.5, ddim_steps=50):
        logging.info("[predict] start ...")

        output_dir = os.path.join(self.output_path, uuid)
        os.makedirs(name=output_dir, exist_ok=True)

        headers = PutObjectHeader()
        headers.contentType = 'text/plain'
        obs_upload_to = "server/text2image/diffusion_wukong_mindspore/{}/".format(uuid)

        batch_size = n_samples

        # prompt = opt.prompt
        # assert prompt is not None
        data = [batch_size * [prompt]]

        start_code = None
        if self.opt.fixed_code:
            std_normal = ms.ops.StandardNormal()
            start_code = std_normal((n_samples, 4, H // 8, W // 8))

        result = {"infer_result": "success", "images": []}
        all_samples = list()
        for n in range(n_iter):
            for prompts in data:
                logging.info("n: {}, prompts: {} ...".format(n, prompts))
                uc = None
                if scale != 1.0:
                    uc = self.model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = self.model.get_learned_conditioning(prompts)
                shape = [4, H // 8, W // 8]
                samples_ddim, _ = self.sampler.sample(S=ddim_steps,
                                                      conditioning=c,
                                                      batch_size=batch_size,
                                                      shape=shape,
                                                      verbose=False,
                                                      unconditional_guidance_scale=scale,
                                                      unconditional_conditioning=uc,
                                                      eta=self.opt.ddim_eta,
                                                      x_T=start_code)
                x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                x_samples_ddim = ms.ops.clip_by_value((x_samples_ddim + 1.0) / 2.0,
                                                      clip_value_min=0.0, clip_value_max=1.0)
                x_samples_ddim_numpy = x_samples_ddim.asnumpy()

                if not self.opt.skip_save:
                    for x_sample in x_samples_ddim_numpy:
                        x_sample = 255. * x_sample.transpose(1, 2, 0)
                        img = Image.fromarray(x_sample.astype(np.uint8))

                        base_count = len(os.listdir(output_dir))
                        local_image = os.path.join(output_dir, f"{base_count:05}.png")
                        img.save(local_image)
                        base_count += 1

                        # 图片上传obs
                        image_name = os.path.basename(local_image)
                        obs_image_path = os.path.join(obs_upload_to, image_name)
                        obsClient.putFile(cube_bucket, obs_image_path, local_image, metadata={}, headers=headers)

                        obs_image = "https://publish-data.obs.cn-central-221.ovaijisuan.com/{}".format(obs_image_path)
                        result["images"].append(obs_image)
                        logging.info("image_path: {}".format(obs_image))

                if not self.opt.skip_grid:
                    all_samples.append(x_samples_ddim_numpy)

            # print(f"Your samples are ready and waiting for you here: \n{outpath} \n\nEnjoy.")

        return result
