from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import os
import torch
# 必须导入 torch_npu 才能在昇腾芯片上运行
try:
    import torch_npu
except ImportError:
    pass

import numpy as np
from PIL import Image
import cv2

import random
import time

from pipelines_ootd.pipeline_ootd import OotdPipeline
from pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel
from pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel
from diffusers import UniPCMultistepScheduler
from diffusers import AutoencoderKL

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTokenizer

VIT_PATH = "../checkpoints/clip-vit-large-patch14"
VAE_PATH = "../checkpoints/ootd"
UNET_PATH = "../checkpoints/ootd/ootd_dc/checkpoint-36000"
MODEL_PATH = "../checkpoints/ootd"

class OOTDiffusionDC:

    def __init__(self, gpu_id):
        # --- 原生 NPU 适配修改 1: 设备定义 ---
        # 将 'cuda:' 替换为 'npu:'
        self.device = torch.device(f"npu:{gpu_id}")
        torch.npu.set_device(self.device)
        print(f"--- 模型初始化：使用设备 {self.device} ---")

        vae = AutoencoderKL.from_pretrained(
            VAE_PATH,
            subfolder="vae",
            torch_dtype=torch.float16,
        )

        unet_garm = UNetGarm2DConditionModel.from_pretrained(
            UNET_PATH,
            subfolder="unet_garm",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        unet_vton = UNetVton2DConditionModel.from_pretrained(
            UNET_PATH,
            subfolder="unet_vton",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        # --- 原生 NPU 适配修改 2: Pipeline 搬运 ---
        self.pipe = OotdPipeline.from_pretrained(
            MODEL_PATH,
            unet_garm=unet_garm,
            unet_vton=unet_vton,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)  # 使用刚才定义的 npu device 对象

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)

        self.auto_processor = AutoProcessor.from_pretrained(VIT_PATH)

        # --- 原生 NPU 适配修改 3: Encoder 搬运 ---
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH).to(self.device)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_PATH,
            subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            MODEL_PATH,
            subfolder="text_encoder",
        ).to(self.device)


    def tokenize_captions(self, captions, max_length):
        inputs = self.tokenizer(
            captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


    def __call__(self,
                model_type='dc',
                category='upperbody',
                image_garm=None,
                image_vton=None,
                mask=None,
                image_ori=None,
                num_samples=1,
                num_steps=20,
                image_scale=1.0,
                seed=-1,
    ):
        if seed == -1:
            random.seed(time.time())
            seed = random.randint(0, 2147483647)
        print('Initial seed: ' + str(seed))

        # 显存优化：NPU 在推理前建议清理一下
        torch.npu.empty_cache()

        generator = torch.manual_seed(seed)

        with torch.no_grad():
            # --- 原生 NPU 适配修改 4: 推理时的数据搬运 ---
            prompt_image = self.auto_processor(images=image_garm, return_tensors="pt").to(self.device)
            prompt_image = self.image_encoder(prompt_image.data['pixel_values']).image_embeds
            prompt_image = prompt_image.unsqueeze(1)
            if model_type == 'hd':
                # 这里的 .to(self.device) 确保 token 在 NPU 上
                token_ids = self.tokenize_captions([""], 2).to(self.device)
                prompt_embeds = self.text_encoder(token_ids)[0]
                prompt_embeds[:, 1:] = prompt_image[:]
            elif model_type == 'dc':
                token_ids = self.tokenize_captions([category], 3).to(self.device)
                prompt_embeds = self.text_encoder(token_ids)[0]
                prompt_embeds = torch.cat([prompt_embeds, prompt_image], dim=1)
            else:
                raise ValueError("model_type must be \'hd\' or \'dc\'!")

            # 执行推理
            images = self.pipe(prompt_embeds=prompt_embeds,
                        image_garm=image_garm,
                        image_vton=image_vton,
                        mask=mask,
                        image_ori=image_ori,
                        num_inference_steps=num_steps,
                        image_guidance_scale=image_scale,
                        num_images_per_prompt=num_samples,
                        generator=generator,
            ).images

        return images
