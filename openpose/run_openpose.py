import config
from pathlib import Path
import sys
import os
import cv2
import einops
import numpy as np
import random
import time
import json
import torch
from PIL import Image

# 尝试导入 NPU 适配库
try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

# 确保项目路径正确
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.annotator.util import resize_image, HWC3
from preprocess.openpose.annotator.openpose import OpenposeDetector

class OpenPose:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        # --- NPU/CUDA 兼容性设置 ---
        if HAS_NPU:
            # 昇腾芯片必须显式指定 device
            torch.npu.set_device(gpu_id)
            print(f"--- OpenPose: 使用昇腾 NPU (ID: {gpu_id}) ---")
        else:
            torch.cuda.set_device(gpu_id)
            print(f"--- OpenPose: 使用 NVIDIA GPU (ID: {gpu_id}) ---")
        
        self.preprocessor = OpenposeDetector()

    def __call__(self, input_image, resolution=384):
        # 每次调用确保在正确设备上
        if HAS_NPU:
            torch.npu.set_device(self.gpu_id)
        else:
            torch.cuda.set_device(self.gpu_id)
            
        # 转换输入图像为 numpy 数组
        if isinstance(input_image, Image.Image):
            input_image = np.asarray(input_image)
        elif isinstance(input_image, str):
            if not os.path.exists(input_image):
                raise FileNotFoundError(f"找不到图像文件: {input_image}")
            input_image = np.asarray(Image.open(input_image))
        else:
            raise ValueError("不支持的输入格式，请传入 PIL.Image 或 文件路径")
            
        with torch.no_grad():
            input_image = HWC3(input_image)
            input_image = resize_image(input_image, resolution)
            H, W, C = input_image.shape
            
            # OOTDiffusion 要求的固定尺寸校验
            if H != 512 or W != 384:
                input_image = cv2.resize(input_image, (384, 512))
            
            # 开始检测
            pose, detected_map = self.preprocessor(input_image, hand_and_face=False)

            # 提取关键点
            if len(pose['bodies']['subset']) == 0:
                print("警告：未在图片中检测到人体骨架")
                return {"pose_keypoints_2d": [[0, 0]] * 18}

            candidate = pose['bodies']['candidate']
            subset = pose['bodies']['subset'][0][:18]
            
            # 关键点补齐逻辑 (18位标准骨架)
            for i in range(18):
                if subset[i] == -1:
                    candidate.insert(i, [0, 0])
                    for j in range(i, 18):
                        if(subset[j]) != -1:
                            subset[j] += 1
                elif subset[i] != i:
                    candidate.pop(i)
                    for j in range(i, 18):
                        if(subset[j]) != -1:
                            subset[j] -= 1

            candidate = candidate[:18]

            # 映射回 384x512 的坐标空间
            for i in range(18):
                candidate[i][0] *= 384
                candidate[i][1] *= 512

            keypoints = {"pose_keypoints_2d": candidate}

        # 显存清理：在 HPC 环境下非常重要
        if HAS_NPU:
            torch.npu.empty_cache()
            
        return keypoints

if __name__ == '__main__':
    # 测试代码
    model = OpenPose(gpu_id=0)
    # 你可以修改下面的路径来测试你的 TaskB 图片
    test_img_path = '../TaskB/character/1.jpg' 
    
    if os.path.exists(test_img_path):
        try:
            res = model(test_img_path)
            print(f"检测成功！检测到 {len(res['pose_keypoints_2d'])} 个关键点")
            print("第一个关键点坐标:", res['pose_keypoints_2d'][0])
        except Exception as e:
            print(f"推理发生错误: {e}")
    else:
        print(f"找不到测试图片: {test_img_path}")