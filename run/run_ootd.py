import os
import sys
import torch
import torch_npu  # 原生导入
from pathlib import Path
from PIL import Image
from utils_ootd import get_mask_location

# 设置项目根目录
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD

import argparse

# 开启昇腾算子并行加速
os.environ['ACLNN_BACKEND'] = '1'
os.environ['PYTORCH_NPU_ALLOC_CONF'] = 'expandable_segments:True'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0) # 这里的 ID 会被映射为 npu ID
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cloth_path', type=str, required=True)
    parser.add_argument('--sample', type=int, default=1)
    args = parser.parse_args()

    # 原生指定 NPU 设备
    device = torch.device(f"npu:{args.gpu_id}")
    torch.npu.set_device(device)
    print(f"--- 成功启动原生 NPU 设备: {device} ---")

    # 初始化模型
    openpose_model = OpenPose(args.gpu_id)
    parsing_model = Parsing(args.gpu_id)
    model = OOTDiffusionHD(args.gpu_id)

    # 路径准备
    output_dir = Path('./images_output')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取与处理图片
    cloth_img = Image.open(args.cloth_path).resize((768, 1024))
    model_img = Image.open(args.model_path).resize((768, 1024))
    
    keypoints = openpose_model(model_img.resize((384, 512)))
    model_parse, _ = parsing_model(model_img.resize((384, 512)))

    mask, mask_gray = get_mask_location('hd', 'upper_body', model_parse, keypoints)
    mask = mask.resize((768, 1024), Image.NEAREST)
    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
    
    masked_vton_img = Image.composite(mask_gray, model_img, mask)

    # 推理
    images = model(
        model_type='hd',
        category='upperbody',
        image_garm=cloth_img,
        image_vton=masked_vton_img,
        mask=mask,
        image_ori=model_img,
        num_samples=args.sample,
    )

    for i, img in enumerate(images):
        img.save(output_dir / f'out_{i}.png')
    print(f"任务完成，结果已保存至 {output_dir}")

if __name__ == '__main__':
    main()