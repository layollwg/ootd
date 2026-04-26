import os
import sys
import torch
import datetime
from pathlib import Path
from PIL import Image

# ================= 路径配置 =================
CHAR_DIR = "../../TaskB/character"
CLOTH_DIR = "../../TaskB/clothes"
FINAL_DIR = "./commercial_results"
OUTPUT_DIR = "./images_output" 

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "ootd"))

try:
    from preprocess.openpose.run_openpose import OpenPose
    from preprocess.humanparsing.run_parsing import Parsing
    from ootd.inference_ootd_hd import OOTDiffusionHD
    from utils_ootd import get_mask_location
except ImportError as e:
    print(f"!!! 模块加载失败: {e}")
    sys.exit(1)
# ==========================================

os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ['ACLNN_BACKEND'] = '1' 

def create_pure_row(model_path, cloth_path, res_img):
    """极简拼图：只拼接三张图，无文字，无复杂边框"""
    W, H = 768, 1024
    gap = 10  # 图片之间只留 10 像素的白边区分
    
    canvas = Image.new('RGB', (W * 3 + gap * 2, H), (255, 255, 255))
    
    img_model = Image.open(model_path).convert('RGB').resize((W, H))
    img_cloth = Image.open(cloth_path).convert('RGB').resize((W, H))
    img_res = res_img.resize((W, H))
    
    canvas.paste(img_model, (0, 0))
    canvas.paste(img_cloth, (W + gap, 0))
    canvas.paste(img_res, ((W + gap) * 2, 0))
    
    return canvas

def main():
    print("--- 正在加载大模型进入 NPU 显存 ---")
    gpu_id = 0
    
    openpose_model = OpenPose(gpu_id)
    parsing_model = Parsing(gpu_id)
    ootd_model = OOTDiffusionHD(gpu_id)
        
    print("--- 模型加载完毕！极速流水线启动 ---")

    char_files = sorted(list(Path(CHAR_DIR).glob("*.jpg")), key=lambda x: x.name)
    all_rows = []

    for char_path in char_files:
        name = char_path.stem
        for cloth_suffix in ["", "0"]:
            cloth_name = f"{cloth_suffix}{name}.jpg"
            cloth_path = os.path.join(CLOTH_DIR, cloth_name)
            
            if os.path.exists(cloth_path):
                print(f"[推理中] 模特:{name} | 衣服:{cloth_name}")
                
                try:
                    # 1. 预处理
                    cloth_img = Image.open(cloth_path).convert('RGB').resize((768, 1024))
                    model_img = Image.open(char_path).convert('RGB').resize((768, 1024))
                    model_img_small = model_img.resize((384, 512))
                    
                    keypoints = openpose_model(model_img_small)
                    model_parse, _ = parsing_model(model_img_small)
                    
                    mask, mask_gray = get_mask_location('hd', 'upper_body', model_parse, keypoints)
                    mask = mask.resize((768, 1024), Image.NEAREST)
                    mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
                    masked_vton_img = Image.composite(mask_gray, model_img, mask)

                    # 2. Diffusion 推理 (保留 35步 和 2.0引导权重以确保高画质)
                    images = ootd_model(
                        model_type='hd',
                        category='upperbody',
                        image_garm=cloth_img,
                        image_vton=masked_vton_img,
                        mask=mask,
                        image_ori=model_img,
                        num_samples=1,
                        num_steps=35,
                        image_scale=2.0 
                    )
                    
                    result_img = images[0]
                    torch.npu.empty_cache()

                    # 3. 极简拼图
                    row_canvas = create_pure_row(char_path, cloth_path, result_img)
                    all_rows.append(row_canvas)

                except Exception as e:
                    print(f"处理报错: {e}")

    # 4. 纵向拼接所有结果，生成最终大图
    if all_rows:
        print("\n--- 正在合成最终长图 ---")
        total_w = all_rows[0].width
        total_h = sum(r.height for r in all_rows) + 10 * (len(all_rows) - 1)
        
        summary_canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
        
        curr_y = 0
        for r in all_rows:
            summary_canvas.paste(r, (0, curr_y))
            curr_y += r.height + 10
            
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        final_path = os.path.join(FINAL_DIR, f"Pure_Batch_Result_{timestamp}.jpg")
        summary_canvas.save(final_path, quality=90)
        print(f"--- 极简高质长图已生成: {final_path} ---")

if __name__ == "__main__":
    main()