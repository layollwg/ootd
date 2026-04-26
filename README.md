# OOTDiffusion — Virtual Try-On on Ascend NPU

This repository adapts [OOTDiffusion](https://github.com/levihsu/OOTDiffusion) for Huawei Ascend NPU (torch_npu), enabling high-quality virtual garment fitting without requiring CUDA GPUs.

## Models

| Class | File | Checkpoint |
|---|---|---|
| `OOTDiffusionHD` | `ootd/inference_ootd_hd.py` | `ootd_hd/checkpoint-36000` |
| `OOTDiffusionDC` | `ootd/inference_ootd_dc.py` | `ootd_dc/checkpoint-36000` |

- **HD** (`model_type='hd'`): half-body model, best for upper-body garments at high resolution.
- **DC** (`model_type='dc'`): full-body / dress-code model, supports `upperbody`, `lowerbody`, and `dress` categories.

## Requirements

- Python ≥ 3.8
- PyTorch + torch_npu (Ascend NPU driver and CANN toolkit must be installed)
- diffusers, transformers, accelerate
- OpenCV, Pillow

Install dependencies:

```bash
pip install torch diffusers transformers accelerate opencv-python pillow
# Install torch_npu separately according to your CANN version
```

## Checkpoints

Download the pretrained checkpoints and place them as follows:

```
checkpoints/
├── clip-vit-large-patch14/
└── ootd/
    ├── vae/
    ├── tokenizer/
    ├── text_encoder/
    ├── ootd_hd/
    │   └── checkpoint-36000/
    │       ├── unet_garm/
    │       └── unet_vton/
    └── ootd_dc/
        └── checkpoint-36000/
            ├── unet_garm/
            └── unet_vton/
```

## Usage

```python
from PIL import Image
from ootd.inference_ootd_hd import OOTDiffusionHD

# gpu_id: index of your Ascend NPU device (e.g. 0)
model = OOTDiffusionHD(gpu_id=0)

image_garm = Image.open("garment.jpg")    # garment image
image_vton = Image.open("model_mask.jpg") # masked model image
mask       = Image.open("mask.jpg")       # binary mask
image_ori  = Image.open("model.jpg")      # original model image

result_images = model(
    model_type='hd',
    category='upperbody',
    image_garm=image_garm,
    image_vton=image_vton,
    mask=mask,
    image_ori=image_ori,
    num_samples=1,
    num_steps=20,
    image_scale=2.0,
    seed=-1,
)
result_images[0].save("result.jpg")
```

For the DC (dress-code) model, replace `OOTDiffusionHD` with `OOTDiffusionDC` and set `model_type='dc'`.

## NPU Adaptation Notes

- `torch_npu` is imported with a graceful fallback so the code can still be imported on non-NPU machines.
- Devices are addressed as `npu:<id>` instead of `cuda:<id>`.
- `torch.npu.empty_cache()` is called before each inference pass to free NPU memory.
- All tensors (tokens, image features) are explicitly moved to the NPU device before computation.

## License

This project is based on OOTDiffusion and inherits its original license. See [LICENSE](LICENSE) for details.
