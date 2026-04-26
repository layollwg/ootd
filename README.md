# OOTDiffusion — Virtual Try-On on Ascend NPU

This repository adapts [OOTDiffusion](https://github.com/levihsu/OOTDiffusion) for Huawei Ascend NPU (torch_npu), with an additional **quality-filtering** layer that generates multiple candidate images and automatically selects the best ones using CLIP-based scoring.

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

```bash
pip install torch diffusers transformers accelerate opencv-python pillow
# Install torch_npu separately according to your CANN version
```

## Checkpoints

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

### Basic (single output)

```python
from PIL import Image
from ootd.inference_ootd_hd import OOTDiffusionHD

model = OOTDiffusionHD(gpu_id=0)

result_images = model(
    model_type='hd',
    category='upperbody',
    image_garm=Image.open("garment.jpg"),
    image_vton=Image.open("model_mask.jpg"),
    mask=Image.open("mask.jpg"),
    image_ori=Image.open("model.jpg"),
    num_steps=20,
    image_scale=2.0,
    seed=-1,
)
result_images[0].save("result.jpg")
```

### Best-of-N quality filtering (recommended for higher quality)

Pass `num_candidates` to generate N independent batches with different seeds, then automatically return the `top_k` highest-scored images:

```python
result_images = model(
    model_type='hd',
    category='upperbody',
    image_garm=Image.open("garment.jpg"),
    image_vton=Image.open("model_mask.jpg"),
    mask=Image.open("mask.jpg"),
    image_ori=Image.open("model.jpg"),
    num_steps=20,
    image_scale=2.0,
    seed=42,
    num_candidates=4,   # generate 4 candidates
    top_k=1,            # return the best 1
    score_alpha=0.7,    # 70% weight on garment alignment, 30% on sharpness
)
result_images[0].save("best_result.jpg")
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `num_steps` | 20 | Denoising steps. More steps = higher quality (try 30–50) |
| `image_scale` | 1.0 | Image guidance scale. Try 1.5–2.5 for stronger garment adherence |
| `num_candidates` | 1 | Number of independent batches to generate (best-of-N) |
| `top_k` | 1 | How many of the best candidates to return |
| `score_alpha` | 0.7 | Weight for garment-alignment score vs. sharpness score |

## How Quality Scoring Works (`quality_scorer.py`)

Each candidate image is scored on two axes using the CLIP encoder already loaded in the model — **no extra models or memory** are needed:

1. **Garment alignment** (weight `alpha`): CLIP cosine similarity between the generated result and the original garment image. A high score means the garment texture and pattern were faithfully transferred.
2. **Sharpness** (weight `1 - alpha`): Laplacian variance of the image (CPU-only). A high score means the image is crisp, not blurry.

Both scores are min-max normalized to [0, 1] and combined into a weighted average. Candidates are ranked and the top-k returned.

## Tips for Better Quality

| Technique | How |
|---|---|
| More denoising steps | `num_steps=30` or `num_steps=50` |
| Stronger garment guidance | `image_scale=2.0` |
| Best-of-N selection | `num_candidates=4, top_k=1` |
| Balance alignment vs. sharpness | `score_alpha=0.8` to favour garment fidelity |

## NPU Adaptation Notes

- `torch_npu` is imported with a graceful fallback so the code can still be used on non-NPU machines.
- Devices are addressed as `npu:<id>` instead of `cuda:<id>`.
- `torch.npu.empty_cache()` is called before each inference pass to free NPU memory.
- All tensors (tokens, image features) are explicitly moved to the NPU device before computation.

## License

This project is based on OOTDiffusion and inherits its original license. See [LICENSE](LICENSE) for details.
