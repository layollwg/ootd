"""
quality_scorer.py — 基于 CLIP 的虚拟试衣图片质量打分模块

打分维度（两个独立信号，加权融合）：
  1. 服装对齐分 (garment_alignment): 生成图与原始服装图之间的 CLIP 余弦相似度。
     分数越高说明服装细节保留越好。
  2. 清晰度分 (sharpness): 图像 Laplacian 方差（CPU 计算，无需 GPU/NPU）。
     分数越高说明图像越清晰，不模糊。

两个分数分别归一化到 [0, 1] 后加权平均，权重可通过
  alpha (garment_alignment 权重) 和 (1 - alpha) (sharpness 权重) 控制。

典型用法：
    scorer = QualityScorer(image_encoder, auto_processor, device)
    best_images = scorer.select_best(
        candidates,          # List[PIL.Image.Image]
        image_garm,          # PIL.Image.Image，原始服装图
        top_k=1,
        alpha=0.7,
    )
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class QualityScorer:
    """
    利用已加载的 CLIP image encoder 对生成图片进行质量打分，
    从多个候选图中选出最优的 top_k 张。

    Parameters
    ----------
    image_encoder : CLIPVisionModelWithProjection
        已移至目标设备的 CLIP 视觉编码器（与 inference 共享，不重复加载）。
    auto_processor : AutoProcessor
        对应的 CLIP 预处理器。
    device : torch.device
        推理设备（npu:0 / cuda:0 / cpu）。
    """

    def __init__(self, image_encoder, auto_processor, device):
        self.image_encoder = image_encoder
        self.auto_processor = auto_processor
        self.device = device

    # ------------------------------------------------------------------
    # 维度 1: 服装对齐分 (CLIP 余弦相似度)
    # ------------------------------------------------------------------
    def _clip_embed(self, pil_image: Image.Image) -> torch.Tensor:
        """返回单张图片的 L2 归一化 CLIP 嵌入向量 (shape: [D])."""
        inputs = self.auto_processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeds = self.image_encoder(inputs.data["pixel_values"]).image_embeds  # [1, D]
        embeds = F.normalize(embeds, dim=-1)
        return embeds.squeeze(0)  # [D]

    def _garment_alignment_scores(
        self,
        candidates: list,
        image_garm: Image.Image,
    ) -> np.ndarray:
        """
        计算每张候选图与服装图的余弦相似度。
        返回 shape [N] 的 float32 numpy 数组，值域 [-1, 1]。
        """
        garm_embed = self._clip_embed(image_garm)  # [D]
        scores = []
        for img in candidates:
            img_embed = self._clip_embed(img)  # [D]
            sim = torch.dot(garm_embed, img_embed).item()
            scores.append(sim)
        return np.array(scores, dtype=np.float32)

    # ------------------------------------------------------------------
    # 维度 2: 清晰度分 (Laplacian 方差，CPU)
    # ------------------------------------------------------------------
    @staticmethod
    def _sharpness_score(pil_image: Image.Image) -> float:
        """
        用 Laplacian 方差度量图像清晰度。
        值越大说明边缘越清晰（不模糊）。
        """
        import cv2
        img_np = np.array(pil_image.convert("L"), dtype=np.float32)
        laplacian = cv2.Laplacian(img_np, cv2.CV_32F)
        return float(laplacian.var())

    def _sharpness_scores(self, candidates: list) -> np.ndarray:
        scores = [self._sharpness_score(img) for img in candidates]
        return np.array(scores, dtype=np.float32)

    # ------------------------------------------------------------------
    # 融合打分 & 选择
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Min-max 归一化到 [0, 1]，若所有值相同则返回全 1 数组。"""
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-8:
            return np.ones_like(scores)
        return (scores - lo) / (hi - lo)

    def score(
        self,
        candidates: list,
        image_garm: Image.Image,
        alpha: float = 0.7,
    ) -> np.ndarray:
        """
        对候选图片列表进行综合打分。

        Parameters
        ----------
        candidates : List[PIL.Image.Image]
            待评分的生成图片列表。
        image_garm : PIL.Image.Image
            原始服装参考图，用于计算服装对齐分。
        alpha : float
            服装对齐分权重 (0~1)；清晰度分权重 = 1 - alpha。

        Returns
        -------
        scores : np.ndarray, shape [N]
            每张图的综合得分（值域 [0, 1]，越高越好）。
        """
        align_scores = self._normalize(self._garment_alignment_scores(candidates, image_garm))
        sharp_scores = self._normalize(self._sharpness_scores(candidates))
        return alpha * align_scores + (1.0 - alpha) * sharp_scores

    def select_best(
        self,
        candidates: list,
        image_garm: Image.Image,
        top_k: int = 1,
        alpha: float = 0.7,
    ) -> list:
        """
        从候选图列表中选出得分最高的 top_k 张，按得分从高到低排序返回。

        Parameters
        ----------
        candidates : List[PIL.Image.Image]
            待筛选的生成图片，通常由多次推理或 num_images_per_prompt > 1 产生。
        image_garm : PIL.Image.Image
            原始服装参考图。
        top_k : int
            返回最优图片的数量，默认 1（仅返回最佳图）。
        alpha : float
            服装对齐分权重，默认 0.7。

        Returns
        -------
        List[PIL.Image.Image]
            得分最高的 top_k 张图片，按得分降序排列。
        """
        if len(candidates) <= top_k:
            return candidates

        scores = self.score(candidates, image_garm, alpha=alpha)
        ranked_indices = np.argsort(scores)[::-1]  # 降序

        print("--- QualityScorer 得分 ---")
        for rank, idx in enumerate(ranked_indices):
            print(f"  排名 {rank + 1}: 图片 {idx}  得分 {scores[idx]:.4f}")

        top_indices = ranked_indices[:top_k]
        return [candidates[i] for i in top_indices]
