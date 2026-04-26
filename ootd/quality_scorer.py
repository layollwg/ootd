"""
quality_scorer.py — CLIP-based image quality scoring for virtual try-on outputs.

Scoring dimensions (two independent signals, weighted and combined):
  1. Garment alignment (garment_alignment): CLIP cosine similarity between the
     generated result image and the original garment image.
     Higher score means the garment details are better preserved.
  2. Sharpness: Laplacian variance of the image (computed on CPU, no GPU/NPU needed).
     Higher score means the image is sharper and less blurry.

Both scores are min-max normalized to [0, 1] and combined as a weighted average,
controlled by `alpha` (garment alignment weight) and `1 - alpha` (sharpness weight).

Typical usage:
    scorer = QualityScorer(image_encoder, auto_processor, device)
    best_images = scorer.select_best(
        candidates,          # List[PIL.Image.Image]
        image_garm,          # PIL.Image.Image — reference garment image
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
    Scores generated try-on images using the already-loaded CLIP image encoder
    and selects the best top_k candidates from a list.

    Parameters
    ----------
    image_encoder : CLIPVisionModelWithProjection
        CLIP vision encoder already moved to the target device (shared with inference,
        no duplicate loading needed).
    auto_processor : AutoProcessor
        Corresponding CLIP image preprocessor.
    device : torch.device
        Inference device (npu:0 / cuda:0 / cpu).
    """

    def __init__(self, image_encoder, auto_processor, device):
        self.image_encoder = image_encoder
        self.auto_processor = auto_processor
        self.device = device

    # ------------------------------------------------------------------
    # Dimension 1: garment alignment (CLIP cosine similarity)
    # ------------------------------------------------------------------
    def _clip_embed(self, pil_image: Image.Image) -> torch.Tensor:
        """Return the L2-normalized CLIP embedding for a single image (shape: [D])."""
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
        Compute the cosine similarity between each candidate image and the garment image.
        Returns a float32 numpy array of shape [N] with values in [-1, 1].
        """
        garm_embed = self._clip_embed(image_garm)  # [D]
        scores = []
        for img in candidates:
            img_embed = self._clip_embed(img)  # [D]
            sim = torch.dot(garm_embed, img_embed).item()
            scores.append(sim)
        return np.array(scores, dtype=np.float32)

    # ------------------------------------------------------------------
    # Dimension 2: sharpness (Laplacian variance, CPU)
    # ------------------------------------------------------------------
    @staticmethod
    def _sharpness_score(pil_image: Image.Image) -> float:
        """
        Measure image sharpness using Laplacian variance.
        Higher values indicate crisper edges (not blurry).
        """
        import cv2
        img_np = np.array(pil_image.convert("L"), dtype=np.float32)
        laplacian = cv2.Laplacian(img_np, cv2.CV_32F)
        return float(laplacian.var())

    def _sharpness_scores(self, candidates: list) -> np.ndarray:
        scores = [self._sharpness_score(img) for img in candidates]
        return np.array(scores, dtype=np.float32)

    # ------------------------------------------------------------------
    # Combined scoring & selection
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1]; return all-ones if all values are equal."""
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
        Compute a combined quality score for each candidate image.

        Parameters
        ----------
        candidates : List[PIL.Image.Image]
            Generated images to evaluate.
        image_garm : PIL.Image.Image
            Reference garment image used to compute the alignment score.
        alpha : float
            Weight for the garment alignment score (0–1);
            sharpness weight = 1 - alpha.

        Returns
        -------
        scores : np.ndarray, shape [N]
            Combined quality score per image (range [0, 1], higher is better).
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
        Select the top_k highest-scoring images from the candidate list, ranked
        from best to worst.

        Parameters
        ----------
        candidates : List[PIL.Image.Image]
            Images to filter, typically produced by multiple inference passes or
            num_images_per_prompt > 1.
        image_garm : PIL.Image.Image
            Reference garment image.
        top_k : int
            Number of best images to return (default 1).
        alpha : float
            Garment alignment score weight (default 0.7).

        Returns
        -------
        List[PIL.Image.Image]
            The top_k images ordered from highest to lowest score.
        """
        if len(candidates) <= top_k:
            return candidates

        scores = self.score(candidates, image_garm, alpha=alpha)
        ranked_indices = np.argsort(scores)[::-1]  # descending

        print("--- QualityScorer scores ---")
        for rank, idx in enumerate(ranked_indices):
            print(f"  Rank {rank + 1}: image {idx}  score {scores[idx]:.4f}")

        top_indices = ranked_indices[:top_k]
        return [candidates[i] for i in top_indices]

