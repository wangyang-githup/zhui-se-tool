"""
模块一：图像色彩特征提取器
=====================================
输入：一张图片（float32 RGB [0,1] 或 uint8 BGR）
输出：色彩特征包 ColorFeatures（dataclass）

包含：
- 影调分布（亮度分位数）
- Lab 均值/标准差
- K-Means 5色调色板
- 肤色区域特征（YCrCb 范围检测）
- 环境主色特征（非肤色区域主色）
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from sklearn.cluster import MiniBatchKMeans


# ─────────────────────────────────────────────────────
#  数据结构
# ─────────────────────────────────────────────────────

@dataclass
class LabStats:
    """Lab 通道统计"""
    mean_L: float
    std_L:  float
    mean_a: float
    std_a:  float
    mean_b: float
    std_b:  float

    def to_dict(self):
        return {
            "mean_L": round(self.mean_L, 2),
            "std_L":  round(self.std_L,  2),
            "mean_a": round(self.mean_a, 2),
            "std_a":  round(self.std_a,  2),
            "mean_b": round(self.mean_b, 2),
            "std_b":  round(self.std_b,  2),
        }


@dataclass
class ToneDistribution:
    """影调分布（亮度 L 通道的分位统计）"""
    p5:   float   # 暗部极值
    p25:  float   # 暗部
    p50:  float   # 中间调
    p75:  float   # 亮部
    p95:  float   # 亮部极值
    mean: float
    std:  float
    histogram: List[float] = field(default_factory=list)  # 256分段归一化直方图

    def to_dict(self):
        return {
            "p5": round(self.p5, 2),
            "p25": round(self.p25, 2),
            "p50": round(self.p50, 2),
            "p75": round(self.p75, 2),
            "p95": round(self.p95, 2),
            "mean": round(self.mean, 2),
            "std":  round(self.std, 2),
        }


@dataclass
class ColorPalette:
    """K-Means 主色调色板（RGB 0-255）"""
    colors: List[Tuple[int, int, int]]   # 每个颜色 (R, G, B)
    weights: List[float]                  # 各颜色占比

    def to_dict(self):
        return {
            "colors":  [list(c) for c in self.colors],
            "weights": [round(w, 4) for w in self.weights],
        }


@dataclass
class SkinFeature:
    """肤色区域特征"""
    detected: bool
    mask_ratio: float              # 肤色像素占全图比例
    lab_stats: Optional[LabStats]  # 肤色区域 Lab 统计
    highlight_lab: Optional[Tuple[float, float, float]]   # 高光区肤色 Lab
    shadow_lab: Optional[Tuple[float, float, float]]      # 阴影区肤色 Lab

    def to_dict(self):
        return {
            "detected":    self.detected,
            "mask_ratio":  round(self.mask_ratio, 4),
            "lab_stats":   self.lab_stats.to_dict() if self.lab_stats else None,
            "highlight_lab": list(self.highlight_lab) if self.highlight_lab else None,
            "shadow_lab":  list(self.shadow_lab) if self.shadow_lab else None,
        }


@dataclass
class ColorFeatures:
    """完整色彩特征包"""
    tone: ToneDistribution
    lab:  LabStats
    palette: ColorPalette
    skin: SkinFeature
    env_palette: ColorPalette   # 环境主色（非肤色区域）

    def to_dict(self):
        return {
            "tone":        self.tone.to_dict(),
            "lab":         self.lab.to_dict(),
            "palette":     self.palette.to_dict(),
            "skin":        self.skin.to_dict(),
            "env_palette": self.env_palette.to_dict(),
        }


# ─────────────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────────────

def load_as_rgb_float(path: str) -> np.ndarray:
    """读取图片 → float32 RGB [0,1]"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def rgb_float_to_lab(img: np.ndarray) -> np.ndarray:
    """float32 RGB [0,1] → OpenCV Lab（L∈0-255, a/b∈0-255 offset 128）"""
    u8 = (img * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)


def lab_normalize(lab_cv: np.ndarray) -> np.ndarray:
    """
    OpenCV Lab → 标准 Lab
    L: [0,255] → [0,100]
    a,b: [0,255] → [-128, 127]
    """
    out = lab_cv.copy()
    out[:, :, 0] = out[:, :, 0] / 255.0 * 100.0
    out[:, :, 1] = out[:, :, 1] - 128.0
    out[:, :, 2] = out[:, :, 2] - 128.0
    return out


def detect_skin_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    YCrCb + HSV 双空间肤色检测
    返回：uint8 mask（0=非肤色，255=肤色）
    """
    u8 = (img_rgb * 255).clip(0, 255).astype(np.uint8)

    # YCrCb 检测（通用肤色范围，适合多人种）
    ycrcb = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
    mask_y = cv2.inRange(ycrcb,
                         np.array([0,  133,  77], np.uint8),
                         np.array([255, 173, 127], np.uint8))

    # HSV 辅助（过滤过饱和非皮肤）
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
    mask_h = cv2.inRange(hsv,
                         np.array([0, 15, 50],  np.uint8),
                         np.array([25, 200, 255], np.uint8))

    mask = cv2.bitwise_and(mask_y, mask_h)

    # 形态学平滑
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


def kmeans_palette(pixels: np.ndarray, k: int = 5) -> Tuple[List, List]:
    """
    K-Means 提取 k 个主色
    pixels: (N, 3) float32 RGB [0,1]
    返回：(colors_rgb_255, weights)
    """
    if len(pixels) < k:
        k = max(1, len(pixels))
    km = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42)
    labels = km.fit_predict(pixels)
    centers = (km.cluster_centers_ * 255).clip(0, 255).astype(int)
    counts = np.bincount(labels, minlength=k)
    weights = counts / counts.sum()
    # 按权重降序排列
    order = np.argsort(-weights)
    colors_out  = [tuple(centers[i]) for i in order]
    weights_sorted = [float(weights[i]) for i in order]
    return colors_out, weights_sorted


def lab_stats_from_pixels(lab_pixels: np.ndarray) -> LabStats:
    """(N, 3) Lab 像素 → LabStats"""
    return LabStats(
        mean_L=float(lab_pixels[:, 0].mean()),
        std_L= float(lab_pixels[:, 0].std()),
        mean_a=float(lab_pixels[:, 1].mean()),
        std_a= float(lab_pixels[:, 1].std()),
        mean_b=float(lab_pixels[:, 2].mean()),
        std_b= float(lab_pixels[:, 2].std()),
    )


# ─────────────────────────────────────────────────────
#  主提取器
# ─────────────────────────────────────────────────────

class FeatureExtractor:
    """
    色彩特征提取器
    用法：
        extractor = FeatureExtractor()
        features = extractor.extract(img_rgb_float)
    """

    def extract(self, img: np.ndarray, max_pixels: int = 50000) -> ColorFeatures:
        """
        提取完整色彩特征包

        img: float32 RGB [0,1]，shape (H, W, 3)
        max_pixels: 下采样到多少像素做统计（加速）
        """
        H, W = img.shape[:2]

        # ── 下采样（加速统计）──
        flat = img.reshape(-1, 3)
        if len(flat) > max_pixels:
            idx = np.random.choice(len(flat), max_pixels, replace=False)
            flat_sample = flat[idx]
        else:
            flat_sample = flat

        # ── Lab 统计 ──
        lab_img = rgb_float_to_lab(img)
        lab_norm = lab_normalize(lab_img)  # L:0-100, a/b:-128~127
        lab_flat = lab_norm.reshape(-1, 3)
        if len(lab_flat) > max_pixels:
            lab_sample = lab_flat[idx]
        else:
            lab_sample = lab_flat

        lab_stats = lab_stats_from_pixels(lab_sample)

        # ── 影调分布（L 通道）──
        L_vals = lab_sample[:, 0]
        tone = ToneDistribution(
            p5=float(np.percentile(L_vals, 5)),
            p25=float(np.percentile(L_vals, 25)),
            p50=float(np.percentile(L_vals, 50)),
            p75=float(np.percentile(L_vals, 75)),
            p95=float(np.percentile(L_vals, 95)),
            mean=float(L_vals.mean()),
            std=float(L_vals.std()),
        )

        # ── 全图主色调色板 ──
        colors, weights = kmeans_palette(flat_sample, k=5)
        palette = ColorPalette(colors=colors, weights=weights)

        # ── 肤色区域特征 ──
        skin_mask_u8 = detect_skin_mask(img)
        skin_ratio = float((skin_mask_u8 > 0).sum()) / (H * W)
        skin_detected = skin_ratio > 0.02   # 超过 2% 才认为有人像

        skin_feature = SkinFeature(
            detected=skin_detected,
            mask_ratio=skin_ratio,
            lab_stats=None,
            highlight_lab=None,
            shadow_lab=None,
        )

        env_colors, env_weights = colors, weights   # 默认

        if skin_detected:
            skin_bool = skin_mask_u8 > 0
            env_bool  = ~skin_bool

            # 肤色区 Lab 统计
            skin_lab_pixels = lab_norm[skin_bool]
            skin_feature.lab_stats = lab_stats_from_pixels(skin_lab_pixels)

            # 高光肤色（亮度 top 25%）
            skin_L = skin_lab_pixels[:, 0]
            hi_thr = np.percentile(skin_L, 75)
            lo_thr = np.percentile(skin_L, 25)
            hi_pixels = skin_lab_pixels[skin_L >= hi_thr]
            lo_pixels = skin_lab_pixels[skin_L <= lo_thr]
            if len(hi_pixels) > 0:
                skin_feature.highlight_lab = (
                    float(hi_pixels[:, 0].mean()),
                    float(hi_pixels[:, 1].mean()),
                    float(hi_pixels[:, 2].mean()),
                )
            if len(lo_pixels) > 0:
                skin_feature.shadow_lab = (
                    float(lo_pixels[:, 0].mean()),
                    float(lo_pixels[:, 1].mean()),
                    float(lo_pixels[:, 2].mean()),
                )

            # 环境主色（非肤色区域）
            k_env = 5
            if env_bool.sum() > k_env:
                env_pixels = flat[env_bool.flatten()]
                if len(env_pixels) > max_pixels:
                    env_idx = np.random.choice(len(env_pixels), max_pixels, replace=False)
                    env_pixels = env_pixels[env_idx]
                env_colors, env_weights = kmeans_palette(env_pixels, k=k_env)

        env_palette = ColorPalette(colors=env_colors, weights=env_weights)

        return ColorFeatures(
            tone=tone,
            lab=lab_stats,
            palette=palette,
            skin=skin_feature,
            env_palette=env_palette,
        )
