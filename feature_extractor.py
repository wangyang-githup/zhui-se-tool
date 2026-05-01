"""
追色工具 - 图像色彩特征提取器
=====================================
输入：一张图片（float32 RGB [0,1] 或 uint8）
输出：色彩特征包 ColorFeatures（dataclass）

包含：
- 影调分布（亮度分位数）
- Lab 均值/标准差
- K-Means 5色调色板
- 肤色区域特征（YCrCb 范围检测）
- 环境主色特征（非肤色区域）

无 cv2 依赖，纯 PIL + NumPy 实现
"""

from PIL import Image
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from sklearn.cluster import MiniBatchKMeans


# ─────────────────────────────────────────────────────
#  ICC 标准色彩转换矩阵（sRGB ↔ XYZ ↔ Lab）
# ─────────────────────────────────────────────────────

_SRGB_TO_XYZ = np.array([
    [0.4123908,  0.35758434, 0.18048079],
    [0.21263901, 0.71516868, 0.07219232],
    [0.01933082, 0.11919478, 0.95053215],
], dtype=np.float32)

_XYZ_TO_SRGB = np.array([
    [ 3.24096994, -1.53738318, -0.49861076],
    [-0.96924364,  1.8759675,  0.04155506],
    [ 0.05563008, -0.20397696,  1.05697151],
], dtype=np.float32)

_XYZ_W = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)


def _gamma_correct(c: np.ndarray) -> np.ndarray:
    """线性化 sRGB gamma 曲线"""
    out = np.where(c > 0.0031308,
                   1.055 * np.power(np.clip(c, 1e-8), 1.0 / 2.4) - 0.055,
                   12.92 * c)
    return np.clip(out, 0.0, 1.0)


def _inv_gamma(c: np.ndarray) -> np.ndarray:
    """逆 gamma（线性化）"""
    out = np.where(c > 0.04045,
                   np.power((c + 0.055) / 1.055, 2.4),
                   c / 12.92)
    return np.clip(out, 0.0, 1.0)


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """XYZ → Lab（L*a*b*，D65 白点）"""
    xyz_w = _XYZ_W
    xyz_n = xyz / xyz_w

    eps = 1e-6
    f = np.where(xyz_n > eps**3,
                 np.power(xyz_n, 1.0 / 3.0),
                 (841.0 / 108.0) * xyz_n + 4.0 / 29.0)

    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])

    lab = np.stack([L, a, b], axis=-1)
    return lab.astype(np.float32)


def _lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    """Lab → XYZ"""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    eps = 6.0 / 29.0
    xyz_n = np.where(
        fy > eps,
        fy ** 3.0,
        (fy - 16.0 / 116.0) * (108.0 / 841.0)
    )
    xr = np.where(fx > eps, fx ** 3.0, (fx - 4.0 / 29.0) * (108.0 / 841.0))
    zr = np.where(fz > eps, fz ** 3.0, (fz - 4.0 / 29.0) * (108.0 / 841.0))

    xyz = np.stack([xr, xyz_n, zr], axis=-1) * _XYZ_W
    return xyz.astype(np.float32)


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
    p5:   float
    p25:  float
    p50:  float
    p75:  float
    p95:  float
    mean: float
    std:  float
    histogram: List[float] = field(default_factory=list)

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
    colors: List[Tuple[int, int, int]]
    weights: List[float]

    def to_dict(self):
        return {
            "colors":  [list(c) for c in self.colors],
            "weights": [round(w, 4) for w in self.weights],
        }


@dataclass
class SkinFeature:
    """肤色区域特征"""
    detected: bool
    mask_ratio: float
    lab_stats: Optional[LabStats]
    highlight_lab: Optional[Tuple[float, float, float]]
    shadow_lab: Optional[Tuple[float, float, float]]

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
    env_palette: ColorPalette

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
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def rgb_float_to_lab(img: np.ndarray) -> np.ndarray:
    """float32 RGB [0,1] → 标准 Lab（L:0-100, a/b:-128~127）"""
    rgb_lin = _inv_gamma(img)
    xyz = np.tensordot(rgb_lin, _SRGB_TO_XYZ.T, axes=[[2], [0]])
    return _xyz_to_lab(xyz)


def detect_skin_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    YCrCb + HSV 双空间肤色检测
    返回：uint8 mask（0=非肤色，255=肤色）
    """
    u8 = np.clip(img_rgb * 255, 0, 255).astype(np.uint8)
    r, g, b = u8[..., 0], u8[..., 1], u8[..., 2]

    # YCrCb 检测
    Y  =  0.299 * r + 0.587 * g + 0.114 * b
    Cr = (r - Y) * 0.713 + 128
    Cb = (b - Y) * 0.564 + 128

    mask_y = ((Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127)).astype(np.uint8) * 255

    # HSV 检测
    v = np.maximum(np.maximum(r / 255.0, g / 255.0), b / 255.0)
    c = v - np.minimum(np.minimum(r / 255.0, g / 255.0), b / 255.0)
    s = np.where(v != 0, c / v, 0.0)

    mask_s = ((s * 255 >= 15) & (s * 255 <= 200) & (v * 255 >= 40)).astype(np.uint8) * 255

    # 合并
    mask = np.bitwise_and(mask_y, mask_s)

    return mask


def kmeans_palette(pixels: np.ndarray, k: int = 5) -> Tuple[List, List]:
    """K-Means 提取 k 个主色"""
    if len(pixels) < k:
        k = max(1, len(pixels))
    km = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42)
    labels = km.fit_predict(pixels)
    centers = (km.cluster_centers_ * 255).clip(0, 255).astype(int)
    counts = np.bincount(labels, minlength=k)
    weights = counts / counts.sum()
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
        """提取完整色彩特征包"""
        H, W = img.shape[:2]

        # 下采样
        flat = img.reshape(-1, 3)
        if len(flat) > max_pixels:
            idx = np.random.choice(len(flat), max_pixels, replace=False)
            flat_sample = flat[idx]
        else:
            flat_sample = flat

        # Lab 统计
        lab_norm = rgb_float_to_lab(img)
        lab_flat = lab_norm.reshape(-1, 3)
        if len(lab_flat) > max_pixels:
            lab_sample = lab_flat[idx]
        else:
            lab_sample = lab_flat

        lab_stats = lab_stats_from_pixels(lab_sample)

        # 影调分布（L 通道）
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

        # 全图主色调色板
        colors, weights = kmeans_palette(flat_sample, k=5)
        palette = ColorPalette(colors=colors, weights=weights)

        # 肤色区域特征
        skin_mask_u8 = detect_skin_mask(img)
        skin_ratio = float((skin_mask_u8 > 0).sum()) / (H * W)
        skin_detected = skin_ratio > 0.02

        skin_feature = SkinFeature(
            detected=skin_detected,
            mask_ratio=skin_ratio,
            lab_stats=None,
            highlight_lab=None,
            shadow_lab=None,
        )

        env_colors, env_weights = colors, weights

        if skin_detected:
            skin_bool = skin_mask_u8 > 0

            # 肤色区 Lab 统计
            skin_lab_pixels = lab_norm[skin_bool]
            skin_feature.lab_stats = lab_stats_from_pixels(skin_lab_pixels)

            # 高光/阴影肤色
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
            if skin_bool.sum() > k_env:
                env_pixels = flat[~skin_bool.flatten()]
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
