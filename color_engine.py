"""
color_engine.py — 追色统一引擎 (v2.0)
==========================================
整合原桌面版 + Web版5模块专家管线，彻底去除 OpenCV 依赖

模块清单:
  1. 色彩空间转换 (纯 NumPy: RGB↔Lab, RGB↔HSV)
  2. 图像 I/O (PIL)
  3. 数据结构 (特征包 + 参数包)
  4. 肤色检测 (HSV + Lab 双空间)
  5. 特征提取器
  6. 追色方案生成器
  7. 渲染器 (含肤色保护 ΔE 监控)
  8. 简化追色引擎 (Reinhard)
  9. LUT 生成器
  10. 评价指标
  11. 风格预设
"""

import os
import math
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_closing, binary_opening

# sklearn 可选依赖（K-Means 色板提取）
try:
    from sklearn.cluster import MiniBatchKMeans
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ══════════════════════════════════════════════════════════
#  1. 色彩空间转换 (纯 NumPy)
# ══════════════════════════════════════════════════════════

# sRGB → XYZ 矩阵 (D65 白点)
_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)

# D65 白点
_D65_XYZ = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

# CIE Lab 常数
_LAB_EPSILON = 216.0 / 24389.0   # (6/29)^3
_LAB_KAPPA   = 24389.0 / 27.0    # (29/3)^3


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB gamma 解压缩 → 线性 RGB [0,1]"""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """线性 RGB [0,1] → sRGB gamma 压缩"""
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (np.clip(c, 1e-10, None) ** (1.0 / 2.4)) - 0.055)


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """
    float32 RGB [0,1] → CIE Lab
    L ∈ [0, 100], a ∈ [-128, 127], b ∈ [-128, 127]
    基于 sRGB D65 → XYZ → Lab 标准转换
    """
    linear = _srgb_to_linear(img.astype(np.float64))
    # (H,W,3) → (H*W,3) 矩阵乘法
    shape = linear.shape
    flat = linear.reshape(-1, 3)
    xyz = flat @ _M_RGB2XYZ.T   # (N, 3)
    xyz = xyz.reshape(shape)

    # XYZ → Lab
    xyz_n = xyz / _D65_XYZ

    def f(t):
        return np.where(t > _LAB_EPSILON, np.cbrt(t), (_LAB_KAPPA * t + 16.0) / 116.0)

    fx = f(xyz_n[:, :, 0])
    fy = f(xyz_n[:, :, 1])
    fz = f(xyz_n[:, :, 2])

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return np.stack([L, a, b], axis=-1).astype(np.float32)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    CIE Lab → float32 RGB [0,1]
    """
    L, a_ch, b_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    fy = (L + 16.0) / 116.0
    fx = a_ch / 500.0 + fy
    fz = fy - b_ch / 200.0

    def f_inv(t):
        t3 = t ** 3
        return np.where(t3 > _LAB_EPSILON, t3, (116.0 * t - 16.0) / _LAB_KAPPA)

    x = f_inv(fx) * _D65_XYZ[0]
    y = f_inv(fy) * _D65_XYZ[1]
    z = f_inv(fz) * _D65_XYZ[2]

    xyz = np.stack([x, y, z], axis=-1)
    shape = xyz.shape
    flat = xyz.reshape(-1, 3)
    linear = flat @ _M_XYZ2RGB.T
    linear = linear.reshape(shape).clip(0, 1)

    srgb = _linear_to_srgb(linear)
    return srgb.clip(0, 1).astype(np.float32)


def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """
    float32 RGB [0,1] → HSV
    H ∈ [0, 360), S ∈ [0, 1], V ∈ [0, 1]
    """
    r, g, b = img[:, :, 0].astype(np.float64), img[:, :, 1].astype(np.float64), img[:, :, 2].astype(np.float64)
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    h = np.zeros_like(r)
    mask_r = (cmax == r) & (delta > 1e-10)
    mask_g = (cmax == g) & (delta > 1e-10) & ~mask_r
    mask_b = (cmax == b) & (delta > 1e-10) & ~mask_r & ~mask_g

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
    h = h % 360.0

    # Saturation
    with np.errstate(invalid='ignore', divide='ignore'):
        s = np.where(cmax > 1e-10, delta / cmax, 0.0)

    # Value
    v = cmax

    return np.stack([h, s, v], axis=-1).astype(np.float32)


# ══════════════════════════════════════════════════════════
#  2. 图像 I/O (PIL)
# ══════════════════════════════════════════════════════════

def load_image(path: str) -> np.ndarray:
    """读取图片，转为 float32 RGB [0,1]"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        raise ValueError(f"无法读取图片: {path}\n{e}")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def save_image(img: np.ndarray, path: str, quality: int = 95):
    """保存 float32 RGB [0,1] 图片"""
    u8 = (img.clip(0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(u8, mode="RGB")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        pil_img.save(path, quality=quality, subsampling=0)
    elif ext == ".webp":
        pil_img.save(path, quality=quality)
    else:
        pil_img.save(path)


# ══════════════════════════════════════════════════════════
#  3. 数据结构
# ══════════════════════════════════════════════════════════

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


# ── 特征包 ──

@dataclass
class LabStats:
    """Lab 通道统计"""
    mean_L: float; std_L: float
    mean_a: float; std_a: float
    mean_b: float; std_b: float

    def to_dict(self):
        return {k: round(v, 2) for k, v in self.__dict__.items()}


@dataclass
class ToneDistribution:
    """影调分布（L 通道分位数）"""
    p5: float; p25: float; p50: float; p75: float; p95: float
    mean: float; std: float

    def to_dict(self):
        return {k: round(v, 2) for k, v in self.__dict__.items()}


@dataclass
class ColorPalette:
    """K-Means 主色调色板 (RGB 0-255)"""
    colors: List[Tuple[int, int, int]]
    weights: List[float]

    def to_dict(self):
        return {
            "colors": [list(c) for c in self.colors],
            "weights": [round(w, 4) for w in self.weights],
        }


@dataclass
class SkinFeature:
    """肤色区域特征"""
    detected: bool
    mask_ratio: float
    lab_stats: Optional[LabStats] = None
    highlight_lab: Optional[Tuple[float, float, float]] = None
    shadow_lab: Optional[Tuple[float, float, float]] = None

    def to_dict(self):
        d = {"detected": self.detected, "mask_ratio": round(self.mask_ratio, 4)}
        if self.lab_stats:
            d["lab_stats"] = self.lab_stats.to_dict()
        if self.highlight_lab:
            d["highlight_lab"] = list(self.highlight_lab)
        if self.shadow_lab:
            d["shadow_lab"] = list(self.shadow_lab)
        return d


@dataclass
class ColorFeatures:
    """完整色彩特征包"""
    tone: ToneDistribution
    lab: LabStats
    palette: ColorPalette
    skin: SkinFeature
    env_palette: ColorPalette

    def to_dict(self):
        return {
            "tone": self.tone.to_dict(),
            "lab": self.lab.to_dict(),
            "palette": self.palette.to_dict(),
            "skin": self.skin.to_dict(),
            "env_palette": self.env_palette.to_dict(),
        }


# ── 参数包 ──

@dataclass
class ToneParams:
    """基础面板 —— 影调参数 (Lightroom 风格)"""
    exposure: float = 0.0    # -5 ~ +5 EV
    contrast: float = 0.0    # -100 ~ +100
    highlights: float = 0.0  # -100 ~ +100
    shadows: float = 0.0     # -100 ~ +100
    whites: float = 0.0      # -100 ~ +100
    blacks: float = 0.0      # -100 ~ +100
    reason: str = ""

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class CurvePoint:
    input: float   # 0-255
    output: float  # 0-255


@dataclass
class CurveParams:
    """RGB 曲线 (锚点列表)"""
    rgb: List[CurvePoint] = field(default_factory=list)
    red: List[CurvePoint] = field(default_factory=list)
    green: List[CurvePoint] = field(default_factory=list)
    blue: List[CurvePoint] = field(default_factory=list)
    reason: str = ""

    def to_dict(self):
        def pts(lst):
            return [[p.input, p.output] for p in lst]
        return {"rgb": pts(self.rgb), "red": pts(self.red),
                "green": pts(self.green), "blue": pts(self.blue), "reason": self.reason}


@dataclass
class HSLChannel:
    hue: float = 0.0; saturation: float = 0.0; luminance: float = 0.0


@dataclass
class HSLParams:
    """HSL 面板 (8色通道)"""
    red: HSLChannel = field(default_factory=HSLChannel)
    orange: HSLChannel = field(default_factory=HSLChannel)
    yellow: HSLChannel = field(default_factory=HSLChannel)
    green: HSLChannel = field(default_factory=HSLChannel)
    aqua: HSLChannel = field(default_factory=HSLChannel)
    blue: HSLChannel = field(default_factory=HSLChannel)
    purple: HSLChannel = field(default_factory=HSLChannel)
    magenta: HSLChannel = field(default_factory=HSLChannel)
    reason: str = ""

    def to_dict(self):
        channels = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]
        out = {}
        for ch in channels:
            c = getattr(self, ch)
            out[ch] = {"hue": c.hue, "saturation": c.saturation, "luminance": c.luminance}
        out["reason"] = self.reason
        return out


@dataclass
class SplitToneParams:
    """分离色调"""
    highlight_hue: float = 0.0; highlight_sat: float = 0.0
    shadow_hue: float = 0.0; shadow_sat: float = 0.0
    balance: float = 0.0
    reason: str = ""

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class CalibrationParams:
    """相机校准 (白平衡偏移)"""
    temp: float = 0.0; tint: float = 0.0
    red_hue: float = 0.0; red_sat: float = 0.0
    green_hue: float = 0.0; green_sat: float = 0.0
    blue_hue: float = 0.0; blue_sat: float = 0.0
    reason: str = ""

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ColorGradingParams:
    """完整追色参数包"""
    tone: ToneParams = field(default_factory=ToneParams)
    curve: CurveParams = field(default_factory=CurveParams)
    hsl: HSLParams = field(default_factory=HSLParams)
    split_tone: SplitToneParams = field(default_factory=SplitToneParams)
    calibration: CalibrationParams = field(default_factory=CalibrationParams)
    strength: float = 1.0
    reasons: Dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return {
            "tone": self.tone.to_dict(), "curve": self.curve.to_dict(),
            "hsl": self.hsl.to_dict(), "split_tone": self.split_tone.to_dict(),
            "calibration": self.calibration.to_dict(),
            "strength": self.strength, "reasons": self.reasons,
        }

    def apply_strength(self, s: float) -> "ColorGradingParams":
        """按强度系数 s (0-1) 缩放所有数值参数"""
        import copy
        p = copy.deepcopy(self)
        for attr in ["exposure", "contrast", "highlights", "shadows", "whites", "blacks"]:
            setattr(p.tone, attr, getattr(p.tone, attr) * s)
        def scale_curve(pts):
            return [CurvePoint(pt.input, pt.input + (pt.output - pt.input) * s) for pt in pts]
        p.curve.rgb = scale_curve(p.curve.rgb)
        p.curve.red = scale_curve(p.curve.red)
        p.curve.green = scale_curve(p.curve.green)
        p.curve.blue = scale_curve(p.curve.blue)
        for ch_name in ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]:
            ch = getattr(p.hsl, ch_name)
            ch.hue *= s; ch.saturation *= s; ch.luminance *= s
        p.split_tone.highlight_sat *= s
        p.split_tone.shadow_sat *= s
        for attr in ["temp", "tint", "red_hue", "red_sat", "green_hue", "green_sat", "blue_hue", "blue_sat"]:
            setattr(p.calibration, attr, getattr(p.calibration, attr) * s)
        p.strength = s
        return p


# ══════════════════════════════════════════════════════════
#  4. 肤色检测 (HSV + Lab 双空间，替代 YCrCb+HSV)
# ══════════════════════════════════════════════════════════

def detect_skin_mask(img_rgb: np.ndarray, blur_radius: int = 15) -> np.ndarray:
    """
    生成肤色保护遮罩 (float32, 0=非肤色, 1=肤色)

    使用 HSV + Lab 双空间检测：
      - HSV: 肤色色相范围 0°-50°，低饱和中亮度
      - Lab: 亚洲人肤色范围 L∈[30,90], a∈[2,18], b∈[8,28]
    """
    u8 = (img_rgb.clip(0, 1) * 255).astype(np.uint8).astype(np.float32) / 255.0

    # ── HSV 空间检测 ──
    hsv = rgb_to_hsv(u8)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask_hsv = (
        (h >= 0) & (h <= 50) &
        (s >= 0.08) & (s <= 0.75) &
        (v >= 0.15) & (v <= 1.0)
    )

    # ── Lab 空间检测 (亚洲人肤色) ──
    lab = rgb_to_lab(u8)
    L, a_ch, b_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    mask_lab = (
        (L >= 30) & (L <= 90) &
        (a_ch >= 2) & (a_ch <= 18) &
        (b_ch >= 8) & (b_ch <= 28)
    )

    # 合并遮罩（HSV OR Lab 命中即可）
    mask = mask_hsv | mask_lab

    # 形态学平滑
    mask = binary_closing(mask, iterations=2)
    mask = binary_opening(mask, iterations=1)
    # 膨胀
    from scipy.ndimage import binary_dilation
    mask = binary_dilation(mask, iterations=2)

    # 高斯模糊羽化边缘
    mask_f = mask.astype(np.float32)
    mask_f = gaussian_filter(mask_f, sigma=blur_radius)
    return mask_f.clip(0, 1)


# ══════════════════════════════════════════════════════════
#  5. 特征提取器
# ══════════════════════════════════════════════════════════

def _kmeans_palette(pixels: np.ndarray, k: int = 5) -> Tuple[List, List]:
    """K-Means 提取 k 个主色"""
    if len(pixels) < k:
        k = max(1, len(pixels))

    if _HAS_SKLEARN:
        km = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42)
        labels = km.fit_predict(pixels)
        centers = (km.cluster_centers_ * 255).clip(0, 255).astype(int)
    else:
        # 简易 K-Means (NumPy)
        idx = np.random.choice(len(pixels), k, replace=False)
        centers_f = pixels[idx].copy()
        for _ in range(20):
            dists = np.linalg.norm(pixels[:, None, :] - centers_f[None, :, :], axis=-1)
            labels = np.argmin(dists, axis=-1)
            for ki in range(k):
                members = pixels[labels == ki]
                if len(members) > 0:
                    centers_f[ki] = members.mean(axis=0)
        centers = (centers_f * 255).clip(0, 255).astype(int)

    counts = np.bincount(labels, minlength=k)
    weights = counts / counts.sum()
    order = np.argsort(-weights)
    colors_out = [tuple(centers[i]) for i in order]
    weights_sorted = [float(weights[i]) for i in order]
    return colors_out, weights_sorted


def _lab_stats_from_pixels(lab_pixels: np.ndarray) -> LabStats:
    """(N, 3) Lab 像素 → LabStats"""
    return LabStats(
        mean_L=float(lab_pixels[:, 0].mean()), std_L=float(lab_pixels[:, 0].std()),
        mean_a=float(lab_pixels[:, 1].mean()), std_a=float(lab_pixels[:, 1].std()),
        mean_b=float(lab_pixels[:, 2].mean()), std_b=float(lab_pixels[:, 2].std()),
    )


class FeatureExtractor:
    """
    色彩特征提取器

    用法:
        extractor = FeatureExtractor()
        features = extractor.extract(img_rgb_float)
    """

    def extract(self, img: np.ndarray, max_pixels: int = 50000) -> ColorFeatures:
        """
        提取完整色彩特征包

        img: float32 RGB [0,1], shape (H, W, 3)
        max_pixels: 下采样像素数（加速统计）
        """
        H, W = img.shape[:2]
        flat = img.reshape(-1, 3)

        # 下采样
        if len(flat) > max_pixels:
            idx = np.random.choice(len(flat), max_pixels, replace=False)
            flat_sample = flat[idx]
        else:
            flat_sample = flat
            idx = None

        # Lab 统计
        lab_img = rgb_to_lab(img)
        lab_flat = lab_img.reshape(-1, 3)
        lab_sample = lab_flat[idx] if idx is not None else lab_flat
        lab_stats = _lab_stats_from_pixels(lab_sample)

        # 影调分布
        L_vals = lab_sample[:, 0]
        tone = ToneDistribution(
            p5=float(np.percentile(L_vals, 5)),
            p25=float(np.percentile(L_vals, 25)),
            p50=float(np.percentile(L_vals, 50)),
            p75=float(np.percentile(L_vals, 75)),
            p95=float(np.percentile(L_vals, 95)),
            mean=float(L_vals.mean()), std=float(L_vals.std()),
        )

        # 全图主色调色板
        colors, weights = _kmeans_palette(flat_sample, k=5)
        palette = ColorPalette(colors=colors, weights=weights)

        # 肤色区域
        skin_mask_f = detect_skin_mask(img)
        skin_bool = skin_mask_f > 0.5
        skin_ratio = float(skin_bool.sum()) / (H * W)
        skin_detected = skin_ratio > 0.02

        skin_feature = SkinFeature(detected=skin_detected, mask_ratio=skin_ratio)
        env_colors, env_weights = colors, weights

        if skin_detected:
            skin_lab_pixels = lab_flat[skin_bool.flatten()]
            if len(skin_lab_pixels) > 0:
                skin_feature.lab_stats = _lab_stats_from_pixels(skin_lab_pixels)
                # 高光/阴影肤色锚点
                skin_L = skin_lab_pixels[:, 0]
                hi_thr = np.percentile(skin_L, 75)
                lo_thr = np.percentile(skin_L, 25)
                hi_px = skin_lab_pixels[skin_L >= hi_thr]
                lo_px = skin_lab_pixels[skin_L <= lo_thr]
                if len(hi_px) > 0:
                    skin_feature.highlight_lab = tuple(hi_px.mean(axis=0))
                if len(lo_px) > 0:
                    skin_feature.shadow_lab = tuple(lo_px.mean(axis=0))

            # 环境主色
            env_bool = ~skin_bool.flatten()
            if env_bool.sum() > 5:
                env_pixels = flat[env_bool]
                if len(env_pixels) > max_pixels:
                    env_idx = np.random.choice(len(env_pixels), max_pixels, replace=False)
                    env_pixels = env_pixels[env_idx]
                env_colors, env_weights = _kmeans_palette(env_pixels, k=5)

        env_palette = ColorPalette(colors=env_colors, weights=env_weights)

        return ColorFeatures(
            tone=tone, lab=lab_stats, palette=palette,
            skin=skin_feature, env_palette=env_palette,
        )


# ══════════════════════════════════════════════════════════
#  6. 追色方案生成器
# ══════════════════════════════════════════════════════════

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ev_from_L_diff(dL: float) -> float:
    """Lab L 差值 → Lightroom 曝光补偿 EV"""
    return _clamp(dL / 8.0, -5.0, 5.0)


def _contrast_from_spread(ref_std: float, src_std: float) -> float:
    """对比度: ref std vs src std → [-100, +100]"""
    return _clamp((ref_std - src_std) * 4.0, -100.0, 100.0)


def _highlight_adj(ref_p95: float, src_p95: float) -> float:
    return _clamp((ref_p95 - src_p95) * 2.5, -100.0, 100.0)


def _shadow_adj(ref_p5: float, src_p5: float) -> float:
    return _clamp((ref_p5 - src_p5) * 2.5, -100.0, 100.0)


def _lab_b_to_temp(delta_b: float) -> float:
    """Lab b* 差 → 色温 -100(蓝)~+100(橙)"""
    return _clamp(delta_b * 3.5, -100.0, 100.0)


def _lab_a_to_tint(delta_a: float) -> float:
    """Lab a* 差 → 色调 -100(绿)~+100(品)"""
    return _clamp(delta_a * 3.5, -100.0, 100.0)


def _build_tone_curve(ref_tone: ToneDistribution, src_tone: ToneDistribution) -> List[CurvePoint]:
    """生成 RGB 主曲线锚点"""
    def to255(v):
        return _clamp(v / 100.0 * 255.0, 0, 255)

    pts = [
        CurvePoint(0, 0),
        CurvePoint(to255(src_tone.p5), _clamp(to255(ref_tone.p5), 0, 60)),
        CurvePoint(to255(src_tone.p50), _clamp(to255(ref_tone.p50), 50, 200)),
        CurvePoint(to255(src_tone.p95), _clamp(to255(ref_tone.p95), 180, 255)),
        CurvePoint(255, 255),
    ]
    # 去重 + 排序
    seen = {}
    for p in pts:
        if p.input not in seen:
            seen[p.input] = p
    return sorted(seen.values(), key=lambda p: p.input)


def _build_color_curves(ref_lab: LabStats, src_lab: LabStats):
    """生成 R/G/B 分通道曲线"""
    delta_b = ref_lab.mean_b - src_lab.mean_b
    delta_a = ref_lab.mean_a - src_lab.mean_a

    red_shift   = _clamp(delta_a * 1.5 + delta_b * 0.8, -30, 30)
    green_shift = _clamp(-delta_a * 0.5, -20, 20)
    blue_shift  = _clamp(-delta_b * 1.5, -30, 30)

    def mid_curve(shift: float) -> List[CurvePoint]:
        return [CurvePoint(0, 0), CurvePoint(128, _clamp(128 + shift, 80, 175)), CurvePoint(255, 255)]

    return mid_curve(red_shift), mid_curve(green_shift), mid_curve(blue_shift)


def _build_hsl_params(ref_lab: LabStats, src_lab: LabStats) -> HSLParams:
    """生成 HSL 面板参数"""
    ref_sat = (ref_lab.std_a + ref_lab.std_b) / 2.0
    src_sat = (src_lab.std_a + src_lab.std_b) / 2.0
    sat_diff = _clamp((ref_sat - src_sat) * 5.0, -60.0, 60.0)
    delta_a = ref_lab.mean_a - src_lab.mean_a
    delta_b = ref_lab.mean_b - src_lab.mean_b
    orange_hue = _clamp(delta_a * 1.5, -10.0, 10.0)
    blue_hue  = _clamp(-delta_b * 1.2, -20.0, 20.0)
    yellow_hue = _clamp(delta_b * 1.0, -15.0, 15.0)

    return HSLParams(
        red=HSLChannel(hue=_clamp(delta_a * 1.0, -8, 8), saturation=sat_diff * 0.7),
        orange=HSLChannel(hue=orange_hue, saturation=sat_diff * 0.5),
        yellow=HSLChannel(hue=yellow_hue, saturation=sat_diff * 0.8),
        green=HSLChannel(saturation=sat_diff * 0.9),
        aqua=HSLChannel(saturation=sat_diff),
        blue=HSLChannel(hue=blue_hue, saturation=sat_diff),
        purple=HSLChannel(saturation=sat_diff * 0.8),
        magenta=HSLChannel(hue=_clamp(-delta_a * 0.8, -12, 12), saturation=sat_diff * 0.6),
        reason=f"饱和度差Δ={sat_diff:.1f}；a*差Δ={delta_a:.2f}→橙色色相{orange_hue:+.1f}；b*差Δ={delta_b:.2f}→蓝色色相{blue_hue:+.1f}",
    )


def _build_split_tone(ref_lab: LabStats, src_lab: LabStats) -> SplitToneParams:
    """分离色调"""
    ref_b, ref_a = ref_lab.mean_b, ref_lab.mean_a
    hi_hue = float(np.degrees(np.arctan2(ref_b, ref_a)) % 360)
    hi_sat = _clamp(np.sqrt(ref_lab.std_a ** 2 + ref_lab.std_b ** 2) * 2.0, 0, 30)
    sh_hue = (hi_hue + 180) % 360
    sh_sat = hi_sat * 0.6

    return SplitToneParams(
        highlight_hue=round(hi_hue, 1), highlight_sat=round(hi_sat, 1),
        shadow_hue=round(sh_hue, 1), shadow_sat=round(sh_sat, 1),
        reason=f"色相方向≈{hi_hue:.0f}°，高光饱和≈{hi_sat:.1f}，阴影补色≈{sh_hue:.0f}°",
    )


class ColorGradingGenerator:
    """
    差异分析与追色方案生成器

    用法:
        gen = ColorGradingGenerator()
        params = gen.generate(ref_features, src_features, strength=0.8)
    """

    def generate(self, ref: ColorFeatures, src: ColorFeatures, strength: float = 1.0) -> ColorGradingParams:
        reasons = {}

        # ── 影调参数 ──
        dL_mean = ref.lab.mean_L - src.lab.mean_L
        exposure = _ev_from_L_diff(dL_mean)
        contrast = _contrast_from_spread(ref.lab.std_L, src.lab.std_L)
        highlights = _highlight_adj(ref.tone.p95, src.tone.p95)
        shadows = _shadow_adj(ref.tone.p5, src.tone.p5)
        whites = _clamp((ref.tone.p95 - src.tone.p95) * 1.5, -60, 60)
        blacks = _clamp((ref.tone.p5 - src.tone.p5) * 1.5, -60, 60)

        reasons["tone"] = (
            f"原片L均值={src.lab.mean_L:.1f}，参考={ref.lab.mean_L:.1f}，Δ={dL_mean:.1f}→曝光{exposure:+.2f}EV；"
            f"对比度{contrast:+.0f}，高光{highlights:+.0f}，阴影{shadows:+.0f}"
        )

        tone = ToneParams(
            exposure=round(exposure, 2), contrast=round(contrast, 1),
            highlights=round(highlights, 1), shadows=round(shadows, 1),
            whites=round(whites, 1), blacks=round(blacks, 1),
            reason=reasons["tone"],
        )

        # ── 曲线 ──
        rgb_curve = _build_tone_curve(ref.tone, src.tone)
        red_c, green_c, blue_c = _build_color_curves(ref.lab, src.lab)
        delta_b = ref.lab.mean_b - src.lab.mean_b
        delta_a = ref.lab.mean_a - src.lab.mean_a
        reasons["curve"] = (
            f"b*差Δ={delta_b:.2f}→蓝通道{'下压' if delta_b > 0 else '上提'}；"
            f"a*差Δ={delta_a:.2f}→红通道{'上提' if delta_a > 0 else '下压'}"
        )
        curve = CurveParams(rgb=rgb_curve, red=red_c, green=green_c, blue=blue_c, reason=reasons["curve"])

        # ── HSL ──
        hsl = _build_hsl_params(ref.lab, src.lab)
        reasons["hsl"] = hsl.reason

        # ── 分离色调 ──
        split = _build_split_tone(ref.lab, src.lab)
        reasons["split_tone"] = split.reason

        # ── 校准 ──
        temp_adj = _lab_b_to_temp(delta_b)
        tint_adj = _lab_a_to_tint(delta_a)
        reasons["calibration"] = (
            f"b*差{delta_b:+.2f}→色温{temp_adj:+.0f}；a*差{delta_a:+.2f}→色调{tint_adj:+.0f}"
        )
        calibration = CalibrationParams(
            temp=round(temp_adj, 1), tint=round(tint_adj, 1),
            reason=reasons["calibration"],
        )

        params = ColorGradingParams(
            tone=tone, curve=curve, hsl=hsl, split_tone=split,
            calibration=calibration, strength=1.0, reasons=reasons,
        )

        if strength != 1.0:
            params = params.apply_strength(strength)

        return params


# ══════════════════════════════════════════════════════════
#  7. 渲染器 (含肤色保护 ΔE 监控)
# ══════════════════════════════════════════════════════════

def _delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE76 ΔE"""
    diff = lab1.astype(np.float64) - lab2.astype(np.float64)
    return np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)


def _interp_curve(img_channel: np.ndarray, curve_pts: list) -> np.ndarray:
    """将曲线锚点插值应用到单通道 (0-1 float)"""
    if not curve_pts:
        return img_channel
    xs = np.array([p.input for p in curve_pts], dtype=float)
    ys = np.array([p.output for p in curve_pts], dtype=float)
    if xs[0] > 0:
        xs = np.insert(xs, 0, 0); ys = np.insert(ys, 0, 0)
    if xs[-1] < 255:
        xs = np.append(xs, 255); ys = np.append(ys, 255)
    src_255 = (img_channel * 255).clip(0, 255)
    out_255 = np.interp(src_255, xs, ys)
    return (out_255 / 255.0).clip(0, 1).astype(np.float32)


class ColorRenderer:
    """
    将追色参数应用到原图，并执行肤色保护

    流程:
      A. 全局应用追色参数 → result_full
      B. 生成肤色蒙版
      C. ΔE 检测 → 按阈值决定肤色区处理
      D. 混合环境色与肤色区
    """

    SKIN_DE_THRESHOLD = 5.0
    SKIN_BLUR_SIGMA = 20

    def render(self, src: np.ndarray, params: ColorGradingParams,
               ref_features: ColorFeatures = None) -> np.ndarray:
        """
        src: float32 RGB [0,1]
        params: ColorGradingParams
        ref_features: 参考图特征（可选，用于肤色锚点）
        返回: float32 RGB [0,1]
        """
        # Step A: 全局追色
        result_full = self._apply_grading(src, params)

        # Step B: 肤色蒙版
        skin_mask_f = detect_skin_mask(src)
        skin_bool = skin_mask_f > 0.5

        if not skin_bool.any():
            return result_full.clip(0, 1)

        mask_f = gaussian_filter(skin_mask_f, sigma=self.SKIN_BLUR_SIGMA).clip(0, 1)
        mask_3ch = mask_f[:, :, np.newaxis]

        # Step C: ΔE 检测
        src_lab = rgb_to_lab(src)
        result_lab = rgb_to_lab(result_full)
        de_map = _delta_e_cie76(src_lab, result_lab)
        skin_de_mean = float(de_map[skin_bool].mean())

        # Step D: 肤色区处理
        if skin_de_mean <= self.SKIN_DE_THRESHOLD:
            skin_result = result_full * 0.7 + src * 0.3
        else:
            skin_result = self._skin_conservative_adjust(src, params)

        final = result_full * (1 - mask_3ch) + skin_result * mask_3ch
        return final.clip(0, 1)

    def _apply_grading(self, src: np.ndarray, p: ColorGradingParams) -> np.ndarray:
        """模拟 Lightroom 处理管线"""
        img = src.copy().astype(np.float32)

        # 1. 曝光
        img = img * (2 ** p.tone.exposure)

        # 2. 对比度
        if p.tone.contrast != 0:
            c = p.tone.contrast / 100.0
            img = (img - 0.5) * (1 + c) + 0.5

        # 3. 高光/阴影/白色/黑色
        img = self._apply_tone_controls(img, p.tone)

        # 4. RGB 曲线
        if p.curve.rgb:
            lum = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
            lum_new = _interp_curve(lum, p.curve.rgb)
            scale = np.where(lum > 1e-4, lum_new / (lum + 1e-4), 1.0)[:, :, np.newaxis]
            img = img * scale

        if p.curve.red:
            img[:, :, 0] = _interp_curve(img[:, :, 0], p.curve.red)
        if p.curve.green:
            img[:, :, 1] = _interp_curve(img[:, :, 1], p.curve.green)
        if p.curve.blue:
            img[:, :, 2] = _interp_curve(img[:, :, 2], p.curve.blue)

        # 5. 分离色调
        img = self._apply_split_tone(img, p.split_tone)

        return img.clip(0, 1)

    def _apply_tone_controls(self, img: np.ndarray, tone: ToneParams) -> np.ndarray:
        """高光/阴影/白色/黑色滑块"""
        lum = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        hi_mask = np.clip((lum - 0.5) * 2, 0, 1)[:, :, np.newaxis]
        lo_mask = np.clip((0.5 - lum) * 2, 0, 1)[:, :, np.newaxis]

        hi_adj = tone.highlights / 500.0
        img = img + hi_adj * hi_mask * (1 - img)

        sh_adj = tone.shadows / 500.0
        img = img + sh_adj * lo_mask * img

        wh_adj = tone.whites / 500.0
        wh_mask = np.clip((lum - 0.8) * 5, 0, 1)[:, :, np.newaxis]
        img = img + wh_adj * wh_mask

        bl_adj = tone.blacks / 500.0
        bl_mask = np.clip((0.2 - lum) * 5, 0, 1)[:, :, np.newaxis]
        img = img + bl_adj * bl_mask

        return img

    def _apply_split_tone(self, img: np.ndarray, st: SplitToneParams) -> np.ndarray:
        """分离色调：高光加色 + 阴影加色"""
        if st.highlight_sat < 1 and st.shadow_sat < 1:
            return img

        lab = rgb_to_lab(img)
        L = lab[:, :, 0]

        hi_mask = np.clip((L - 60) / 40, 0, 1)[:, :, np.newaxis]
        sh_mask = np.clip((40 - L) / 40, 0, 1)[:, :, np.newaxis]

        hi_hue_rad = np.radians(st.highlight_hue)
        sh_hue_rad = np.radians(st.shadow_hue)
        hi_sat_norm = st.highlight_sat / 100.0 * 20
        sh_sat_norm = st.shadow_sat / 100.0 * 15

        hi_da = np.cos(hi_hue_rad) * hi_sat_norm
        hi_db = np.sin(hi_hue_rad) * hi_sat_norm
        sh_da = np.cos(sh_hue_rad) * sh_sat_norm
        sh_db = np.sin(sh_hue_rad) * sh_sat_norm

        lab[:, :, 1] += float(hi_da) * hi_mask[:, :, 0] + float(sh_da) * sh_mask[:, :, 0]
        lab[:, :, 2] += float(hi_db) * hi_mask[:, :, 0] + float(sh_db) * sh_mask[:, :, 0]

        lab[:, :, 0] = lab[:, :, 0].clip(0, 100)
        lab[:, :, 1] = lab[:, :, 1].clip(-128, 127)
        lab[:, :, 2] = lab[:, :, 2].clip(-128, 127)

        return lab_to_rgb(lab)

    def _skin_conservative_adjust(self, src: np.ndarray, params: ColorGradingParams) -> np.ndarray:
        """ΔE 超标时肤色区保守处理：仅保留影调 + 提亮 + 保暖"""
        lab = rgb_to_lab(src)

        ev_scale = 2 ** (params.tone.exposure * 0.5)
        lab[:, :, 0] = (lab[:, :, 0] * ev_scale).clip(0, 100)
        lab[:, :, 0] = (lab[:, :, 0] + 1.5).clip(0, 100)
        lab[:, :, 1] = (lab[:, :, 1] + 1.0).clip(-128, 127)
        lab[:, :, 2] = (lab[:, :, 2] + 2.0).clip(-128, 127)

        return lab_to_rgb(lab)


# ══════════════════════════════════════════════════════════
#  8. 简化追色引擎 (Reinhard Lab 迁移)
# ══════════════════════════════════════════════════════════

class ColorTransfer:
    """
    基于 Lab 空间的 Reinhard 颜色迁移（简化版）

    支持影调/色调强度独立控制、肤色保护
    """

    def __init__(self):
        self.ref_stats = None
        self.ref_features = None

    def analyze(self, ref_img: np.ndarray) -> dict:
        """分析参考图，提取 Lab 统计特征"""
        lab = rgb_to_lab(ref_img)
        L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        self.ref_stats = {
            'mean_L': float(L.mean()), 'std_L': float(L.std()),
            'mean_a': float(a.mean()), 'std_a': float(a.std()),
            'mean_b': float(b.mean()), 'std_b': float(b.std()),
        }
        # 同时提取完整特征
        extractor = FeatureExtractor()
        self.ref_features = extractor.extract(ref_img)
        return self.ref_stats

    def transfer(
        self,
        src_img: np.ndarray,
        tone_strength: float = 1.0,
        color_strength: float = 1.0,
        skin_protect: float = 0.85,
        use_skin_protect: bool = True,
    ) -> np.ndarray:
        """
        执行追色 (Reinhard 方法)

        参数:
            src_img:        float32 RGB [0,1]
            tone_strength:  影调匹配强度 0~1
            color_strength: 色调匹配强度 0~1
            skin_protect:   肤色保护强度 0~1 (1=完全保护)
            use_skin_protect: 是否启用肤色保护
        """
        if self.ref_stats is None:
            raise ValueError("请先调用 analyze() 分析参考图")

        s = self.ref_stats
        src_lab = rgb_to_lab(src_img)
        result_lab = src_lab.copy()

        src_L, src_a, src_b = src_lab[:, :, 0], src_lab[:, :, 1], src_lab[:, :, 2]
        src_mean_L, src_std_L = float(src_L.mean()), float(src_L.std())
        src_mean_a, src_std_a = float(src_a.mean()), float(src_a.std())
        src_mean_b, src_std_b = float(src_b.mean()), float(src_b.std())
        eps = 1e-6

        # L 通道 (影调)
        transferred_L = (src_L - src_mean_L) * (s['std_L'] / (src_std_L + eps)) + s['mean_L']
        result_lab[:, :, 0] = src_L + tone_strength * (transferred_L - src_L)

        # a 通道 (绿-红)
        transferred_a = (src_a - src_mean_a) * (s['std_a'] / (src_std_a + eps)) + s['mean_a']
        result_lab[:, :, 1] = src_a + color_strength * (transferred_a - src_a)

        # b 通道 (蓝-黄)
        transferred_b = (src_b - src_mean_b) * (s['std_b'] / (src_std_b + eps)) + s['mean_b']
        result_lab[:, :, 2] = src_b + color_strength * (transferred_b - src_b)

        result_lab[:, :, 0] = result_lab[:, :, 0].clip(0, 100)
        result_lab[:, :, 1] = result_lab[:, :, 1].clip(-128, 127)
        result_lab[:, :, 2] = result_lab[:, :, 2].clip(-128, 127)

        result_rgb = lab_to_rgb(result_lab)

        # 肤色保护
        if use_skin_protect and skin_protect > 0:
            mask = detect_skin_mask(src_img)
            mask_3ch = mask[:, :, np.newaxis] * skin_protect
            result_rgb = result_rgb * (1 - mask_3ch) + src_img * mask_3ch

        return result_rgb.clip(0, 1)


# ══════════════════════════════════════════════════════════
#  9. LUT 生成器
# ══════════════════════════════════════════════════════════

class LUTGenerator:
    """
    基于 ColorTransfer 生成标准 .cube 3D LUT 文件
    可直接导入 PR / AE / 达芬奇 / Lightroom
    """

    def __init__(self, lut_size: int = 33):
        self.lut_size = lut_size

    def generate(
        self,
        transfer: ColorTransfer,
        tone_strength: float = 1.0,
        color_strength: float = 1.0,
        output_path: str = "output.cube",
        title: str = "Zhuise LUT",
    ):
        """生成 .cube 文件 (LUT 不含肤色保护，是全局映射)"""
        n = self.lut_size
        lin = np.linspace(0, 1, n, dtype=np.float32)
        r, g, b = np.meshgrid(lin, lin, lin, indexing='ij')
        identity = np.stack([r, g, b], axis=-1).reshape(n * n, n, 3)

        mapped = transfer.transfer(
            identity,
            tone_strength=tone_strength,
            color_strength=color_strength,
            use_skin_protect=False,
        ).reshape(n, n, n, 3)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f'TITLE "{title}"\n')
            f.write(f'LUT_3D_SIZE {n}\n')
            f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
            f.write('DOMAIN_MAX 1.0 1.0 1.0\n\n')
            for ri in range(n):
                for gi in range(n):
                    for bi in range(n):
                        rv, gv, bv = mapped[ri, gi, bi]
                        f.write(f'{rv:.6f} {gv:.6f} {bv:.6f}\n')

        return output_path

    def generate_from_params(
        self,
        engine: 'ZhuiseEngine',
        output_path: str = "output.cube",
        title: str = "Zhuise LUT",
    ):
        """基于 ZhuiseEngine 的当前参数生成 LUT"""
        n = self.lut_size
        lin = np.linspace(0, 1, n, dtype=np.float32)
        r, g, b = np.meshgrid(lin, lin, lin, indexing='ij')
        identity = np.stack([r, g, b], axis=-1).reshape(n * n, n, 3)

        mapped = engine.render(identity).reshape(n, n, n, 3)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f'TITLE "{title}"\n')
            f.write(f'LUT_3D_SIZE {n}\n')
            f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
            f.write('DOMAIN_MAX 1.0 1.0 1.0\n\n')
            for ri in range(n):
                for gi in range(n):
                    for bi in range(n):
                        rv, gv, bv = mapped[ri, gi, bi]
                        f.write(f'{rv:.6f} {gv:.6f} {bv:.6f}\n')

        return output_path


# ══════════════════════════════════════════════════════════
#  10. 评价指标
# ══════════════════════════════════════════════════════════

class Evaluator:
    """追色效果评价指标"""

    @staticmethod
    def histogram_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
        """L 通道直方图 Bhattacharyya 相似度 (0-1，越高越像)"""
        def get_l_hist(img):
            lab = rgb_to_lab(img)
            L = lab[:, :, 0]
            hist, _ = np.histogram(L.ravel(), bins=256, range=(0, 100), density=True)
            hist = hist.astype(np.float64)
            hist /= (hist.sum() + 1e-10)
            return hist

        h_a, h_b = get_l_hist(img_a), get_l_hist(img_b)
        bc = np.sum(np.sqrt(h_a * h_b))
        dist = np.sqrt(1 - bc)
        return round(float(max(0, 1 - dist)), 4)

    @staticmethod
    def skin_delta_e(src: np.ndarray, result: np.ndarray) -> dict:
        """肤色区域 ΔE CIE76"""
        skin_mask_f = detect_skin_mask(src)
        skin_bool = skin_mask_f > 0.5

        if not skin_bool.any():
            return {"mean_de": 0.0, "max_de": 0.0, "safe": True, "skin_detected": False}

        src_lab = rgb_to_lab(src)
        result_lab = rgb_to_lab(result)
        de = _delta_e_cie76(src_lab, result_lab)
        skin_de_vals = de[skin_bool]
        mean_de = float(skin_de_vals.mean())
        max_de = float(skin_de_vals.max())

        return {
            "mean_de": round(mean_de, 2), "max_de": round(max_de, 2),
            "safe": mean_de < 5.0, "skin_detected": True,
        }

    @staticmethod
    def lab_match_score(ref_feat: ColorFeatures, result: np.ndarray) -> dict:
        """追色结果与参考图 Lab 均值匹配度"""
        result_lab = rgb_to_lab(result)
        res_L = float(result_lab[:, :, 0].mean())
        res_a = float(result_lab[:, :, 1].mean())
        res_b = float(result_lab[:, :, 2].mean())

        dL = abs(res_L - ref_feat.lab.mean_L)
        da = abs(res_a - ref_feat.lab.mean_a)
        db = abs(res_b - ref_feat.lab.mean_b)
        overall = float(np.sqrt(dL ** 2 + da ** 2 + db ** 2))

        return {
            "dL": round(dL, 2), "da": round(da, 2), "db": round(db, 2),
            "overall_de": round(overall, 2),
            "grade": "优秀" if overall < 5 else "良好" if overall < 10 else "待优化",
        }

    @staticmethod
    def evaluate(src: np.ndarray, ref: np.ndarray, result: np.ndarray,
                 ref_feat: ColorFeatures, src_feat: ColorFeatures) -> dict:
        """综合评价"""
        hist_sim = Evaluator.histogram_similarity(ref, result)
        skin_de = Evaluator.skin_delta_e(src, result)
        lab_match = Evaluator.lab_match_score(ref_feat, result)

        return {
            "histogram_similarity": hist_sim,
            "skin_delta_e": skin_de,
            "lab_match": lab_match,
            "summary": (
                f"影调相似度 {hist_sim:.1%}，"
                f"Lab偏差 ΔE={lab_match['overall_de']:.1f}（{lab_match['grade']}），"
                f"肤色安全：{'✓' if skin_de.get('safe', True) else '✗'}"
            ),
        }


# ══════════════════════════════════════════════════════════
#  11. 风格预设
# ══════════════════════════════════════════════════════════

@dataclass
class StylePreset:
    """风格预设"""
    name: str
    name_en: str
    description: str
    # 简化模式参数
    tone_strength: float = 0.8
    color_strength: float = 0.9
    skin_protect: float = 0.85
    # 专家模式参数
    expert_params: Optional[ColorGradingParams] = None
    # 参考图特征 (可选，用于自动追色)
    ref_features: Optional[ColorFeatures] = None


BUILTIN_PRESETS: Dict[str, StylePreset] = {
    "日系清新": StylePreset(
        name="日系清新", name_en="Japanese Fresh",
        description="滨田英明风格：高亮度低对比，微冷偏青白，低饱和通透肤色",
        tone_strength=0.7, color_strength=0.8, skin_protect=0.9,
        expert_params=ColorGradingParams(
            tone=ToneParams(exposure=0.8, contrast=-20, highlights=-30, shadows=+15, blacks=+10),
            split_tone=SplitToneParams(highlight_hue=200, highlight_sat=8, shadow_hue=40, shadow_sat=5),
            hsl=HSLParams(
                blue=HSLChannel(saturation=+15),
                orange=HSLChannel(hue=-8, saturation=-10),
            ),
            calibration=CalibrationParams(temp=-40),
            reasons={"preset": "日系清新预设：提曝光+降对比+青白高光+暖阴影"},
        ),
    ),
    "暗调电影": StylePreset(
        name="暗调电影", name_en="Cinematic Dark",
        description="低调高反差，冷蓝暗调，适合夜景/情绪人像",
        tone_strength=0.85, color_strength=0.9, skin_protect=0.85,
        expert_params=ColorGradingParams(
            tone=ToneParams(exposure=-0.7, contrast=+35, highlights=-20, shadows=-10, blacks=-40),
            split_tone=SplitToneParams(highlight_hue=30, highlight_sat=12, shadow_hue=220, shadow_sat=15),
            calibration=CalibrationParams(temp=-20),
            reasons={"preset": "暗调电影预设：降曝光+高对比+蓝冷阴影+暖色高光"},
        ),
    ),
    "复古胶片": StylePreset(
        name="复古胶片", name_en="Vintage Film",
        description="提黑场复古感，高光偏橙阴影偏绿，胶片颗粒感色调",
        tone_strength=0.8, color_strength=0.85, skin_protect=0.8,
        expert_params=ColorGradingParams(
            tone=ToneParams(exposure=0.2, contrast=-10, highlights=-15, shadows=+10, blacks=+35),
            split_tone=SplitToneParams(highlight_hue=35, highlight_sat=20, shadow_hue=160, shadow_sat=12),
            hsl=HSLParams(
                orange=HSLChannel(saturation=+15),
                yellow=HSLChannel(saturation=+10),
            ),
            calibration=CalibrationParams(temp=+15, tint=+5),
            reasons={"preset": "复古胶片预设：提黑场+橙暖高光+绿冷阴影"},
        ),
    ),
    "青橙色调": StylePreset(
        name="青橙色调", name_en="Teal & Orange",
        description="电影级青橙对比色调，暗部偏青亮部偏橙",
        tone_strength=0.8, color_strength=0.95, skin_protect=0.75,
        expert_params=ColorGradingParams(
            tone=ToneParams(contrast=+20, highlights=-10, shadows=-5),
            split_tone=SplitToneParams(highlight_hue=30, highlight_sat=25, shadow_hue=190, shadow_sat=20),
            hsl=HSLParams(
                orange=HSLChannel(saturation=+25),
                aqua=HSLChannel(hue=-10, saturation=+20),
                blue=HSLChannel(hue=-15, saturation=+15),
            ),
            calibration=CalibrationParams(temp=+10),
            reasons={"preset": "青橙色调预设：阴影加青+高光加橙+整体增饱和"},
        ),
    ),
    "Kodak Portra": StylePreset(
        name="Kodak Portra", name_en="Kodak Portra",
        description="经典人像胶片色调：柔和暖白，肤色奶油般通透",
        tone_strength=0.75, color_strength=0.8, skin_protect=0.9,
        expert_params=ColorGradingParams(
            tone=ToneParams(exposure=0.3, contrast=-8, highlights=-10, shadows=+8, blacks=+15),
            split_tone=SplitToneParams(highlight_hue=40, highlight_sat=10, shadow_hue=50, shadow_sat=5),
            hsl=HSLParams(
                orange=HSLChannel(hue=-5, saturation=+8),
                yellow=HSLChannel(saturation=+5),
                red=HSLChannel(saturation=-5),
            ),
            calibration=CalibrationParams(temp=+10, tint=+5),
            reasons={"preset": "Kodak Portra预设：柔和曝光+暖白偏色+肤色通透"},
        ),
    ),
}


# ══════════════════════════════════════════════════════════
#  12. 统一引擎 (ZhuiseEngine)
# ══════════════════════════════════════════════════════════

class ZhuiseEngine:
    """
    追色统一引擎 — 整合简化模式 + 专家模式

    两种工作模式:
      1. 简化模式 (Reinhard): 使用 ColorTransfer，影调/色调双滑块
      2. 专家模式 (Lightroom): 使用 ColorGradingGenerator + ColorRenderer，完整 LR 参数面板

    用法:
        engine = ZhuiseEngine()
        engine.load_reference(ref_img)
        result = engine.render(src_img, mode='expert', strength=0.8)
    """

    def __init__(self):
        self.ref_img: Optional[np.ndarray] = None
        self.ref_features: Optional[ColorFeatures] = None
        self.ref_stats: Optional[dict] = None

        self._extractor = FeatureExtractor()
        self._generator = ColorGradingGenerator()
        self._renderer = ColorRenderer()
        self._transfer = ColorTransfer()
        self._evaluator = Evaluator()

        # 当前专家模式参数
        self.current_params: Optional[ColorGradingParams] = None

    def load_reference(self, ref_img: np.ndarray):
        """加载并分析参考图"""
        self.ref_img = ref_img.copy()
        self.ref_features = self._extractor.extract(ref_img)
        self._transfer.analyze(ref_img)
        self.ref_stats = self._transfer.ref_stats

    def render(
        self,
        src_img: np.ndarray,
        mode: str = 'expert',
        strength: float = 1.0,
        tone_strength: float = 0.8,
        color_strength: float = 0.9,
        skin_protect: float = 0.85,
        use_skin_protect: bool = True,
        params: Optional[ColorGradingParams] = None,
    ) -> np.ndarray:
        """
        执行追色

        mode: 'simple' (Reinhard) 或 'expert' (Lightroom 风格)
        """
        if self.ref_features is None:
            raise ValueError("请先加载参考图 (load_reference)")

        if mode == 'simple':
            result = self._transfer.transfer(
                src_img,
                tone_strength=tone_strength,
                color_strength=color_strength,
                skin_protect=skin_protect,
                use_skin_protect=use_skin_protect,
            )
        else:
            # 专家模式
            src_features = self._extractor.extract(src_img)
            if params is not None:
                grading_params = params
            else:
                grading_params = self._generator.generate(
                    self.ref_features, src_features, strength=strength
                )
            self.current_params = grading_params
            result = self._renderer.render(src_img, grading_params, self.ref_features)

        return result

    def evaluate(self, src_img: np.ndarray, result_img: np.ndarray) -> dict:
        """评价追色效果"""
        if self.ref_features is None:
            return {"error": "未加载参考图"}
        src_features = self._extractor.extract(src_img)
        return self._evaluator.evaluate(
            src_img, self.ref_img, result_img, self.ref_features, src_features
        )

    def export_lut(self, output_path: str, mode: str = 'simple',
                    tone_strength: float = 0.8, color_strength: float = 0.9,
                    lut_size: int = 33, title: str = "Zhuise LUT"):
        """导出 .cube LUT 文件"""
        lut_gen = LUTGenerator(lut_size=lut_size)
        if mode == 'simple':
            return lut_gen.generate(
                self._transfer, tone_strength=tone_strength,
                color_strength=color_strength, output_path=output_path, title=title,
            )
        else:
            return lut_gen.generate_from_params(self, output_path=output_path, title=title)

    def apply_preset(self, preset_name: str) -> Optional[ColorGradingParams]:
        """应用内置风格预设，返回参数包"""
        preset = BUILTIN_PRESETS.get(preset_name)
        if preset and preset.expert_params:
            self.current_params = preset.expert_params
            return preset.expert_params
        return None

    def get_ref_features_dict(self) -> Optional[dict]:
        """获取参考图特征 (字典格式)"""
        if self.ref_features:
            return self.ref_features.to_dict()
        return None

    def get_builtin_preset_names(self) -> List[str]:
        """获取内置预设名称列表"""
        return list(BUILTIN_PRESETS.keys())
