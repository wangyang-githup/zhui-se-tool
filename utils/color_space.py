"""
utils/color_space.py — 统一色彩空间转换 + 肤色检测
=====================================================
纯 NumPy 实现，无 OpenCV 依赖。
从 color_engine.py v2.0 提取，供所有模块共享。
"""

import numpy as np
from scipy.ndimage import gaussian_filter, binary_closing, binary_opening, binary_dilation


# ════════════════════════════════════════════════════
#  常量
# ════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════
#  sRGB Gamma 编解码
# ════════════════════════════════════════════════════

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (np.clip(c, 1e-10, None) ** (1.0 / 2.4)) - 0.055)


# ════════════════════════════════════════════════════
#  RGB ↔ Lab  (纯 NumPy)
# ════════════════════════════════════════════════════

def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """
    float32 RGB [0,1] → CIE Lab
    L ∈ [0, 100], a ∈ [-128, 127], b ∈ [-128, 127]
    """
    linear = _srgb_to_linear(img.astype(np.float64))
    shape = linear.shape
    flat = linear.reshape(-1, 3)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        xyz = flat @ _M_RGB2XYZ.T
    xyz = xyz.reshape(shape)

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
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        linear = flat @ _M_XYZ2RGB.T
    linear = linear.reshape(shape).clip(0, 1)

    srgb = _linear_to_srgb(linear)
    return srgb.clip(0, 1).astype(np.float32)


# ════════════════════════════════════════════════════
#  RGB → HSV  (纯 NumPy)
# ════════════════════════════════════════════════════

def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """
    float32 RGB [0,1] → HSV
    H ∈ [0, 360), S ∈ [0, 1], V ∈ [0, 1]
    """
    r, g, b = img[:, :, 0].astype(np.float64), img[:, :, 1].astype(np.float64), img[:, :, 2].astype(np.float64)
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(r)
    mask_r = (cmax == r) & (delta > 1e-10)
    mask_g = (cmax == g) & (delta > 1e-10) & ~mask_r
    mask_b = (cmax == b) & (delta > 1e-10) & ~mask_r & ~mask_g

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
    h = h % 360.0

    with np.errstate(invalid='ignore', divide='ignore'):
        s = np.where(cmax > 1e-10, delta / cmax, 0.0)

    v = cmax
    return np.stack([h, s, v], axis=-1).astype(np.float32)


# ════════════════════════════════════════════════════
#  肤色检测 (HSV + Lab 双空间)
# ════════════════════════════════════════════════════

def detect_skin_mask(img_rgb: np.ndarray, blur_radius: int = 15) -> np.ndarray:
    """
    生成肤色保护遮罩 (float32, 0=非肤色, 1=肤色)

    使用 HSV + Lab 双空间检测：
      - HSV: 肤色色相范围 0°-50°，低饱和中亮度
      - Lab: 亚洲人肤色范围 L∈[30,90], a∈[2,18], b∈[8,28]
    """
    u8 = (img_rgb.clip(0, 1) * 255).astype(np.uint8).astype(np.float32) / 255.0

    hsv = rgb_to_hsv(u8)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask_hsv = (
        (h >= 0) & (h <= 50) &
        (s >= 0.08) & (s <= 0.75) &
        (v >= 0.15) & (v <= 1.0)
    )

    lab = rgb_to_lab(u8)
    L, a_ch, b_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    mask_lab = (
        (L >= 30) & (L <= 90) &
        (a_ch >= 2) & (a_ch <= 18) &
        (b_ch >= 8) & (b_ch <= 28)
    )

    mask = mask_hsv | mask_lab

    mask = binary_closing(mask, iterations=2)
    mask = binary_opening(mask, iterations=1)
    mask = binary_dilation(mask, iterations=2)

    mask_f = mask.astype(np.float32)
    mask_f = gaussian_filter(mask_f, sigma=blur_radius)
    return mask_f.clip(0, 1)
