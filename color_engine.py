"""
color_engine.py — 核心追色引擎（无 cv2 依赖版）
功能：
  1. Lab 颜色迁移（Reinhard 方法）
  2. 肤色保护遮罩（YCrCb 肤色检测）
  3. 影调强度 / 色调强度独立控制
  4. 3D LUT 生成（输出 .cube 文件）

依赖：PIL, numpy, scipy
"""

from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter, binary_closing, binary_opening, binary_dilation


# ═══════════════════════════════════════════════════════
#  ICC 标准色彩转换矩阵（sRGB ↔ XYZ ↔ Lab）
# ═══════════════════════════════════════════════════════

# sRGB → XYZ (D65 白点, 2.4 gamma)
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

# D65 白点 XYZ
_XYZ_W = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)


def _gamma_correct(c: np.ndarray) -> np.ndarray:
    """线性化 sRGB gamma 曲线"""
    out = np.where(c > 0.0031308,
                    1.055 * np.power(np.maximum(c, 1e-8), 1.0 / 2.4) - 0.055,
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
    xyz_n = xyz / xyz_w  # 归一化到 D65

    eps = 1e-6
    f = np.where(xyz_n > eps**3,
                 np.power(xyz_n, 1.0 / 3.0),
                 (841.0 / 108.0) * xyz_n + 4.0 / 29.0)

    L = 116.0 * f[..., 1] - 16.0           # L*: 0~100
    a = 500.0 * (f[..., 0] - f[..., 1])   # a*: -128~127 范围近似
    b = 200.0 * (f[..., 1] - f[..., 2])   # b*: -128~127 范围近似

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


# ═══════════════════════════════════════════════════════
#  基础图像 I/O
# ═══════════════════════════════════════════════════════

def load_image(path: str) -> np.ndarray:
    """读取图片 → float32 RGB [0,1]"""
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def save_image(img: np.ndarray, path: str):
    """保存 float32 RGB [0,1]"""
    out = np.clip(img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(out, mode='RGB').save(path)


# ═══════════════════════════════════════════════════════
#  色彩空间转换
# ═══════════════════════════════════════════════════════

def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """
    float32 RGB [0,1] → Lab (L:0~100, a/b: ±128 范围近似)
    """
    rgb_lin = _inv_gamma(img)                              # [H,W,3]
    xyz = np.tensordot(rgb_lin, _SRGB_TO_XYZ.T, axes=[[2], [0]])  # [H,W,3]
    return _xyz_to_lab(xyz)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Lab → float32 RGB [0,1]"""
    xyz = _lab_to_xyz(lab)                                 # [H,W,3]
    rgb_lin = np.tensordot(xyz, _XYZ_TO_SRGB.T, axes=[[2], [0]])  # [H,W,3]
    return _gamma_correct(rgb_lin).astype(np.float32)


# ═══════════════════════════════════════════════════════
#  肤色保护遮罩
# ═══════════════════════════════════════════════════════

def _rgb_to_ycrcb(img_u8: np.ndarray) -> np.ndarray:
    """uint8 RGB → YCrCb（numpy 实现，与 cv2 结果一致）"""
    r, g, b = img_u8[..., 0], img_u8[..., 1], img_u8[..., 2]
    Y  =  0.299 * r + 0.587 * g + 0.114 * b
    Cr = (r - Y) * 0.713 + 128
    Cb = (b - Y) * 0.564 + 128
    return np.stack([Y, Cr, Cb], axis=-1).astype(np.uint8)


def _rgb_to_hsv(img_u8: np.ndarray) -> np.ndarray:
    """uint8 RGB → HSV（numpy 实现）"""
    r, g, b = img_u8[..., 0] / 255.0, img_u8[..., 1] / 255.0, img_u8[..., 2] / 255.0
    v = np.maximum(np.maximum(r, g), b)
    c = v - np.minimum(np.minimum(r, g), b)
    s = np.where(v != 0, c / v, 0.0)
    h = np.zeros_like(v)

    mask_c = c != 0
    idx = mask_c & (v == r)
    h[idx] = ((g[idx] - b[idx]) / c[idx]) % 6
    idx = mask_c & (v == g)
    h[idx] = (b[idx] - r[idx]) / c[idx] + 2
    idx = mask_c & (v == b)
    h[idx] = (r[idx] - g[idx]) / c[idx] + 4
    h = (h / 6.0) % 1.0

    return np.stack([h * 179, s * 255, v * 255], axis=-1).astype(np.uint8)


def skin_mask(img_rgb: np.ndarray, blur_radius: int = 15) -> np.ndarray:
    """
    生成肤色保护遮罩 (float32, 0=非肤色/完全追色, 1=肤色/完全保护)
    YCrCb + HSV 双空间检测，高斯羽化
    """
    u8 = np.clip(img_rgb * 255, 0, 255).astype(np.uint8)

    # YCrCb 肤色检测
    ycrcb = _rgb_to_ycrcb(u8)
    mask_ycrcb = ((ycrcb[..., 1] >= 133) & (ycrcb[..., 1] <= 173) &
                  (ycrcb[..., 2] >= 77)  & (ycrcb[..., 2] <= 127))
    mask_ycrcb = mask_ycrcb.astype(np.uint8) * 255

    # HSV 辅助过滤
    hsv = _rgb_to_hsv(u8)
    mask_hsv = ((hsv[..., 0] <= 25) &
                (hsv[..., 1] >= 15) & (hsv[..., 1] <= 200) &
                (hsv[..., 2] >= 40))
    mask_hsv = mask_hsv.astype(np.uint8) * 255

    # 合并（AND）
    mask = (mask_ycrcb.astype(bool) & mask_hsv.astype(bool)).astype(np.uint8) * 255

    # 形态学处理（填补空洞 + 去除噪点）
    struct_9 = np.ones((9, 9), dtype=bool)
    mask = binary_closing(mask, structure=struct_9, iterations=2).astype(np.uint8) * 255
    mask = binary_opening(mask, structure=struct_9, iterations=1).astype(np.uint8) * 255
    mask = binary_dilation(mask, iterations=2).astype(np.uint8) * 255

    # 高斯羽化
    mask_f = gaussian_filter(mask.astype(np.float32) / 255.0, sigma=blur_radius)
    return mask_f.clip(0.0, 1.0)


# ═══════════════════════════════════════════════════════
#  核心追色引擎 — Lab Reinhard
# ═══════════════════════════════════════════════════════

class ColorTransfer:
    """
    基于 Lab 空间的 Reinhard 颜色迁移
    支持：影调强度、色调强度独立控制，肤色保护
    """

    def __init__(self):
        self.ref_stats = None

    def analyze(self, ref_img: np.ndarray):
        """分析参考图，提取 Lab 统计特征"""
        lab = rgb_to_lab(ref_img)
        L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        self.ref_stats = {
            'mean_L': float(L.mean()),  'std_L': float(L.std()),
            'mean_a': float(a.mean()),  'std_a': float(a.std()),
            'mean_b': float(b.mean()),  'std_b': float(b.std()),
        }
        return self.ref_stats

    def transfer(
        self,
        src_img: np.ndarray,
        tone_strength: float = 1.0,
        color_strength: float = 1.0,
        skin_protect: float = 0.85,
        use_skin_protect: bool = True,
    ) -> np.ndarray:
        """执行追色，返回 float32 RGB [0,1]"""
        if self.ref_stats is None:
            raise ValueError("请先调用 analyze() 分析参考图")

        # 限制强度范围，防止数值溢出
        tone_strength = np.clip(tone_strength, 0.0, 2.0)
        color_strength = np.clip(color_strength, 0.0, 2.0)

        s = self.ref_stats
        src_lab = rgb_to_lab(src_img)
        result_lab = src_lab.copy()

        src_L = src_lab[:, :, 0]
        src_a = src_lab[:, :, 1]
        src_b = src_lab[:, :, 2]
        src_mean_L = float(src_L.mean())
        src_std_L  = float(src_L.std())
        src_mean_a = float(src_a.mean())
        src_std_a  = float(src_a.std())
        src_mean_b = float(src_b.mean())
        src_std_b  = float(src_b.std())

        eps = 1e-6

        # Reinhard: (x - src_mean) * (ref_std / src_std) + ref_mean
        transferred_L = (src_L - src_mean_L) * (s['std_L'] / (src_std_L + eps)) + s['mean_L']
        result_lab[:, :, 0] = src_L + tone_strength * (transferred_L - src_L)

        transferred_a = (src_a - src_mean_a) * (s['std_a'] / (src_std_a + eps)) + s['mean_a']
        result_lab[:, :, 1] = src_a + color_strength * (transferred_a - src_a)

        transferred_b = (src_b - src_mean_b) * (s['std_b'] / (src_std_b + eps)) + s['mean_b']
        result_lab[:, :, 2] = src_b + color_strength * (transferred_b - src_b)

        # 严格限制Lab值范围
        result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 100)
        result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], -128, 127)
        result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], -128, 127)

        result_rgb = lab_to_rgb(result_lab)

        # 处理NaN/Inf
        result_rgb = np.nan_to_num(result_rgb, nan=0.5, posinf=1.0, neginf=0.0)
        result_rgb = np.clip(result_rgb, 0.0, 1.0)

        if use_skin_protect and skin_protect > 0:
            mask = skin_mask(src_img)
            protect = mask[:, :, np.newaxis] * skin_protect
            result_rgb = result_rgb * (1.0 - protect) + src_img * protect

        return np.clip(result_rgb, 0.0, 1.0)


# ═══════════════════════════════════════════════════════
#  3D LUT 生成器
# ═══════════════════════════════════════════════════════

class LUTGenerator:
    """生成 .cube 格式 3D LUT，可导入 PR / AE / 达芬奇"""

    def __init__(self, lut_size: int = 33):
        self.lut_size = lut_size

    def generate(
        self,
        transfer: ColorTransfer,
        tone_strength: float = 1.0,
        color_strength: float = 1.0,
        output_path: str = "output.cube",
        title: str = "ColorMatch LUT",
    ):
        """生成 .cube 文件"""
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
            f.write(f"TITLE \"{title}\"\n")
            f.write(f"LUT_3D_SIZE {n}\n")
            f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
            f.write("DOMAIN_MAX 1.0 1.0 1.0\n\n")
            for ri in range(n):
                for gi in range(n):
                    for bi in range(n):
                        rv, gv, bv = mapped[ri, gi, bi]
                        f.write(f"{rv:.6f} {gv:.6f} {bv:.6f}\n")

        return output_path
