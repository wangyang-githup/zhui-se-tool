"""
color_engine.py — 核心追色引擎
功能：
  1. Lab 颜色迁移（Reinhard 方法）
  2. 肤色保护遮罩（YCrCb 肤色检测）
  3. 影调强度 / 色调强度独立控制
  4. 3D LUT 生成（输出 .cube 文件）
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


# ──────────────────────────────────────────────
#  工具函数
# ──────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """读取图片，转为 float32 RGB [0,1]"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def save_image(img: np.ndarray, path: str):
    """保存 float32 RGB [0,1] 图片"""
    out = np.clip(img * 255, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, out_bgr)


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """float32 RGB [0,1] → Lab"""
    u8 = (img * 255).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    # OpenCV Lab: L∈[0,255] a,b∈[0,255] (偏移了128)
    lab[:, :, 0] = lab[:, :, 0] / 255.0 * 100.0          # L: 0-100
    lab[:, :, 1] = lab[:, :, 1] - 128.0                   # a: -128~127
    lab[:, :, 2] = lab[:, :, 2] - 128.0                   # b: -128~127
    return lab


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Lab → float32 RGB [0,1]"""
    tmp = lab.copy()
    tmp[:, :, 0] = (tmp[:, :, 0] / 100.0 * 255.0).clip(0, 255)
    tmp[:, :, 1] = (tmp[:, :, 1] + 128.0).clip(0, 255)
    tmp[:, :, 2] = (tmp[:, :, 2] + 128.0).clip(0, 255)
    tmp = tmp.astype(np.uint8)
    rgb = cv2.cvtColor(tmp, cv2.COLOR_LAB2RGB)
    return rgb.astype(np.float32) / 255.0


# ──────────────────────────────────────────────
#  肤色保护遮罩
# ──────────────────────────────────────────────

def skin_mask(img_rgb: np.ndarray, blur_radius: int = 15) -> np.ndarray:
    """
    生成肤色保护遮罩 (float32, 0=非肤色/完全追色, 1=肤色/完全保护)
    使用 YCrCb 空间检测肤色，结合 HSV 辅助过滤
    """
    u8 = (img_rgb * 255).clip(0, 255).astype(np.uint8)

    # YCrCb 肤色范围（人种通用范围）
    ycrcb = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
    mask_ycrcb = cv2.inRange(ycrcb,
                             np.array([0,  133, 77], dtype=np.uint8),
                             np.array([255, 173, 127], dtype=np.uint8))

    # HSV 辅助过滤（排除过饱和的非皮肤区域）
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
    mask_hsv = cv2.inRange(hsv,
                           np.array([0, 15, 40], dtype=np.uint8),
                           np.array([25, 200, 255], dtype=np.uint8))

    # 合并遮罩
    mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

    # 形态学处理：填补空洞、平滑边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 高斯模糊羽化边缘，让过渡自然
    mask_f = mask.astype(np.float32) / 255.0
    mask_f = gaussian_filter(mask_f, sigma=blur_radius)
    return mask_f.clip(0, 1)


# ──────────────────────────────────────────────
#  核心追色引擎 — Lab Reinhard 方法
# ──────────────────────────────────────────────

class ColorTransfer:
    """
    基于 Lab 空间的 Reinhard 颜色迁移
    支持：影调强度、色调强度独立控制，肤色保护
    """

    def __init__(self):
        self.ref_stats = None   # (mean_L, std_L, mean_a, std_a, mean_b, std_b)

    def analyze(self, ref_img: np.ndarray):
        """分析参考图，提取 Lab 统计特征"""
        lab = rgb_to_lab(ref_img)
        L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        self.ref_stats = {
            'mean_L': L.mean(), 'std_L': L.std(),
            'mean_a': a.mean(), 'std_a': a.std(),
            'mean_b': b.mean(), 'std_b': b.std(),
        }
        return self.ref_stats

    def transfer(
        self,
        src_img: np.ndarray,
        tone_strength: float = 1.0,    # 影调强度 0~1（L通道）
        color_strength: float = 1.0,   # 色调强度 0~1（a,b通道）
        skin_protect: float = 0.85,    # 肤色保护强度 0~1（1=完全保护）
        use_skin_protect: bool = True,
    ) -> np.ndarray:
        """
        执行追色

        参数:
            src_img       : float32 RGB [0,1] 原图
            tone_strength : 影调匹配强度 (0=不调影调, 1=完全匹配)
            color_strength: 色调匹配强度 (0=不调色调, 1=完全匹配)
            skin_protect  : 肤色保护强度 (0=不保护, 1=完全保护)
            use_skin_protect: 是否启用肤色保护

        返回:
            float32 RGB [0,1] 结果图
        """
        if self.ref_stats is None:
            raise ValueError("请先调用 analyze() 分析参考图")

        s = self.ref_stats
        src_lab = rgb_to_lab(src_img)
        result_lab = src_lab.copy()

        src_L = src_lab[:, :, 0]
        src_a = src_lab[:, :, 1]
        src_b = src_lab[:, :, 2]

        src_mean_L, src_std_L = src_L.mean(), src_L.std()
        src_mean_a, src_std_a = src_a.mean(), src_a.std()
        src_mean_b, src_std_b = src_b.mean(), src_b.std()

        # Reinhard 公式: (x - src_mean) * (ref_std / src_std) + ref_mean
        # 用强度参数做线性插值: result = src + strength * (transfer - src)

        eps = 1e-6  # 防止除零

        # L 通道（影调）
        transferred_L = (src_L - src_mean_L) * (s['std_L'] / (src_std_L + eps)) + s['mean_L']
        result_lab[:, :, 0] = src_L + tone_strength * (transferred_L - src_L)

        # a 通道（绿-红）
        transferred_a = (src_a - src_mean_a) * (s['std_a'] / (src_std_a + eps)) + s['mean_a']
        result_lab[:, :, 1] = src_a + color_strength * (transferred_a - src_a)

        # b 通道（蓝-黄）
        transferred_b = (src_b - src_mean_b) * (s['std_b'] / (src_std_b + eps)) + s['mean_b']
        result_lab[:, :, 2] = src_b + color_strength * (transferred_b - src_b)

        # 裁剪到合法范围
        result_lab[:, :, 0] = result_lab[:, :, 0].clip(0, 100)
        result_lab[:, :, 1] = result_lab[:, :, 1].clip(-128, 127)
        result_lab[:, :, 2] = result_lab[:, :, 2].clip(-128, 127)

        result_rgb = lab_to_rgb(result_lab)

        # 肤色保护：在结果图和原图之间，肤色区域按比例混回原图
        if use_skin_protect and skin_protect > 0:
            mask = skin_mask(src_img)           # [H,W] float32
            mask_3ch = mask[:, :, np.newaxis]   # [H,W,1] 广播
            protect = mask_3ch * skin_protect
            result_rgb = result_rgb * (1 - protect) + src_img * protect

        return result_rgb.clip(0, 1)


# ──────────────────────────────────────────────
#  3D LUT 生成器
# ──────────────────────────────────────────────

class LUTGenerator:
    """
    基于颜色迁移结果，生成标准 .cube 3D LUT 文件
    可直接导入 PR / AE / 达芬奇 / Lightroom
    """

    def __init__(self, lut_size: int = 33):
        """
        lut_size: LUT 格点数量（17=轻量, 33=标准, 65=高精度）
        """
        self.lut_size = lut_size

    def generate(
        self,
        transfer: ColorTransfer,
        tone_strength: float = 1.0,
        color_strength: float = 1.0,
        output_path: str = "output.cube",
        title: str = "ColorMatch LUT",
    ):
        """
        生成 .cube 文件

        原理：构造一张包含所有颜色组合的"标准色卡"图，
        经过追色引擎处理后，对比前后颜色得到映射关系，
        写成 .cube 格式。注意：LUT 不含肤色保护（LUT 是全局映射）。
        """
        n = self.lut_size
        # 构造 identity LUT 色卡：所有 (R,G,B) 组合
        lin = np.linspace(0, 1, n, dtype=np.float32)
        r, g, b = np.meshgrid(lin, lin, lin, indexing='ij')
        # shape: [n,n,n,3]，重排为图片形状 [n*n, n, 3]
        identity = np.stack([r, g, b], axis=-1).reshape(n * n, n, 3)

        # 通过引擎追色（不做肤色保护，LUT 是全局的）
        mapped = transfer.transfer(
            identity,
            tone_strength=tone_strength,
            color_strength=color_strength,
            use_skin_protect=False,
        ).reshape(n, n, n, 3)

        # 写 .cube 文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"TITLE \"{title}\"\n")
            f.write(f"LUT_3D_SIZE {n}\n")
            f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
            f.write("DOMAIN_MAX 1.0 1.0 1.0\n\n")
            # .cube 顺序：B 最快变，R 最慢变
            for ri in range(n):
                for gi in range(n):
                    for bi in range(n):
                        rv, gv, bv = mapped[ri, gi, bi]
                        f.write(f"{rv:.6f} {gv:.6f} {bv:.6f}\n")

        return output_path
