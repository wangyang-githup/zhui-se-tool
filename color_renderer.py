"""
追色工具 - 肤色保护与融合器
==========================================
核心规则：
  1. 寻找参考图肤色锚点（高光/阴影 Lab）
  2. 追踪追色后肤色的 ΔE 偏移
  3. 如 ΔE > 阈值(默认5)，自动回滚肤色区色彩调整
  4. 环境色可以激进，肤色调整必须保守

无 cv2 依赖，纯 PIL + NumPy + scipy 实现
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from feature_extractor import ColorFeatures, detect_skin_mask
from grading_generator import ColorGradingParams, CurvePoint


# ─────────────────────────────────────────────────────
#  ICC 标准色彩转换矩阵
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
    out = np.where(c > 0.0031308,
                   1.055 * np.power(np.maximum(c, 1e-8), 1.0 / 2.4) - 0.055,
                   12.92 * c)
    return np.clip(out, 0.0, 1.0)


def _inv_gamma(c: np.ndarray) -> np.ndarray:
    out = np.where(c > 0.04045,
                   np.power((c + 0.055) / 1.055, 2.4),
                   c / 12.92)
    return np.clip(out, 0.0, 1.0)


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    xyz_w = _XYZ_W
    xyz_n = xyz / xyz_w
    eps = 1e-6
    f = np.where(xyz_n > eps**3,
                 np.power(xyz_n, 1.0 / 3.0),
                 (841.0 / 108.0) * xyz_n + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def _lab_to_xyz(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    eps = 6.0 / 29.0
    xyz_n = np.where(fy > eps, fy ** 3.0, (fy - 16.0 / 116.0) * (108.0 / 841.0))
    xr = np.where(fx > eps, fx ** 3.0, (fx - 4.0 / 29.0) * (108.0 / 841.0))
    zr = np.where(fz > eps, fz ** 3.0, (fz - 4.0 / 29.0) * (108.0 / 841.0))
    xyz = np.stack([xr, xyz_n, zr], axis=-1) * _XYZ_W
    return xyz.astype(np.float32)


# ─────────────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────────────

def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE76 ΔE 计算"""
    diff = lab1.astype(float) - lab2.astype(float)
    return np.sqrt((diff ** 2).sum(axis=-1))


def rgb_to_lab_std(img: np.ndarray) -> np.ndarray:
    """float32 RGB [0,1] → 标准 Lab"""
    rgb_lin = _inv_gamma(img)
    xyz = np.tensordot(rgb_lin, _SRGB_TO_XYZ.T, axes=[[2], [0]])
    return _xyz_to_lab(xyz)


def lab_std_to_rgb(lab: np.ndarray) -> np.ndarray:
    """标准 Lab → float32 RGB [0,1]"""
    xyz = _lab_to_xyz(lab)
    rgb_lin = np.tensordot(xyz, _XYZ_TO_SRGB.T, axes=[[2], [0]])
    return _gamma_correct(rgb_lin).astype(np.float32)


def interp_curve(img_channel: np.ndarray, curve_pts: list) -> np.ndarray:
    """将曲线锚点列表应用到单通道图像"""
    if not curve_pts:
        return img_channel

    xs = np.array([p.input  for p in curve_pts], dtype=float)
    ys = np.array([p.output for p in curve_pts], dtype=float)

    if xs[0] > 0:
        xs = np.insert(xs, 0, 0);  ys = np.insert(ys, 0, 0)
    if xs[-1] < 255:
        xs = np.append(xs, 255);   ys = np.append(ys, 255)

    src_255 = (img_channel * 255).clip(0, 255)
    out_255 = np.interp(src_255, xs, ys)
    return (out_255 / 255.0).clip(0, 1).astype(np.float32)


# ─────────────────────────────────────────────────────
#  主渲染器
# ─────────────────────────────────────────────────────

class ColorRenderer:
    """
    将追色参数应用到原图，并执行肤色保护
    """

    SKIN_DE_THRESHOLD = 5.0
    SKIN_BLUR_SIGMA   = 20

    def render(
        self,
        src: np.ndarray,
        params: ColorGradingParams,
        ref_features: ColorFeatures,
    ) -> np.ndarray:
        """
        src:          float32 RGB [0,1]
        params:       ColorGradingParams
        ref_features: 参考图特征包

        返回 float32 RGB [0,1] 结果图
        """
        # Step A: 全局追色
        result_full = self._apply_grading(src, params)

        # Step B: 肤色蒙版
        skin_mask_u8 = detect_skin_mask(src)
        skin_bool = skin_mask_u8 > 0

        if not skin_bool.any():
            return result_full.clip(0, 1)

        # 羽化蒙版
        mask_f = skin_mask_u8.astype(np.float32) / 255.0
        mask_f = gaussian_filter(mask_f, sigma=self.SKIN_BLUR_SIGMA)
        mask_f = mask_f.clip(0, 1)
        mask_3ch = mask_f[:, :, np.newaxis]

        # Step C: ΔE 检测
        src_lab    = rgb_to_lab_std(src)
        result_lab = rgb_to_lab_std(result_full)

        de_map = delta_e_cie76(src_lab, result_lab)
        skin_de_mean = float(de_map[skin_bool].mean())

        # Step D: 肤色区域处理
        if skin_de_mean <= self.SKIN_DE_THRESHOLD:
            skin_result = result_full * 0.7 + src * 0.3
        else:
            skin_result = self._skin_conservative_adjust(src, params, ref_features)

        final = result_full * (1 - mask_3ch) + skin_result * mask_3ch
        return final.clip(0, 1)

    def _apply_grading(self, src: np.ndarray, p: ColorGradingParams) -> np.ndarray:
        """将 ColorGradingParams 应用到图像"""
        img = src.copy().astype(np.float32)

        # 曝光
        img = img * (2 ** p.tone.exposure)

        # 对比度
        if p.tone.contrast != 0:
            c = p.tone.contrast / 100.0
            img = (img - 0.5) * (1 + c) + 0.5

        # 高光/阴影/白色/黑色
        img = self._apply_tone_controls(img, p.tone)

        # RGB 曲线
        if p.curve.rgb:
            lum = img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114
            lum_new = interp_curve(lum, p.curve.rgb)
            scale = np.where(lum > 1e-4, lum_new / (lum + 1e-4), 1.0)
            scale = scale[:, :, np.newaxis]
            img = img * scale

        if p.curve.red:
            img[:,:,0] = interp_curve(img[:,:,0], p.curve.red)
        if p.curve.green:
            img[:,:,1] = interp_curve(img[:,:,1], p.curve.green)
        if p.curve.blue:
            img[:,:,2] = interp_curve(img[:,:,2], p.curve.blue)

        # 分离色调
        img = self._apply_split_tone(img, p.split_tone)

        return img.clip(0, 1)

    def _apply_tone_controls(self, img: np.ndarray, tone) -> np.ndarray:
        """高光/阴影/白色/黑色调整"""
        lum = img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114

        hi_mask = np.clip((lum - 0.5) * 2, 0, 1)[:,:,np.newaxis]
        lo_mask = np.clip((0.5 - lum) * 2, 0, 1)[:,:,np.newaxis]

        hi_adj = tone.highlights / 500.0
        img = img + hi_adj * hi_mask * (1 - img)

        sh_adj = tone.shadows / 500.0
        img = img + sh_adj * lo_mask * img

        wh_adj = tone.whites / 500.0
        wh_mask = np.clip((lum - 0.8) * 5, 0, 1)[:,:,np.newaxis]
        img = img + wh_adj * wh_mask

        bl_adj = tone.blacks / 500.0
        bl_mask = np.clip((0.2 - lum) * 5, 0, 1)[:,:,np.newaxis]
        img = img + bl_adj * bl_mask

        return img

    def _apply_split_tone(self, img: np.ndarray, st) -> np.ndarray:
        """分离色调"""
        if st.highlight_sat < 1 and st.shadow_sat < 1:
            return img

        lab = rgb_to_lab_std(img)
        L = lab[:,:,0]

        hi_mask = np.clip((L - 60) / 40, 0, 1)[:,:,np.newaxis]
        sh_mask = np.clip((40 - L) / 40, 0, 1)[:,:,np.newaxis]

        hi_hue_rad = np.radians(st.highlight_hue)
        sh_hue_rad = np.radians(st.shadow_hue)
        hi_sat_norm = st.highlight_sat / 100.0 * 20
        sh_sat_norm = st.shadow_sat  / 100.0 * 15

        hi_da = np.cos(hi_hue_rad) * hi_sat_norm
        hi_db = np.sin(hi_hue_rad) * hi_sat_norm
        sh_da = np.cos(sh_hue_rad) * sh_sat_norm
        sh_db = np.sin(sh_hue_rad) * sh_sat_norm

        lab[:,:,1] += float(hi_da) * hi_mask[:,:,0] + float(sh_da) * sh_mask[:,:,0]
        lab[:,:,2] += float(hi_db) * hi_mask[:,:,0] + float(sh_db) * sh_mask[:,:,0]

        lab[:,:,0] = lab[:,:,0].clip(0, 100)
        lab[:,:,1] = lab[:,:,1].clip(-128, 127)
        lab[:,:,2] = lab[:,:,2].clip(-128, 127)

        return lab_std_to_rgb(lab)

    def _skin_conservative_adjust(
        self,
        src: np.ndarray,
        params: ColorGradingParams,
        ref_features: ColorFeatures,
    ) -> np.ndarray:
        """肤色区保守处理（ΔE 超标时）"""
        lab = rgb_to_lab_std(src)

        # 影调：仅应用一半曝光
        ev_scale = 2 ** (params.tone.exposure * 0.5)
        lab[:,:,0] = (lab[:,:,0] * ev_scale).clip(0, 100)

        # 轻微提亮
        lab[:,:,0] = (lab[:,:,0] + 1.5).clip(0, 100)

        # 保暖：+a*1, +b*2
        lab[:,:,1] = (lab[:,:,1] + 1.0).clip(-128, 127)
        lab[:,:,2] = (lab[:,:,2] + 2.0).clip(-128, 127)

        return lab_std_to_rgb(lab)
