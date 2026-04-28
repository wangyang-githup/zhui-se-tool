"""
模块三：肤色保护与融合器
==========================================
核心规则（强制优先级）：
  1. 寻找参考图肤色锚点（高光/阴影 Lab）
  2. 追踪追色后肤色的 ΔE 偏移
  3. 如 ΔE > 阈值(默认5)，自动回滚肤色区色彩调整，改为轻微提亮加暖
  4. 环境色可以激进，肤色调整必须保守

输入：原图(float32 RGB), 追色参数包, 参考图特征包
输出：追色后预览图(float32 RGB)
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from feature_extractor import ColorFeatures, detect_skin_mask
from grading_generator import ColorGradingParams, CurvePoint


# ─────────────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────────────

def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    CIE76 ΔE 计算
    lab1, lab2: (..., 3) 标准 Lab（L∈0-100, a/b∈-128~127）
    返回 ΔE 数组（每像素）
    """
    diff = lab1.astype(float) - lab2.astype(float)
    return np.sqrt((diff ** 2).sum(axis=-1))


def rgb_to_lab_std(img: np.ndarray) -> np.ndarray:
    """float32 RGB [0,1] → 标准 Lab"""
    u8 = (img * 255).clip(0, 255).astype(np.uint8)
    lab_cv = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_cv[:, :, 0] = lab_cv[:, :, 0] / 255.0 * 100.0
    lab_cv[:, :, 1] -= 128.0
    lab_cv[:, :, 2] -= 128.0
    return lab_cv


def lab_std_to_rgb(lab: np.ndarray) -> np.ndarray:
    """标准 Lab → float32 RGB [0,1]"""
    tmp = lab.copy()
    tmp[:, :, 0] = (tmp[:, :, 0] / 100.0 * 255.0).clip(0, 255)
    tmp[:, :, 1] = (tmp[:, :, 1] + 128.0).clip(0, 255)
    tmp[:, :, 2] = (tmp[:, :, 2] + 128.0).clip(0, 255)
    u8 = tmp.astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def interp_curve(img_channel: np.ndarray, curve_pts: list) -> np.ndarray:
    """
    将曲线锚点列表插值应用到单通道图像（0-1 float）

    curve_pts: List[CurvePoint]，input/output 均为 0-255
    """
    if not curve_pts:
        return img_channel

    xs = np.array([p.input  for p in curve_pts], dtype=float)
    ys = np.array([p.output for p in curve_pts], dtype=float)

    # 确保端点存在
    if xs[0] > 0:
        xs = np.insert(xs, 0, 0);  ys = np.insert(ys, 0, 0)
    if xs[-1] < 255:
        xs = np.append(xs, 255);   ys = np.append(ys, 255)

    # 将图像映射到 0-255，插值后归回 0-1
    src_255 = (img_channel * 255).clip(0, 255)
    out_255 = np.interp(src_255, xs, ys)
    return (out_255 / 255.0).clip(0, 1).astype(np.float32)


# ─────────────────────────────────────────────────────
#  主渲染器
# ─────────────────────────────────────────────────────

class ColorRenderer:
    """
    将追色参数应用到原图，并执行肤色保护

    主流程：
      A. 全局应用追色参数 → 得到 result_full
      B. 生成肤色蒙版
      C. 检查 ΔE，按阈值决定肤色区是否回滚
      D. 混合环境色（result_full）与肤色区（回滚/保守结果）
    """

    SKIN_DE_THRESHOLD = 5.0    # 肤色 ΔE 容差（CIE76）
    SKIN_BLUR_SIGMA   = 20     # 蒙版羽化半径（px）

    def render(
        self,
        src: np.ndarray,
        params: ColorGradingParams,
        ref_features: ColorFeatures,
    ) -> np.ndarray:
        """
        src:          float32 RGB [0,1]
        params:       ColorGradingParams
        ref_features: 参考图特征包（用于肤色锚点）

        返回 float32 RGB [0,1] 结果图
        """

        # ── Step A：全局追色 ──────────────────────────
        result_full = self._apply_grading(src, params)

        # ── Step B：生成肤色蒙版 ──────────────────────
        skin_mask_u8 = detect_skin_mask(src)
        skin_bool = skin_mask_u8 > 0

        if not skin_bool.any():
            # 无肤色区域，直接返回全局追色结果
            return result_full.clip(0, 1)

        # 羽化蒙版（高斯模糊）
        mask_f = skin_mask_u8.astype(np.float32) / 255.0
        mask_f = gaussian_filter(mask_f, sigma=self.SKIN_BLUR_SIGMA)
        mask_f = mask_f.clip(0, 1)
        mask_3ch = mask_f[:, :, np.newaxis]

        # ── Step C：ΔE 检测 ──────────────────────────
        src_lab    = rgb_to_lab_std(src)
        result_lab = rgb_to_lab_std(result_full)

        # 仅在肤色区域计算 ΔE
        de_map = delta_e_cie76(src_lab, result_lab)  # (H,W)
        skin_de_mean = float(de_map[skin_bool].mean())

        # ── Step D：肤色区域处理 ──────────────────────
        if skin_de_mean <= self.SKIN_DE_THRESHOLD:
            # ΔE 可接受：直接融合，肤色区轻微保留原色
            skin_result = result_full * 0.7 + src * 0.3
        else:
            # ΔE 过大：回滚色彩，仅保留影调调整 + 轻微提亮加暖
            skin_result = self._skin_conservative_adjust(src, params, ref_features)

        # 最终混合：环境色用 result_full，肤色区用 skin_result
        final = result_full * (1 - mask_3ch) + skin_result * mask_3ch

        return final.clip(0, 1)

    # ─────────────────────────────────────────────────
    #  追色应用
    # ─────────────────────────────────────────────────

    def _apply_grading(self, src: np.ndarray, p: ColorGradingParams) -> np.ndarray:
        """
        将 ColorGradingParams 应用到图像

        应用顺序（模拟 Lightroom 处理管线）：
          1. 基础面板（曝光/对比度/高光/阴影）
          2. RGB 曲线
          3. Lab 色彩调整（HSL → Lab 空间操作）
          4. 分离色调
        """
        img = src.copy().astype(np.float32)

        # ── 1. 曝光 (EV) ──
        img = img * (2 ** p.tone.exposure)

        # ── 2. 对比度（以 0.5 为中点的 S 形调整）──
        if p.tone.contrast != 0:
            c = p.tone.contrast / 100.0   # 归一化到 -1~+1
            img = (img - 0.5) * (1 + c) + 0.5

        # ── 3. 高光/阴影/白色/黑色 ──
        img = self._apply_tone_controls(img, p.tone)

        # ── 4. RGB 曲线 ──
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

        # ── 5. Lab 色彩调整（分离色调）──
        img = self._apply_split_tone(img, p.split_tone)

        return img.clip(0, 1)

    def _apply_tone_controls(self, img: np.ndarray, tone) -> np.ndarray:
        """
        模拟 LR 高光/阴影/白色/黑色滑块

        原理：
          - 高光/白色：影响亮部（L > 0.7）
          - 阴影/黑色：影响暗部（L < 0.3）
          - 用亮度蒙版限定影响范围
        """
        lum = img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114

        # 高光蒙版（亮度>0.5，平滑过渡）
        hi_mask = np.clip((lum - 0.5) * 2, 0, 1)[:,:,np.newaxis]
        # 阴影蒙版（亮度<0.5，平滑过渡）
        lo_mask = np.clip((0.5 - lum) * 2, 0, 1)[:,:,np.newaxis]

        # 高光调整
        hi_adj = tone.highlights / 500.0
        img = img + hi_adj * hi_mask * (1 - img)  # 朝白收缩或扩展

        # 阴影调整
        sh_adj = tone.shadows / 500.0
        img = img + sh_adj * lo_mask * img         # 朝黑收缩或扩展

        # 白色（极亮区）
        wh_adj = tone.whites / 500.0
        wh_mask = np.clip((lum - 0.8) * 5, 0, 1)[:,:,np.newaxis]
        img = img + wh_adj * wh_mask

        # 黑色（极暗区）
        bl_adj = tone.blacks / 500.0
        bl_mask = np.clip((0.2 - lum) * 5, 0, 1)[:,:,np.newaxis]
        img = img + bl_adj * bl_mask

        return img

    def _apply_split_tone(self, img: np.ndarray, st) -> np.ndarray:
        """
        分离色调：高光加色 + 阴影加色

        用 Lab 空间在高光/阴影区叠加色调
        """
        if st.highlight_sat < 1 and st.shadow_sat < 1:
            return img

        lab = rgb_to_lab_std(img)
        L = lab[:,:,0]

        # 高光区（L > 60），阴影区（L < 40）
        hi_mask = np.clip((L - 60) / 40, 0, 1)[:,:,np.newaxis]
        sh_mask = np.clip((40 - L) / 40, 0, 1)[:,:,np.newaxis]

        # 色相 → Lab a/b
        hi_hue_rad = np.radians(st.highlight_hue)
        sh_hue_rad = np.radians(st.shadow_hue)
        hi_sat_norm = st.highlight_sat / 100.0 * 20  # 最大叠加 20 Lab 单位
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

    # ─────────────────────────────────────────────────
    #  肤色保守调整（ΔE 超标时触发）
    # ─────────────────────────────────────────────────

    def _skin_conservative_adjust(
        self,
        src: np.ndarray,
        params: ColorGradingParams,
        ref_features: ColorFeatures,
    ) -> np.ndarray:
        """
        肤色区专用保守处理：
          - 只保留影调调整（曝光/亮度）
          - 轻微提亮（+L*0.03）
          - 轻微加暖（+b*2，减a*1）→ 防青灰偏色
          - 完全跳过 HSL/色调/曲线色彩部分
        """
        lab = rgb_to_lab_std(src)

        # 影调：仅应用曝光（不做色彩）
        ev_scale = 2 ** (params.tone.exposure * 0.5)   # 肤色区只应用一半曝光
        lab[:,:,0] = (lab[:,:,0] * ev_scale).clip(0, 100)

        # 轻微提亮
        lab[:,:,0] = (lab[:,:,0] + 1.5).clip(0, 100)

        # 保暖：b* 轻微加黄（+2），a* 微正（+1）→ 防青灰
        lab[:,:,1] = (lab[:,:,1] + 1.0).clip(-128, 127)   # a* +1 (偏红)
        lab[:,:,2] = (lab[:,:,2] + 2.0).clip(-128, 127)   # b* +2 (偏黄/暖)

        return lab_std_to_rgb(lab)
