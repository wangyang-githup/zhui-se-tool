"""
模块二：差异分析与追色方案生成器
=========================================
输入：参考图 ColorFeatures、原图 ColorFeatures
输出：ColorGradingParams（类 Lightroom 参数包）+ 调整理由说明

核心算法逻辑：
  1. 影调差异 → 曝光/对比度/高光/阴影/白色/黑色 + RGB 曲线
  2. Lab a/b 差异 → HSL 色相/饱和度/明度 + 分离色调
  3. 色温倾向差异 → 白平衡校准参数
  4. 所有参数以相对值表示，附带理由字符串
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from feature_extractor import ColorFeatures, LabStats, ToneDistribution


# ─────────────────────────────────────────────────────
#  输出参数结构
# ─────────────────────────────────────────────────────

@dataclass
class ToneParams:
    """基础面板 —— 影调参数（相对值，Lightroom 风格）"""
    exposure:    float = 0.0   # 曝光  -5 ~ +5 EV
    contrast:    float = 0.0   # 对比度 -100 ~ +100
    highlights:  float = 0.0   # 高光  -100 ~ +100
    shadows:     float = 0.0   # 阴影  -100 ~ +100
    whites:      float = 0.0   # 白色  -100 ~ +100
    blacks:      float = 0.0   # 黑色  -100 ~ +100
    reason: str = ""

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class CurvePoint:
    input: float   # 0-255
    output: float  # 0-255


@dataclass
class CurveParams:
    """RGB 曲线（锚点列表）"""
    rgb:   List[CurvePoint] = field(default_factory=list)   # 主 RGB 曲线
    red:   List[CurvePoint] = field(default_factory=list)
    green: List[CurvePoint] = field(default_factory=list)
    blue:  List[CurvePoint] = field(default_factory=list)
    reason: str = ""

    def to_dict(self):
        def pts(lst):
            return [[p.input, p.output] for p in lst]
        return {
            "rgb":    pts(self.rgb),
            "red":    pts(self.red),
            "green":  pts(self.green),
            "blue":   pts(self.blue),
            "reason": self.reason,
        }


@dataclass
class HSLChannel:
    hue:        float = 0.0   # 色相 -100 ~ +100
    saturation: float = 0.0   # 饱和度 -100 ~ +100
    luminance:  float = 0.0   # 明度 -100 ~ +100


@dataclass
class HSLParams:
    """HSL 面板（8色通道）"""
    red:     HSLChannel = field(default_factory=HSLChannel)
    orange:  HSLChannel = field(default_factory=HSLChannel)
    yellow:  HSLChannel = field(default_factory=HSLChannel)
    green:   HSLChannel = field(default_factory=HSLChannel)
    aqua:    HSLChannel = field(default_factory=HSLChannel)
    blue:    HSLChannel = field(default_factory=HSLChannel)
    purple:  HSLChannel = field(default_factory=HSLChannel)
    magenta: HSLChannel = field(default_factory=HSLChannel)
    reason: str = ""

    def to_dict(self):
        channels = ["red","orange","yellow","green","aqua","blue","purple","magenta"]
        out = {}
        for ch in channels:
            c = getattr(self, ch)
            out[ch] = {"hue": c.hue, "saturation": c.saturation, "luminance": c.luminance}
        out["reason"] = self.reason
        return out


@dataclass
class SplitToneParams:
    """分离色调"""
    highlight_hue: float = 0.0   # 高光色相 0-360
    highlight_sat: float = 0.0   # 高光饱和 0-100
    shadow_hue:    float = 0.0   # 阴影色相 0-360
    shadow_sat:    float = 0.0   # 阴影饱和 0-100
    balance:       float = 0.0   # 平衡 -100~+100
    reason: str = ""

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class CalibrationParams:
    """相机校准（色温偏移修正）"""
    temp:  float = 0.0   # 色温偏移 -100~+100（蓝→橙）
    tint:  float = 0.0   # 色调偏移 -100~+100（绿→品）
    red_hue:    float = 0.0
    red_sat:    float = 0.0
    green_hue:  float = 0.0
    green_sat:  float = 0.0
    blue_hue:   float = 0.0
    blue_sat:   float = 0.0
    reason: str = ""

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ColorGradingParams:
    """完整追色参数包"""
    tone:        ToneParams
    curve:       CurveParams
    hsl:         HSLParams
    split_tone:  SplitToneParams
    calibration: CalibrationParams
    strength:    float = 1.0   # 全局强度系数 0-1（供用户微调）
    reasons:     Dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return {
            "tone":        self.tone.to_dict(),
            "curve":       self.curve.to_dict(),
            "hsl":         self.hsl.to_dict(),
            "split_tone":  self.split_tone.to_dict(),
            "calibration": self.calibration.to_dict(),
            "strength":    self.strength,
            "reasons":     self.reasons,
        }

    def apply_strength(self, s: float) -> "ColorGradingParams":
        """按强度系数 s (0-1) 缩放所有数值参数"""
        import copy
        p = copy.deepcopy(self)
        # 影调
        for attr in ["exposure","contrast","highlights","shadows","whites","blacks"]:
            setattr(p.tone, attr, getattr(p.tone, attr) * s)
        # 曲线（锚点偏移缩放）
        def scale_curve(pts):
            return [CurvePoint(pt.input, pt.input + (pt.output - pt.input) * s) for pt in pts]
        p.curve.rgb   = scale_curve(p.curve.rgb)
        p.curve.red   = scale_curve(p.curve.red)
        p.curve.green = scale_curve(p.curve.green)
        p.curve.blue  = scale_curve(p.curve.blue)
        # HSL
        for ch_name in ["red","orange","yellow","green","aqua","blue","purple","magenta"]:
            ch = getattr(p.hsl, ch_name)
            ch.hue *= s; ch.saturation *= s; ch.luminance *= s
        # 分离色调
        p.split_tone.highlight_sat *= s
        p.split_tone.shadow_sat    *= s
        # 校准
        for attr in ["temp","tint","red_hue","red_sat","green_hue","green_sat","blue_hue","blue_sat"]:
            setattr(p.calibration, attr, getattr(p.calibration, attr) * s)
        p.strength = s
        return p


# ─────────────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────────────

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def ev_from_L_diff(dL: float) -> float:
    """
    Lab L 差值 → Lightroom 曝光补偿 EV
    L∈[0,100]，经验公式：每 ~8L ≈ 1EV
    """
    return clamp(dL / 8.0, -5.0, 5.0)


def contrast_from_spread(ref_std: float, src_std: float) -> float:
    """
    对比度：参考图影调标准差 vs 原图标准差
    差值越大 → 需要增/减对比度
    映射到 [-100, +100]
    """
    diff = ref_std - src_std
    return clamp(diff * 4.0, -100.0, 100.0)


def highlight_adj(ref_p95: float, src_p95: float) -> float:
    """高光：亮部极值差 → LR 高光滑块"""
    diff = ref_p95 - src_p95
    return clamp(diff * 2.5, -100.0, 100.0)


def shadow_adj(ref_p5: float, src_p5: float) -> float:
    """阴影：暗部极值差 → LR 阴影滑块"""
    diff = ref_p5 - src_p5
    return clamp(diff * 2.5, -100.0, 100.0)


def lab_b_to_temp(delta_b: float) -> float:
    """
    Lab b* 差 → 色温滑块估算
    b* 正 = 偏黄（暖），b* 负 = 偏蓝（冷）
    LR 色温 -100(蓝) ~ +100(橙)
    """
    return clamp(delta_b * 3.5, -100.0, 100.0)


def lab_a_to_tint(delta_a: float) -> float:
    """
    Lab a* 差 → 色调滑块估算
    a* 正 = 偏红，a* 负 = 偏绿
    LR 色调 -100(绿) ~ +100(品红)
    """
    return clamp(delta_a * 3.5, -100.0, 100.0)


def build_tone_curve(
    ref_tone: ToneDistribution, src_tone: ToneDistribution
) -> List[CurvePoint]:
    """
    生成 RGB 主曲线锚点

    策略：
      - 暗部锚点（输入=src.p5）→ 输出按 ref.p5 的比例缩放
      - 中灰锚点（输入=src.p50）→ 输出按 ref.p50 调整
      - 亮部锚点（输入=src.p95）→ 输出按 ref.p95 调整
    所有值映射到 [0, 255]
    """
    def to255(v): return clamp(v / 100.0 * 255.0, 0, 255)

    pts = [
        CurvePoint(0, 0),
        CurvePoint(
            to255(src_tone.p5),
            clamp(to255(ref_tone.p5), 0, 60)          # 黑场
        ),
        CurvePoint(
            to255(src_tone.p50),
            clamp(to255(ref_tone.p50), 50, 200)        # 中灰
        ),
        CurvePoint(
            to255(src_tone.p95),
            clamp(to255(ref_tone.p95), 180, 255)       # 白场
        ),
        CurvePoint(255, 255),
    ]
    # 去掉重复/乱序点
    seen = {}
    for p in pts:
        if p.input not in seen:
            seen[p.input] = p
    pts = sorted(seen.values(), key=lambda p: p.input)
    return pts


def build_color_curves(
    ref_lab: LabStats, src_lab: LabStats
) -> Tuple[List[CurvePoint], List[CurvePoint], List[CurvePoint]]:
    """
    生成 R/G/B 分通道曲线（简化：仅用中点偏移）

    原理：
      - b* 对应蓝黄轴：ref 比 src 更暖(b*↑) → 提红/降蓝
      - a* 对应红绿轴：ref 比 src 更红(a*↑) → 提红/降绿
    """
    delta_b = ref_lab.mean_b - src_lab.mean_b  # 正=参考更暖
    delta_a = ref_lab.mean_a - src_lab.mean_a  # 正=参考更红

    # 中点偏移量（映射到曲线输出偏移，单位：0-255空间）
    red_shift   = clamp(delta_a * 1.5 + delta_b * 0.8, -30, 30)
    green_shift = clamp(-delta_a * 0.5, -20, 20)
    blue_shift  = clamp(-delta_b * 1.5, -30, 30)

    def mid_curve(shift: float) -> List[CurvePoint]:
        return [
            CurvePoint(0, 0),
            CurvePoint(128, clamp(128 + shift, 80, 175)),
            CurvePoint(255, 255),
        ]

    return mid_curve(red_shift), mid_curve(green_shift), mid_curve(blue_shift)


def build_hsl_params(ref_lab: LabStats, src_lab: LabStats) -> HSLParams:
    """
    生成 HSL 面板参数

    核心逻辑：
      - 整体饱和度差异 → 各通道等比调整
      - a* 差异 → 橙/红通道色相微调（影响肤色，需保守）
      - b* 差异 → 蓝/黄通道饱和度调整
    """
    # 整体饱和度差（用 a/b 的标准差之和近似评估饱和度）
    ref_sat_proxy = (ref_lab.std_a + ref_lab.std_b) / 2.0
    src_sat_proxy = (src_lab.std_a + src_lab.std_b) / 2.0
    sat_diff = clamp((ref_sat_proxy - src_sat_proxy) * 5.0, -60.0, 60.0)

    # a* 差 → 橙红通道色相偏移（人像中橙色主要是肤色，保守±10）
    delta_a = ref_lab.mean_a - src_lab.mean_a
    orange_hue = clamp(delta_a * 1.5, -10.0, 10.0)  # 保守，防肤色偏移

    # b* 差 → 蓝黄通道
    delta_b = ref_lab.mean_b - src_lab.mean_b
    blue_hue  = clamp(-delta_b * 1.2, -20.0, 20.0)
    yellow_hue = clamp(delta_b * 1.0, -15.0, 15.0)

    hsl = HSLParams(
        red    = HSLChannel(hue=clamp(delta_a*1.0, -8, 8), saturation=sat_diff*0.7, luminance=0),
        orange = HSLChannel(hue=orange_hue, saturation=sat_diff*0.5, luminance=0),  # ⚠ 肤色通道，保守
        yellow = HSLChannel(hue=yellow_hue, saturation=sat_diff*0.8, luminance=0),
        green  = HSLChannel(hue=0, saturation=sat_diff*0.9, luminance=0),
        aqua   = HSLChannel(hue=0, saturation=sat_diff, luminance=0),
        blue   = HSLChannel(hue=blue_hue, saturation=sat_diff, luminance=0),
        purple = HSLChannel(hue=0, saturation=sat_diff*0.8, luminance=0),
        magenta= HSLChannel(hue=clamp(-delta_a*0.8, -12, 12), saturation=sat_diff*0.6, luminance=0),
        reason = (
            f"整体饱和度差 Δ={sat_diff:.1f}；"
            f"a*差 Δ={delta_a:.2f} → 橙色色相{orange_hue:+.1f}（保守保护肤色）；"
            f"b*差 Δ={delta_b:.2f} → 蓝色色相{blue_hue:+.1f}"
        )
    )
    return hsl


def build_split_tone(ref_lab: LabStats, src_lab: LabStats) -> SplitToneParams:
    """
    分离色调

    原理：
      - 参考图高光的 a/b 均值 → 高光色相/饱和度
      - 参考图阴影的 a/b 均值 → 阴影色相/饱和度
    这里用全局 Lab 均值的高低分位近似代替（特征包没分通道采样时）
    """
    # 高光倾向（用 b* 正负判断暖/冷）
    ref_b = ref_lab.mean_b
    ref_a = ref_lab.mean_a

    # Lab a/b → HSL 色相（近似）
    hi_hue = float(np.degrees(np.arctan2(ref_b, ref_a)) % 360)  # 参考图整体色相
    # 高光饱和度：依据参考图 a/b std 估算氛围浓淡
    hi_sat = clamp(np.sqrt(ref_lab.std_a**2 + ref_lab.std_b**2) * 2.0, 0, 30)
    # 阴影通常与高光互补（偏冷处理）
    sh_hue = (hi_hue + 180) % 360
    sh_sat = hi_sat * 0.6

    return SplitToneParams(
        highlight_hue=round(hi_hue, 1),
        highlight_sat=round(hi_sat, 1),
        shadow_hue=round(sh_hue, 1),
        shadow_sat=round(sh_sat, 1),
        balance=0.0,
        reason=(
            f"参考图色相方向 ≈ {hi_hue:.0f}°，"
            f"高光饱和度 ≈ {hi_sat:.1f}，"
            f"阴影补色 ≈ {sh_hue:.0f}°"
        )
    )


# ─────────────────────────────────────────────────────
#  主生成器
# ─────────────────────────────────────────────────────

class ColorGradingGenerator:
    """
    差异分析与追色方案生成器

    用法：
        gen = ColorGradingGenerator()
        params = gen.generate(ref_features, src_features)
    """

    def generate(
        self,
        ref: ColorFeatures,
        src: ColorFeatures,
        strength: float = 1.0,
    ) -> ColorGradingParams:
        """
        根据参考图和原图特征，生成完整追色参数包

        strength: 全局强度系数 0-1
        """
        reasons = {}

        # ── 1. 影调参数 ──────────────────────────────
        dL_mean = ref.lab.mean_L - src.lab.mean_L
        exposure = ev_from_L_diff(dL_mean)

        contrast = contrast_from_spread(ref.lab.std_L, src.lab.std_L)
        highlights = highlight_adj(ref.tone.p95, src.tone.p95)
        shadows    = shadow_adj(ref.tone.p5,  src.tone.p5)
        whites     = clamp((ref.tone.p95 - src.tone.p95) * 1.5, -60, 60)
        blacks     = clamp((ref.tone.p5  - src.tone.p5)  * 1.5, -60, 60)

        reasons["tone"] = (
            f"原片 L均值={src.lab.mean_L:.1f}，参考图={ref.lab.mean_L:.1f}，"
            f"差Δ={dL_mean:.1f} → 曝光{exposure:+.2f}EV；"
            f"原片对比度std={src.lab.std_L:.1f}，参考={ref.lab.std_L:.1f} → 对比度{contrast:+.0f}；"
            f"高光{highlights:+.0f}，阴影{shadows:+.0f}"
        )

        tone = ToneParams(
            exposure=round(exposure, 2),
            contrast=round(contrast, 1),
            highlights=round(highlights, 1),
            shadows=round(shadows, 1),
            whites=round(whites, 1),
            blacks=round(blacks, 1),
            reason=reasons["tone"],
        )

        # ── 2. RGB 曲线 ──────────────────────────────
        rgb_curve = build_tone_curve(ref.tone, src.tone)
        red_curve, green_curve, blue_curve = build_color_curves(ref.lab, src.lab)

        delta_b = ref.lab.mean_b - src.lab.mean_b
        delta_a = ref.lab.mean_a - src.lab.mean_a
        reasons["curve"] = (
            f"b*差Δ={delta_b:.2f}（正=参考更暖）→ 蓝通道{'下压' if delta_b>0 else '上提'}；"
            f"a*差Δ={delta_a:.2f}（正=参考更红）→ 红通道{'上提' if delta_a>0 else '下压'}"
        )

        curve = CurveParams(
            rgb=rgb_curve,
            red=red_curve,
            green=green_curve,
            blue=blue_curve,
            reason=reasons["curve"],
        )

        # ── 3. HSL ──────────────────────────────────
        hsl = build_hsl_params(ref.lab, src.lab)
        reasons["hsl"] = hsl.reason

        # ── 4. 分离色调 ──────────────────────────────
        split = build_split_tone(ref.lab, src.lab)
        reasons["split_tone"] = split.reason

        # ── 5. 校准（白平衡偏移）────────────────────
        temp_adj = lab_b_to_temp(delta_b)
        tint_adj = lab_a_to_tint(delta_a)
        reasons["calibration"] = (
            f"b*差{delta_b:+.2f} → 色温{temp_adj:+.0f}（正=偏暖）；"
            f"a*差{delta_a:+.2f} → 色调{tint_adj:+.0f}（正=偏品红）"
        )

        calibration = CalibrationParams(
            temp=round(temp_adj, 1),
            tint=round(tint_adj, 1),
            red_hue=0, red_sat=0,
            green_hue=0, green_sat=0,
            blue_hue=0, blue_sat=0,
            reason=reasons["calibration"],
        )

        params = ColorGradingParams(
            tone=tone,
            curve=curve,
            hsl=hsl,
            split_tone=split,
            calibration=calibration,
            strength=1.0,
            reasons=reasons,
        )

        # 按强度缩放
        if strength != 1.0:
            params = params.apply_strength(strength)

        return params
