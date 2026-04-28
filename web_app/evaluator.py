"""
模块五：评价指标 + 测试样例说明
==========================================
评价维度：
  1. 直方图相似度（L 通道 Bhattacharyya 距离）
  2. 主色色相差（Palette 色相 Wasserstein 距离）
  3. 肤色偏移量（ΔE CIE76，肤色区域均值）
  4. Lab 均值匹配度
"""

import numpy as np
import cv2
from feature_extractor import ColorFeatures, detect_skin_mask


def histogram_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    L 通道直方图 Bhattacharyya 相似度 (0=完全不同, 1=完全相同)

    img_a/b: float32 RGB [0,1]
    """
    def get_l_hist(img):
        u8 = (img * 255).clip(0, 255).astype(np.uint8)
        lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0]
        hist = cv2.calcHist([L], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        return hist

    h_a = get_l_hist(img_a)
    h_b = get_l_hist(img_b)
    # Bhattacharyya：0=相同，2=最不同；转为相似度 0-1
    dist = cv2.compareHist(h_a, h_b, cv2.HISTCMP_BHATTACHARYYA)
    return round(float(1 - dist / 2), 4)


def palette_hue_diff(palette_a, palette_b) -> float:
    """
    主色色相差（取前3主色，计算加权色相差均值）

    palette_a/b: ColorPalette
    返回：平均色相差（度，0-180）
    """
    def rgb_to_hue(r, g, b):
        # 归一化到 0-1
        r_, g_, b_ = r/255, g/255, b/255
        Cmax = max(r_, g_, b_)
        Cmin = min(r_, g_, b_)
        delta = Cmax - Cmin
        if delta < 0.01:
            return 0.0
        if Cmax == r_:
            h = 60 * (((g_ - b_) / delta) % 6)
        elif Cmax == g_:
            h = 60 * (((b_ - r_) / delta) + 2)
        else:
            h = 60 * (((r_ - g_) / delta) + 4)
        return h % 360

    diffs = []
    n = min(3, len(palette_a.colors), len(palette_b.colors))
    for i in range(n):
        h_a = rgb_to_hue(*palette_a.colors[i])
        h_b = rgb_to_hue(*palette_b.colors[i])
        diff = abs(h_a - h_b)
        if diff > 180:
            diff = 360 - diff
        diffs.append(diff * palette_a.weights[i])

    return round(float(np.mean(diffs)) if diffs else 0.0, 2)


def skin_delta_e(src: np.ndarray, result: np.ndarray) -> dict:
    """
    肤色区域 ΔE CIE76（原图 vs 追色结果）

    返回：{mean_de, max_de, safe: bool}
    """
    skin_mask_u8 = detect_skin_mask(src)
    skin_bool = skin_mask_u8 > 0

    if not skin_bool.any():
        return {"mean_de": 0.0, "max_de": 0.0, "safe": True, "skin_detected": False}

    def to_lab(img):
        u8 = (img * 255).clip(0, 255).astype(np.uint8)
        lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:,:,0] = lab[:,:,0] / 255.0 * 100.0
        lab[:,:,1] -= 128.0
        lab[:,:,2] -= 128.0
        return lab

    src_lab    = to_lab(src)
    result_lab = to_lab(result)

    diff = src_lab - result_lab
    de   = np.sqrt((diff ** 2).sum(axis=-1))

    skin_de_vals = de[skin_bool]
    mean_de = float(skin_de_vals.mean())
    max_de  = float(skin_de_vals.max())

    return {
        "mean_de":       round(mean_de, 2),
        "max_de":        round(max_de, 2),
        "safe":          mean_de < 5.0,
        "skin_detected": True,
    }


def lab_match_score(ref_feat: ColorFeatures, result: np.ndarray) -> dict:
    """
    追色结果与参考图 Lab 均值匹配度

    返回：{dL, da, db, overall_de}（越小越好）
    """
    u8 = (result * 255).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:,:,0] = lab[:,:,0] / 255.0 * 100.0
    lab[:,:,1] -= 128.0
    lab[:,:,2] -= 128.0

    res_L = float(lab[:,:,0].mean())
    res_a = float(lab[:,:,1].mean())
    res_b = float(lab[:,:,2].mean())

    dL = abs(res_L - ref_feat.lab.mean_L)
    da = abs(res_a - ref_feat.lab.mean_a)
    db = abs(res_b - ref_feat.lab.mean_b)
    overall = float(np.sqrt(dL**2 + da**2 + db**2))

    return {
        "dL": round(dL, 2),
        "da": round(da, 2),
        "db": round(db, 2),
        "overall_de": round(overall, 2),
        "grade": "优秀" if overall < 5 else "良好" if overall < 10 else "待优化"
    }


def evaluate_result(
    src: np.ndarray,
    ref: np.ndarray,
    result: np.ndarray,
    ref_feat: ColorFeatures,
    src_feat: ColorFeatures,
) -> dict:
    """
    综合评价函数

    返回完整评价指标字典
    """
    hist_sim = histogram_similarity(ref, result)
    hue_diff = palette_hue_diff(ref_feat.palette, ref_feat.env_palette)  # TODO：result palette
    skin_de  = skin_delta_e(src, result)
    lab_match = lab_match_score(ref_feat, result)

    return {
        "histogram_similarity": hist_sim,         # 0-1，越高越像参考图影调
        "palette_hue_diff_deg": hue_diff,         # 度，越小越接近参考色调
        "skin_delta_e":         skin_de,           # ΔE，肤色偏移
        "lab_match":            lab_match,         # Lab 整体匹配度
        "summary": (
            f"影调相似度 {hist_sim:.1%}，"
            f"Lab偏差 ΔE={lab_match['overall_de']:.1f}（{lab_match['grade']}），"
            f"肤色安全：{'✓' if skin_de.get('safe', True) else '✗ ΔE=' + str(skin_de.get('mean_de', 0))}"
        )
    }


# ─────────────────────────────────────────────────────
#  测试样例说明
# ─────────────────────────────────────────────────────

TEST_CASES = {
    "日系清新": {
        "参考图特征": {
            "影调": "高亮度（L均值≈70），低对比（std≈12），高光轻微溢出",
            "色温": "微冷偏青白（b*≈+3，a*≈-2）",
            "主色": "低饱和白/青/米色",
            "肤色": "白皙通透，高光Lab≈(85, 5, 10)，阴影Lab≈(60, 8, 15)",
        },
        "原图": "室外暖午光人像，L均值≈55，b*≈+18（偏暖黄），对比度std≈18",
        "预期调整": {
            "曝光": "+0.8EV",
            "对比度": "-20",
            "高光": "-30（压制过曝）",
            "色温": "-40（降暖）",
            "蓝色饱和度": "+15",
            "橙色色相": "-8（肤色偏移修正）",
        },
    },
    "暗调电影": {
        "参考图特征": {
            "影调": "低调（L均值≈35），高反差（std≈22），黑位深",
            "色温": "冷蓝调（b*≈-8，a*≈0）",
            "主色": "深蓝/青灰/黑",
            "肤色": "偏暗，高光Lab≈(65, 8, 12)，阴影Lab≈(30, 6, 5)",
        },
        "原图": "室内柔光人像，L均值≈62，b*≈+12（暖），对比度std≈14",
        "预期调整": {
            "曝光": "-0.7EV",
            "对比度": "+35",
            "黑色": "-40（压黑场）",
            "分离色调/阴影": "蓝色（色相220°，饱和15）",
            "肤色保护": "ΔE监控，阴影区不加蓝到肤色",
        },
    },
    "复古胶片": {
        "参考图特征": {
            "影调": "提黑场（黑色不纯，L_p5≈8），对比中等",
            "色温": "高光偏橙（b*≈+15），阴影偏绿（a*≈-5）",
            "主色": "橙黄/绿灰",
            "肤色": "奶油偏黄，高光Lab≈(78, 8, 20)，阴影Lab≈(48, 5, 12)",
        },
        "原图": "标准色彩人像，黑场干净，L_p5≈2",
        "预期调整": {
            "黑色": "+35（提黑场，复古感）",
            "分离色调/高光": "橙（35°，饱和20）",
            "分离色调/阴影": "绿（160°，饱和12）",
            "橙色饱和": "+15（强化暖肤感）",
        },
    },
}
