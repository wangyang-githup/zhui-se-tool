"""
evaluator.py — 追色效果评价指标
=================================
提供客观指标衡量追色结果质量：
1. 直方图相似度（Bhattacharyya 距离）
2. 肤色区域 ΔE CIE76
3. Lab 均值匹配度
4. 综合评价摘要
"""

import numpy as np
from color_engine import rgb_to_lab
from feature_extractor import ColorFeatures, detect_skin_mask
from color_renderer import delta_e_cie76


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
        de = delta_e_cie76(src_lab, result_lab)
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

    @staticmethod
    def compute_metrics(src: np.ndarray, result: np.ndarray) -> dict:
        """快速指标计算（无需参考图特征）"""
        h, w = src.shape[:2]

        # 处理速度估算（预留，由调用方传入）
        # 肤色偏差
        skin_de = Evaluator.skin_delta_e(src, result)

        # 覆盖度（结果图中有效像素占比）
        result_gray = np.mean(result, axis=2)
        coverage = float(np.mean(result_gray > 0.01))

        # 色差 ΔE（全图）
        src_lab = rgb_to_lab(src)
        result_lab = rgb_to_lab(result)
        full_de = delta_e_cie76(src_lab, result_lab)
        mean_de = float(np.mean(full_de))

        return {
            "delta_e": round(mean_de, 1),
            "skin_safe": skin_de.get("safe", True),
            "skin_detected": skin_de.get("skin_detected", False),
            "coverage": round(coverage * 100),
        }
