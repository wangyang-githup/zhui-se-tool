"""
测试：特征提取与追色方案生成
验证特征提取和评分生成器的基本功能
"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_extractor import FeatureExtractor
from grading_generator import ColorGradingGenerator


def test_feature_extractor_shape():
    """特征提取器返回有效特征包"""
    extractor = FeatureExtractor()
    img = np.random.rand(64, 64, 3).astype(np.float32)
    features = extractor.extract(img)

    assert features.lab is not None
    assert features.tone is not None
    assert features.palette is not None
    assert hasattr(features, "skin")
    assert features.lab.mean_L >= 0


def test_grading_generator():
    """方案生成器返回有效参数包"""
    extractor = FeatureExtractor()
    generator = ColorGradingGenerator()

    ref = extractor.extract(np.random.rand(64, 64, 3).astype(np.float32))
    src = extractor.extract(np.random.rand(64, 64, 3).astype(np.float32))

    params = generator.generate(ref, src)
    assert params is not None
    assert hasattr(params, "tone")
    assert hasattr(params, "split_tone")

    # 应用强度
    applied = params.apply_strength(0.5)
    assert applied is not None
