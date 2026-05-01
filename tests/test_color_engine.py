"""
测试：色彩引擎核心功能
验证色彩空间转换、图像加载、LUT 生成
"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from color_engine import (
    rgb_to_lab, lab_to_rgb, load_image, save_image,
    ColorTransfer, LUTGenerator, skin_mask,
)


def test_rgb_to_lab_roundtrip():
    """RGB -> Lab -> RGB 保持数据完整性"""
    rgb = np.random.rand(16, 16, 3).astype(np.float32)
    lab = rgb_to_lab(rgb)
    recovered = lab_to_rgb(lab)
    assert recovered.shape == rgb.shape
    assert recovered.dtype == rgb.dtype
    # 允许少量精度损失
    max_diff = np.max(np.abs(recovered - rgb))
    assert max_diff < 0.05, f"最大偏差: {max_diff}"


def test_skin_mask_output():
    """肤色掩码输出有效形状和类型"""
    img = np.random.rand(64, 64, 3).astype(np.float32)
    mask = skin_mask(img)
    assert mask.shape[:2] == (64, 64), f"掩码形状错误: {mask.shape}"
    assert mask.dtype in (np.uint8, np.float32), f"掩码类型错误: {mask.dtype}"
    assert mask.max() <= 255 and mask.min() >= 0


def test_color_transfer_basic():
    """基础色彩迁移不崩溃"""
    ref = np.random.rand(32, 32, 3).astype(np.float32)
    src = np.random.rand(32, 32, 3).astype(np.float32)

    transfer = ColorTransfer()
    transfer.analyze(ref)
    result = transfer.transfer(src)

    assert result.shape == src.shape
    assert result.dtype == np.float32
    assert result.min() >= -0.01 and result.max() <= 1.01


def test_lut_generator():
    """LUT 生成为有效 .cube 文件"""
    import tempfile
    ref = np.random.rand(16, 16, 3).astype(np.float32)
    transfer = ColorTransfer()
    transfer.analyze(ref)

    gen = LUTGenerator(lut_size=17)
    with tempfile.NamedTemporaryFile(suffix=".cube", delete=False) as f:
        gen.generate(transfer, output_path=f.name)
        content = Path(f.name).read_text()
        assert "TITLE" in content
        assert "LUT_3D_SIZE" in content
        size_line = [l for l in content.splitlines() if "LUT_3D_SIZE" in l][0]
        assert "17" in size_line
        Path(f.name).unlink()
