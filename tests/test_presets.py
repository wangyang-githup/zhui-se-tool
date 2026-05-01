"""
测试：预设配置校验
确保 presets.json 格式正确、所有参数完整
"""
import json
from pathlib import Path

PRESET_PATH = Path(__file__).parent.parent / "presets.json"


def test_presets_load():
    assert PRESET_PATH.exists(), f"预设文件不存在: {PRESET_PATH}"
    with open(PRESET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert "presets" in data
    assert len(data["presets"]) >= 8, f"预设数不足8个: {len(data['presets'])}"


def test_preset_params():
    with open(PRESET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    required_params = [
        "exposure", "contrast", "highlights", "shadows",
        "whites", "blacks", "temperature", "tint",
        "texture", "clarity", "dehaze", "grain",
        "highlight_hue", "highlight_saturation",
        "shadow_hue", "shadow_saturation",
    ]

    for preset in data["presets"]:
        assert "id" in preset, f"预设缺少id: {preset.get('name', '?')}"
        assert "name" in preset, f"预设缺少name"
        assert "params" in preset, f"预设 {preset['id']} 缺少params"
        for param in required_params:
            assert param in preset["params"], \
                f"预设 {preset['id']} 缺少参数: {param}"
