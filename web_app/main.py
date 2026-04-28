"""
模块四：FastAPI 后端
==========================================
API 设计：

POST /api/analyze
  - 上传 ref_image + src_image
  - 返回：{ref_features, src_features, params, preview_base64}

POST /api/render
  - 接受 src_image + params_json（含用户微调的 strength）
  - 返回：{preview_base64, metrics}

GET /
  - 返回前端 HTML 页面
"""

import io
import base64
import json
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from feature_extractor import FeatureExtractor, load_as_rgb_float
from grading_generator import ColorGradingGenerator
from color_renderer import ColorRenderer
from evaluator import evaluate_result   # 模块五

app = FastAPI(title="追色 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = FeatureExtractor()
generator = ColorGradingGenerator()
renderer  = ColorRenderer()


# ─────────────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────────────

def file_to_rgb_float(upload: UploadFile) -> np.ndarray:
    """UploadFile → float32 RGB [0,1]"""
    data = upload.file.read()
    arr  = np.frombuffer(data, np.uint8)
    bgr  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法解码图片：{upload.filename}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # 限制最大分辨率（加速处理）
    H, W = rgb.shape[:2]
    MAX_SIDE = 1200
    if max(H, W) > MAX_SIDE:
        scale = MAX_SIDE / max(H, W)
        rgb = cv2.resize(rgb, (int(W*scale), int(H*scale)))
    return rgb


def rgb_float_to_jpeg_b64(img: np.ndarray, quality: int = 88) -> str:
    """float32 RGB [0,1] → base64 JPEG 字符串"""
    u8  = (img.clip(0,1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ─────────────────────────────────────────────────────
#  路由
# ─────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/analyze")
async def analyze(
    ref_image: UploadFile = File(..., description="参考图"),
    src_image: UploadFile = File(..., description="原图"),
    strength:  float      = Form(1.0, description="全局强度 0-1"),
):
    """
    上传两张图 → 返回追色参数 + 预览图 + 评价指标

    Response JSON:
    {
      "params": {...},          // 完整追色参数包
      "ref_features": {...},    // 参考图特征
      "src_features": {...},    // 原图特征
      "preview_b64": "...",     // 追色结果 base64 JPEG
      "metrics": {...}          // 评价指标
    }
    """
    try:
        ref_img = file_to_rgb_float(ref_image)
        src_img = file_to_rgb_float(src_image)

        ref_feat = extractor.extract(ref_img)
        src_feat = extractor.extract(src_img)

        params = generator.generate(ref_feat, src_feat, strength=strength)

        result = renderer.render(src_img, params, ref_feat)

        metrics = evaluate_result(src_img, ref_img, result, ref_feat, src_feat)

        return JSONResponse({
            "params":       params.to_dict(),
            "ref_features": ref_feat.to_dict(),
            "src_features": src_feat.to_dict(),
            "preview_b64":  rgb_float_to_jpeg_b64(result),
            "metrics":      metrics,
        })

    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()}, status_code=500)


@app.post("/api/render")
async def render_with_params(
    src_image:   UploadFile = File(...),
    ref_image:   UploadFile = File(...),
    params_json: str        = Form(..., description="修改后的追色参数 JSON"),
    strength:    float      = Form(1.0),
):
    """
    用户微调参数后重新渲染

    接受修改后的 params_json（前端滑块调节后序列化），重新应用并返回预览
    """
    try:
        src_img = file_to_rgb_float(src_image)
        ref_img = file_to_rgb_float(ref_image)
        ref_feat = extractor.extract(ref_img)

        # 从前端回传的 JSON 重建参数（仅更新 strength 和核心数值）
        raw = json.loads(params_json)
        params = generator.generate(ref_feat, extractor.extract(src_img), strength=strength)
        # 覆盖前端微调的核心数值
        if "tone" in raw:
            t = raw["tone"]
            for field in ["exposure","contrast","highlights","shadows","whites","blacks"]:
                if field in t:
                    setattr(params.tone, field, float(t[field]))

        result = renderer.render(src_img, params, ref_feat)
        metrics = evaluate_result(src_img, ref_img, result, ref_feat, extractor.extract(src_img))

        return JSONResponse({
            "preview_b64": rgb_float_to_jpeg_b64(result),
            "metrics":     metrics,
        })

    except Exception as e:
        import traceback
        return JSONResponse({"error": str(e), "traceback": traceback.format_exc()}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
