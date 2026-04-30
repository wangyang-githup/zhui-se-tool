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
import os
import sys
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from color_engine import (
    ZhuiseEngine,
    FeatureExtractor,
    ColorGradingGenerator,
)

app = FastAPI(title="追色 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = FeatureExtractor()
generator = ColorGradingGenerator()
engine = ZhuiseEngine()


# ─────────────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────────────

def file_to_rgb_float(upload: UploadFile) -> np.ndarray:
    """UploadFile → float32 RGB [0,1]"""
    data = upload.file.read()
    pil = PILImage.open(io.BytesIO(data)).convert("RGB")
    W, H = pil.size
    MAX_SIDE = 1200
    if max(H, W) > MAX_SIDE:
        scale = MAX_SIDE / max(H, W)
        pil = pil.resize((int(W * scale), int(H * scale)), PILImage.LANCZOS)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return arr


def rgb_float_to_jpeg_b64(img: np.ndarray, quality: int = 88) -> str:
    """float32 RGB [0,1] → base64 JPEG 字符串"""
    u8 = (img.clip(0, 1) * 255).astype(np.uint8)
    pil = PILImage.fromarray(u8, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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

        engine.load_reference(ref_img)
        ref_feat = engine.ref_features
        src_feat = extractor.extract(src_img)
        params = generator.generate(ref_feat, src_feat, strength=strength)
        result = engine.render(src_img, mode="expert", strength=strength, params=params)
        metrics = engine.evaluate(src_img, result)

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
        engine.load_reference(ref_img)
        ref_feat = engine.ref_features

        raw = json.loads(params_json)
        src_feat = extractor.extract(src_img)
        params = generator.generate(ref_feat, src_feat, strength=strength)
        if "tone" in raw:
            t = raw["tone"]
            for field in ["exposure","contrast","highlights","shadows","whites","blacks"]:
                if field in t:
                    setattr(params.tone, field, float(t[field]))

        result = engine.render(src_img, mode="expert", strength=strength, params=params)
        metrics = engine.evaluate(src_img, result)

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
