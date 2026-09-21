# -*- coding: utf-8 -*-
"""OCR Web 服务：FastAPI + PaddleOCR。图片二进制随请求直传，即传即识，服务端不存储任何图片。"""
import io
import os
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent

# 模型加载：优先本地已解压目录（开发环境）；部署环境则解压随包分发的
# models.zip 到临时目录。两者都缺失时回落到 paddlex 默认的联网下载。
# 注意：缓存目录必须在导入 paddle 前设置。
_official = BASE_DIR / "models" / "paddlex" / "official_models"
if not _official.is_dir():
    _models_zip = BASE_DIR / "models.zip"
    if _models_zip.is_file():
        _extract_root = Path(tempfile.gettempdir()) / "webocr_models"
        if not (_extract_root / "official_models").is_dir():
            with zipfile.ZipFile(_models_zip) as zf:
                zf.extractall(_extract_root)
        _official = _extract_root / "official_models"
if _official.is_dir():
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(_official.parent)
    # 模型已随包备好，跳过 paddlex 在线模型源检查，保证离线可用
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# 关闭 paddlex 的 eager 初始化：推理不依赖它，且可避免部署环境在同一进程内
# 重复初始化导致的 "RuntimeError: PDX has already been initialized"
os.environ.setdefault("PADDLE_PDX_EAGER_INIT", "False")

# 部署平台镜像缺少 opencv-contrib-python(非 headless，paddlex 硬性依赖)所需的
# libGL/libX11/libglib 系统库，以及 paddle 所需的 libgomp。这里把 Debian
# bullseye 提取的 .so 随包分发（vendored_libs/），在导入 paddleocr 前按依赖序
# 以 RTLD_GLOBAL 预加载，使 DT_NEEDED 直接命中已加载对象，无需改动平台镜像。
if sys.platform == "linux":
    import ctypes
    _vendored = BASE_DIR / "vendored_libs"
    if _vendored.is_dir():
        for _name in (
            "libgomp.so.1", "libGLdispatch.so.0", "libXau.so.6", "libmd.so.0",
            "libbsd.so.0", "libXdmcp.so.6", "libxcb.so.1", "libX11.so.6",
            "libGLX.so.0", "libGL.so.1", "libpcre.so.3", "libglib-2.0.so.0",
            "libgthread-2.0.so.0",
        ):
            _lib = _vendored / _name
            if _lib.is_file():
                ctypes.CDLL(str(_lib), mode=ctypes.RTLD_GLOBAL)

app = FastAPI(title="文本提取器")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _diag(msg: str):
    # 诊断日志：stdout 即时刷新，确保在平台 runtime_logs 中可见
    print(f"[webocr] {msg}", flush=True)

# ---------- PaddleOCR 初始化（中文识别，模型随包分发于 models/paddlex，无需联网下载） ----------
_ocr = None
_ocr_lock = threading.Lock()


def get_ocr():
    global _ocr
    with _ocr_lock:
        if _ocr is None:
            import logging
            import traceback

            _diag("initializing PaddleOCR (import paddleocr) ...")
            try:
                from paddleocr import PaddleOCR
            except Exception:
                _diag("import paddleocr FAILED:\n" + traceback.format_exc())
                raise

            # 关闭 paddlex 冗余日志
            for name in ("paddlex", "ppx", "paddleocr"):
                logging.getLogger(name).setLevel(logging.WARNING)
            # enable_mkldnn=False 规避 paddlepaddle 3.3 在部分 CPU 上
            # 的 oneDNN bug；关闭方向分类/去畸变等预处理模型，提升速度。
            # 检测/识别均用 small：模型随包分发（共约 30MB），适配部署包体积限制
            _diag("creating PaddleOCR instance ...")
            try:
                _ocr = PaddleOCR(
                    lang="ch",
                    text_detection_model_name="PP-OCRv6_small_det",
                    text_recognition_model_name="PP-OCRv6_small_rec",
                    enable_mkldnn=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            except Exception:
                _diag("PaddleOCR() FAILED:\n" + traceback.format_exc())
                raise
            _diag("PaddleOCR ready")
    return _ocr


MAX_IMAGE_BYTES = 15 * 1024 * 1024


@app.on_event("startup")
def _warmup_ocr():
    # 启动横幅：记录进程与环境信息。若容器反复被杀重启，日志里会出现
    # 多条 startup 横幅，可据此识别崩溃循环（如 OOM）。
    import platform

    _diag(f"startup pid={os.getpid()} python={platform.python_version()} {platform.platform()}")
    if sys.platform == "linux":
        try:
            mem = {}
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith(("VmRSS", "VmPeak")):
                    k, v = line.split(":", 1)
                    mem[k] = v.strip()
            _diag(f"mem {mem}")
        except Exception:
            pass

    # 启动即后台预热模型：服务立即监听端口（健康检查不受影响），
    # 用户首次识别时模型通常已就绪，无需等待加载
    def _warm():
        try:
            get_ocr()
        except Exception:
            import traceback

            _diag("warmup FAILED:\n" + traceback.format_exc())

    threading.Thread(target=_warm, daemon=True).start()


# ---------- 页面 ----------
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# ---------- API：无状态 OCR，图片二进制随请求上传，不落盘不缓存 ----------
@app.post("/api/ocr")
async def ocr(request: Request):
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="缺少图片数据")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="图片过大")
    try:
        # 解码为 ndarray 并转成 PaddleOCR 习惯的 BGR 通道顺序
        arr = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))[:, :, ::-1].copy()
    except Exception:
        raise HTTPException(status_code=400, detail="无法解析的图片数据")
    lines = []
    try:
        for r in get_ocr().predict(arr):
            texts = list(r["rec_texts"])
            scores = list(r["rec_scores"])
            # 过滤低置信度结果（阈值 0.75），避免输出明显误识别内容
            lines.extend(t for t, s in zip(texts, scores) if s >= 0.75)
    except Exception as e:
        import traceback

        # 错误既进运行日志（traceback 完整），也进响应体（便于前端/排查定位）
        _diag("/api/ocr predict FAILED:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"OCR failure: {e}"[:500])
    return {"lines": lines, "full_text": "\n".join(lines)}


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="OCR Web 服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，0.0.0.0 允许局域网访问")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("main:app", host=args.host, port=args.port, reload=args.reload)
