# webOCR 文本提取器

手机网页拍照 OCR：打开网页即可拍照/选图，服务端 PaddleOCR 识别后返回文字，支持历史记录与一键复制。

**在线体验**：https://webocr.pocketbay.app

> 建议用手机内置浏览器打开（微信内置浏览器不支持摄像头调用，会提示跳转到系统浏览器）。

## 功能

- 拍照识别：竖屏实时取景，点击扫描键即拍即识别
- 无摄像头兜底：环境不支持摄像头时自动调起系统相机/相册
- 历史记录：图片与识别文本保存在浏览器本地 IndexedDB（左图右文对照），服务端不留存图片
- 一键复制：点击已识别文本即复制到剪贴板
- 保活与自愈：页面可见期间每 25s 心跳防止平台休眠；识别请求遇到平台唤醒态（204/403/502/503）自动等待唤醒并重试

## 技术栈

- 前端：原生 HTML/CSS/JS 单页（`templates/index.html`），FastAPI 模板渲染
- 服务端：FastAPI + Uvicorn（`main.py`）
- OCR：PaddleOCR（CPU 推理），模型随包分发（`models.zip`），启动即预热
- 部署：PocketBay 托管（空闲会缩容到零，由前端心跳与自动重试保障体验）

## 本地运行

需要 Python 3.10+：

```bash
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

浏览器访问 http://127.0.0.1:8000 。首次启动会加载模型，等待日志出现 `PaddleOCR ready` 后即可识别。

## 接口

`POST /api/ocr`，请求体为图片二进制（`Content-Type: image/jpeg` 或 `image/png`），返回：

```json
{"lines": ["..."], "full_text": "..."}
```

## 仓库说明

- `models.zip`：OCR 模型离线包（约 25MB），服务端启动时解压使用，避免运行时下载
- `vendored_libs/`：Linux 运行所需的系统动态库（libGL/libgomp 等），解决极简容器镜像缺库问题
- `pocketbay-deploy.zip`：临时部署产物，已在 .gitignore 中忽略
