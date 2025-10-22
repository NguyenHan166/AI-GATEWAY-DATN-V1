from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Request,
    Query,
    UploadFile,
    File,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import PlainTextResponse
from slowapi.middleware import SlowAPIMiddleware

from .config import ALLOWED_ORIGINS, PRESIGN_EXPIRES_SECONDS
from .models import (
    Manifest,
    Pack,
    PresetFile,
    PresignReq,
    PresignResp,
    StylizeReq,
    StylizeResp,
    ManifestPaged,
    RestoreResp,
)
from .storage import presign_get, get_or_put_cached, make_inference_key_from_bytes
from .security import require_api_key
from .ai_service.manifest_service import get_manifest_cached, filter_packs, paginate
from .ai_service.inference_service import (
    call_restore_from_bytes,
    call_remove_bg_from_bytes,
    InferenceError,
)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Filters Backend (Cloudflare R2)")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Lưu ý: handler này cũng cần tham số request
@app.exception_handler(RateLimitExceeded)
def ratelimit_handler(request: Request, exc: RateLimitExceeded):
    return PlainTextResponse("Too Many Requests", status_code=429)


# ---------- ENDPOINTS ----------


# tạm thời thêm vào main.py
from .storage import list_objects


@app.get("/_debug/ls")
async def debug_ls():
    r = list_objects("ON1_BW_LUTs/")
    return r.get("Contents", [])[:10]


from .security import require_api_key


@app.get("/manifest", response_model=ManifestPaged)  # đổi response_model
@limiter.limit("30/minute")
async def get_manifest(
    request: Request,
    authorized=Depends(
        require_api_key
    ),  # hoặc optional_user nếu bạn vẫn muốn cho public
    category: str | None = Query(None),
    target: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),  # giới hạn an toàn
):
    """
    Trả manifest đã lọc theo category/target, có phân trang packs.
    Ví dụ:
      /manifest?category=ON1_BW_LUTs&target=For_Lightroom&page=1&page_size=20
    """
    data = get_manifest_cached()  # {'version': '...', 'packs': [...]}

    filtered = filter_packs(data, category, target)
    page_items, total, page, total_pages = paginate(filtered, page, page_size)

    # ép kiểu về Pack cho đúng pydantic (Pack trong models.py)
    packs_model = [Pack(**p) for p in page_items]

    return ManifestPaged(
        version=data.get("version", "unknown"),
        packs=packs_model,
        total_packs=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@app.post("/presign", response_model=PresignResp)
@limiter.limit("20/minute")
async def presign(
    request: Request, req: PresignReq, authorized=Depends(require_api_key)
):
    data = get_manifest_cached()
    pack = next((p for p in data["packs"] if p["id"] == req.pack_id), None)
    if not pack:
        raise HTTPException(404, "Pack not found")
    if not any(f["key"] == req.key for f in pack["files"]):
        raise HTTPException(404, "File not found in pack")

    url = presign_get(req.key, expires=PRESIGN_EXPIRES_SECONDS)
    return PresignResp(url=url, expires_in=PRESIGN_EXPIRES_SECONDS)

    # uvicorn app.main:app --reload --port 8080

    """
        
        fun downloadPreset(context: Context, presignedUrl: String, fileName: String) {
        val uri = Uri.parse(presignedUrl)
        val request = DownloadManager.Request(uri)
            .setTitle(fileName)
            .setDescription("Đang tải filter...")
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            .setDestinationInExternalFilesDir(
                context,
                Environment.DIRECTORY_DOWNLOADS,
                "filters/$fileName"
            )
            .setAllowedOverMetered(true)
            .setAllowedOverRoaming(true)

        val dm = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
        dm.enqueue(request)
    }
    
    """


# ---- Inference: Restore (multipart upload) ----
@app.post("/inference/restore", response_model=RestoreResp)
@limiter.limit("30/minute")
async def restore_multipart(
    request: Request,
    authorized=Depends(require_api_key),
    image: UploadFile = File(..., description="Ảnh upload từ client (jpg/png/webp)"),
    task: str = Form("upscale+face_restore"),
    upscale: int = Form(4),
    tile: int = Form(0),
    codeformer_fidelity: float = Form(0.5),
    background_enhance: bool = Form(True),
    face_upsample: bool = Form(True),
):
    """
    Nâng cấp & khôi phục ảnh: Real-ESRGAN (upscale) + CodeFormer (face restore).
    - Nhận ảnh trực tiếp từ client (multipart/form-data).
    - Cache kết quả lên R2 theo nội dung ảnh + tham số.
    """
    # Validate tham số cơ bản
    if task not in ("upscale", "upscale+face_restore"):
        raise HTTPException(422, "task must be 'upscale' or 'upscale+face_restore'")
    if upscale not in (2, 3, 4):
        raise HTTPException(422, "upscale must be 2, 3, or 4")
    if tile < 0:
        raise HTTPException(422, "tile must be >= 0")
    if not (0.0 <= codeformer_fidelity <= 1.0):
        raise HTTPException(422, "codeformer_fidelity must be in [0, 1]")

    # Đọc file ảnh
    try:
        img_bytes = await image.read()
    except Exception:
        raise HTTPException(400, "Cannot read uploaded file")
    if not img_bytes:
        raise HTTPException(400, "Empty image file")

    content_type = image.content_type or "image/jpeg"

    # Gọi HF endpoint
    try:
        out_bytes, meta = await call_restore_from_bytes(
            img_bytes,
            content_type=content_type,
            task=task,
            upscale=upscale,
            tile=tile,
            codeformer_fidelity=codeformer_fidelity,
            background_enhance=background_enhance,
            face_upsample=face_upsample,
        )
    except InferenceError as e:
        raise HTTPException(502, f"Inference failed: {e}")
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {e}")

    # Cache lên R2
    key = make_inference_key_from_bytes(
        task="restore",
        image_bytes=img_bytes,
        params={
            "task": task,
            "upscale": upscale,
            "tile": tile,
            "codeformer_fidelity": codeformer_fidelity,
            "background_enhance": background_enhance,
            "face_upsample": face_upsample,
        },
        ext="png",
    )
    url, key, hit = get_or_put_cached(
        data=out_bytes,
        key=key,
        content_type=meta.get("content_type", "image/png"),
        metadata={"model": "realesrgan+codeformer", "task": task},
    )
    meta.update({"r2_key": key, "cache_hit": hit, "model": "realesrgan+codeformer"})

    return RestoreResp(output_url=url, meta=meta)


# ---------------------------------------------------------------------
# THÊM ENDPOINT MỚI NÀY
# ---------------------------------------------------------------------
@app.post("/inference/remove_bg", response_model=RestoreResp)  # Tái sử dụng RestoreResp
@limiter.limit("30/minute")  # Giữ rate limit
async def remove_bg_multipart(
    request: Request,
    authorized=Depends(require_api_key),  # Giữ bảo mật
    image: UploadFile = File(..., description="Ảnh upload từ client (jpg/png/webp)"),
):
    """
    Tách nền ảnh (Background Removal) sử dụng briaai/RMBG-1.4.
    - Nhận ảnh trực tiếp từ client (multipart/form-data).
    - Cache kết quả (ảnh PNG) lên R2.
    """

    # 1. Đọc file ảnh
    try:
        img_bytes = await image.read()
    except Exception:
        raise HTTPException(400, "Cannot read uploaded file")
    if not img_bytes:
        raise HTTPException(400, "Empty image file")

    # 2. Gọi HF endpoint (hàm mới)
    try:
        out_bytes, meta = await call_remove_bg_from_bytes(img_bytes)

    except InferenceError as e:
        raise HTTPException(502, f"Inference failed: {e}")
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {e}")

    # 3. Cache lên R2
    # Key cache chỉ dựa trên hash của ảnh và tên tác vụ
    key = make_inference_key_from_bytes(
        task="remove_bg",
        image_bytes=img_bytes,
        params=None,  # Không có tham số nào khác
        ext="png",  # Output luôn là PNG
    )

    # Hàm này sẽ upload nếu chưa có, hoặc lấy link nếu đã có
    url, key, hit = get_or_put_cached(
        data=out_bytes,
        key=key,
        content_type="image/png",  # Rất quan trọng
        metadata={"model": "briaai/RMBG-1.4", "task": "remove_bg"},
    )

    meta.update(
        {"r2_key": key, "cache_hit": hit, "model": meta.get("model", "briaai/RMBG-1.4")}
    )

    # 4. Trả về response (tái sử dụng model RestoreResp)
    return RestoreResp(output_url=url, meta=meta)
