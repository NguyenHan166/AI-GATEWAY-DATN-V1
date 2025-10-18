from fastapi import FastAPI, HTTPException, Depends, Request
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
)
from .storage import presign_get, get_or_put_cached, make_inference_key
from .security import require_api_key
from .manifest_service import get_manifest_cached
from .inference_service import call_sdxl_turbo

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


@app.get("/manifest", response_model=Manifest)
@limiter.limit("30/minute")
async def get_manifest(request: Request, authorized=Depends(require_api_key)):
    data = get_manifest_cached()
    return Manifest(**data)


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


# ---- Inference: stylize ----
@app.post("/inference/stylize", response_model=StylizeResp)
@limiter.limit("30/minute")
async def stylize(
    request: Request, req: StylizeReq, authorized=Depends(require_api_key)
):
    # Giới hạn hợp lý: phía client nên resize về <= STYLE_MAX_SIDE trước khi upload R2
    # (Ở đây ta không tải ảnh về để kiểm; tin client để tiết kiệm tài nguyên.)

    img_bytes, meta = await call_sdxl_turbo(
        str(req.image_url),
        req.style,
        req.strength,
        req.steps,
        req.cfg,
        req.negative_prompt,
    )

    key = make_inference_key(
        task="stylize",
        source_url=str(req.image_url),
        style=req.style,
        params={"strength": req.strength, "steps": req.steps, "cfg": req.cfg},
        ext="png",
    )
    url, key, hit = get_or_put_cached(
        data=img_bytes,
        key=key,
        content_type=meta.get("content_type", "image/png"),
        metadata={"model": "sdxl-turbo", "style": req.style},
    )
    meta.update(
        {"r2_key": key, "cache_hit": hit, "model": "sdxl-turbo", "style": req.style}
    )
    return StylizeResp(output_url=url, meta=meta)
