import base64
from typing import Tuple, Dict, Any
import httpx

from ..config import (
    HF_ENDPOINT_URL_RESTORE,
    HF_TOKEN,
    INFERENCE_TIMEOUT_SEC,
    HF_ENDPOINT_URL_REMOVE_BG,
)


class InferenceError(RuntimeError):
    pass


async def call_restore_from_bytes(
    image_bytes: bytes,
    *,
    content_type: str = "image/jpeg",
    task: str = "upscale+face_restore",
    upscale: int = 4,
    tile: int = 0,
    codeformer_fidelity: float = 0.5,
    background_enhance: bool = True,
    face_upsample: bool = True,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Gọi Hugging Face Inference Endpoint bằng JSON:
      {
        "image": "<base64>",
        "task": "...",
        ...
      }
    Endpoint sẽ trả JSON: { "image": "<base64>", "meta": {...} }
    """
    if not HF_ENDPOINT_URL_RESTORE or not HF_TOKEN:
        raise InferenceError("HF endpoint/token for restore is not configured")

    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        # các field phổ biến mà handler custom của bạn/endpoint có thể hiểu
        "inputs": img_b64,  # hoặc "inputs": img_b64 nếu endpoint yêu cầu "inputs"
        "task": task,
        "upscale": upscale,
        "tile": tile,
        "codeformer_fidelity": codeformer_fidelity,
        "background_enhance": background_enhance,
        "face_upsample": face_upsample,
        # có thể thêm "mime": content_type nếu handler cần
    }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT_SEC) as client:
        resp = await client.post(
            HF_ENDPOINT_URL_RESTORE,
            headers=headers,
            json=payload,
        )

    if resp.status_code != 200:
        # nếu vẫn bị 400/415 thì thử đổi khóa "image" -> "inputs" (tùy handler)
        if resp.status_code in (400, 415):
            alt_payload = payload.copy()
            alt_payload.pop("image", None)
            alt_payload["inputs"] = img_b64
            async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT_SEC) as client:
                resp2 = await client.post(
                    HF_ENDPOINT_URL_RESTORE,
                    headers=headers,
                    json=alt_payload,
                )
            if resp2.status_code != 200:
                raise InferenceError(
                    f"HF restore failed: {resp2.status_code} {resp2.text[:300]}"
                )
            data = resp2.json()
        else:
            raise InferenceError(
                f"HF restore failed: {resp.status_code} {resp.text[:300]}"
            )
    else:
        data = resp.json()

    if "image" not in data and "outputs" not in data:
        raise InferenceError("HF response missing 'image'/'outputs' field")

    # một số endpoint trả "outputs": "<base64>", số khác trả "image": "<base64>"
    out_b64 = data.get("image") or data.get("outputs")
    try:
        out_bytes = base64.b64decode(out_b64)
    except Exception as e:
        raise InferenceError(f"Decode image failed: {e}")

    meta = data.get("meta", {})
    meta.setdefault("content_type", "image/png")
    meta.setdefault("task", task)
    meta.setdefault("upscale", upscale)
    meta.setdefault("tile", tile)
    meta.setdefault("codeformer_fidelity", codeformer_fidelity)
    meta.setdefault("background_enhance", background_enhance)
    meta.setdefault("face_upsample", face_upsample)
    return out_bytes, meta


# ---------------------------------------------------------------------
# THÊM HÀM MỚI NÀY
# ---------------------------------------------------------------------
async def call_remove_bg_from_bytes(img_bytes: bytes) -> tuple[bytes, dict]:
    """
    Gọi Hugging Face Endpoint cho tác vụ remove background (briaai/RMBG-1.4).
    Handler này nhận JSON {"image": "base64..."}
    Và trả về JSON {"image": "base64...", "meta": {...}}
    """
    if not HF_ENDPOINT_URL_REMOVE_BG:
        raise InferenceError("HF_ENDPOINT_URL_REMOVE_BG is not configured")
    if not HF_TOKEN:
        raise InferenceError("HF_TOKEN is not configured")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    # 1. Chuyển image bytes -> base64 string
    b64_img = base64.b64encode(img_bytes).decode("utf-8")

    # 2. Chuẩn bị payload
    payload = {"inputs": b64_img}

    # 3. Gọi API
    async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT_SEC) as client:
        try:
            response = await client.post(
                HF_ENDPOINT_URL_REMOVE_BG,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()  # Ném lỗi nếu status là 4xx/5xx

        except httpx.RequestError as e:
            raise InferenceError(f"HTTP request failed: {e}")
        except Exception as e:
            raise InferenceError(f"An unexpected error occurred: {e}")

    # 4. Xử lý kết quả
    try:
        data = response.json()
        if "error" in data:
            raise InferenceError(f"HF Endpoint returned an error: {data['error']}")

        out_b64 = data.get("image")
        if not out_b64:
            raise InferenceError("No 'image' key in HF response")

        # 5. Decode base64 (ảnh PNG) -> bytes
        out_bytes = base64.b64decode(out_b64)

        meta = data.get("meta", {})
        meta["content_type"] = "image/png"  # Output luôn là PNG (có alpha)

        return out_bytes, meta

    except Exception as e:
        raise InferenceError(f"Failed to parse HF response: {e}")
