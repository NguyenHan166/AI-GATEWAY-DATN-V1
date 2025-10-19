import base64
from typing import Tuple, Dict, Any
import httpx

from ..config import HF_ENDPOINT_URL_RESTORE, HF_TOKEN, INFERENCE_TIMEOUT_SEC


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
