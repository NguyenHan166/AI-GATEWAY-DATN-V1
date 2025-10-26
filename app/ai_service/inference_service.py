import base64
from typing import Tuple, Dict, List, Any, Optional
import httpx
import logging
from PIL import Image
import io

from ..config import (
    HF_ENDPOINT_URL_RESTORE,
    HF_TOKEN,
    INFERENCE_TIMEOUT_SEC,
    HF_ENDPOINT_URL_REMOVE_BG,
    HF_ENDPOINT_URL_INSTRUCTPIX2PIX,
    HF_ENDPOINT_URL_ARCANE_STYLE,
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


async def call_edit_by_text_from_bytes(
    img_bytes: bytes,
    prompt: str,
    num_inference_steps: int = 20,
    image_guidance_scale: float = 1.5,
    guidance_scale: float = 7.0,
) -> tuple[bytes, dict]:
    """
    Gọi Hugging Face Endpoint cho tác vụ InstructPix2Pix.
    Handler này nhận JSON {"inputs": "base64...", "prompt": "..."}
    Và trả về JSON {"image": "base64...", "meta": {...}}
    """
    if not HF_ENDPOINT_URL_INSTRUCTPIX2PIX:
        raise InferenceError("HF_ENDPOINT_URL_INSTRUCTPIX2PIX is not configured")
    if not HF_TOKEN:
        raise InferenceError("HF_TOKEN is not configured")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    # 1. Chuyển image bytes -> base64 string
    # Chúng ta dùng key "inputs" để nhất quán với hàm remove_bg
    b64_img = base64.b64encode(img_bytes).decode("utf-8")

    # 2. Chuẩn bị payload (bao gồm ảnh và các tham số)
    payload = {
        "inputs": b64_img,  # Key "inputs" cho ảnh
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
        "image_guidance_scale": image_guidance_scale,
        "guidance_scale": guidance_scale,
    }

    # 3. Gọi API
    async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT_SEC) as client:
        try:
            response = await client.post(
                HF_ENDPOINT_URL_INSTRUCTPIX2PIX,
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
        meta["content_type"] = "image/png"  # Output luôn là PNG

        return out_bytes, meta

    except Exception as e:
        raise InferenceError(f"Failed to parse HF response: {e}")


# Thiết lập logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def call_arcane_style_from_bytes(
    img_bytes: bytes,
    prompt: str,
    strength: float = 0.75,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Gọi Hugging Face Endpoint cho tác vụ style transfer với nitrosocke/Arcane-Diffusion.
    Handler nhận JSON {"inputs": "base64...", "prompt": "...", "strength": ..., "num_inference_steps": ..., "guidance_scale": ...}
    Trả về JSON {"image": "base64...", "meta": {...}}, chuỗi base64 thô, hoặc JSON chuỗi trực tiếp.
    Args:
        img_bytes: Bytes của ảnh gốc
        prompt: Văn bản mô tả phong cách (e.g., "a character in Arcane style, vibrant colors")
        strength: Độ thay đổi phong cách [0.0-1.0]
        num_inference_steps: Số bước inference [10-50]
        guidance_scale: Độ ảnh hưởng của prompt [1.0-20.0]
    Returns:
        Tuple chứa bytes của ảnh output và metadata
    """
    if not HF_ENDPOINT_URL_ARCANE_STYLE:
        raise InferenceError("HF_ENDPOINT_URL_ARCANE_STYLE is not configured")
    if not HF_TOKEN:
        raise InferenceError("HF_TOKEN is not configured")

    # Kiểm tra ảnh đầu vào
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.verify()  # Xác minh ảnh hợp lệ
        img.close()
    except Exception as e:
        raise InferenceError(f"Invalid input image: {e}")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    # 1. Chuyển image bytes -> base64 string
    b64_img = base64.b64encode(img_bytes).decode("utf-8")

    # 2. Chuẩn bị payload
    payload = {
        "inputs": b64_img,
        "prompt": prompt,
        "strength": strength,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }

    # 3. Gọi API
    async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT_SEC) as client:
        try:
            response = await client.post(
                HF_ENDPOINT_URL_ARCANE_STYLE,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()  # Ném lỗi nếu status là 4xx/5xx

        except httpx.RequestError as e:
            logger.error(f"HTTP request failed: {e}")
            raise InferenceError(f"HTTP request failed: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HF endpoint failed: {e.response.status_code} {e.response.text[:300]}"
            )
            raise InferenceError(
                f"HF endpoint failed: {e.response.status_code} {e.response.text[:300]}"
            )

    # 4. Xử lý kết quả
    try:
        content_type = response.headers.get("content-type", "").lower()
        logger.debug(f"Response Content-Type: {content_type}")
        logger.debug(f"Response Text (first 300 chars): {response.text[:300]}")

        if "application/json" in content_type:
            data = response.json()
            if isinstance(data, str):
                # Response JSON là chuỗi base64 trực tiếp
                out_b64 = data.strip()
                meta = {}
            else:
                # Response JSON là object
                if "error" in data:
                    logger.error(f"HF Endpoint error: {data['error']}")
                    raise InferenceError(
                        f"HF Endpoint returned an error: {data['error']}"
                    )
                out_b64 = data.get("image") or data.get("outputs") or data.get("output")
                if not out_b64:
                    logger.error(
                        "No 'image', 'outputs', or 'output' key in HF JSON response"
                    )
                    raise InferenceError(
                        "No 'image', 'outputs', or 'output' key in HF JSON response"
                    )
                meta = data.get("meta", {})
        else:
            # Response là chuỗi base64 thô hoặc binary
            out_b64 = response.text.strip()
            meta = {}

        # 5. Kiểm tra và decode base64
        if not out_b64:
            logger.error("Empty base64 string received from HF endpoint")
            raise InferenceError("Empty base64 string received from HF endpoint")

        try:
            out_bytes = base64.b64decode(out_b64)
        except Exception as e:
            logger.error(f"Failed to decode base64: {e}")
            raise InferenceError(f"Failed to decode base64: {e}")

        # 6. Đảm bảo ảnh là PNG hợp lệ
        try:
            img = Image.open(io.BytesIO(out_bytes))
            img.verify()  # Xác minh ảnh hợp lệ
            img = Image.open(io.BytesIO(out_bytes))  # Mở lại để xử lý
            if img.mode != "RGB":
                img = img.convert("RGB")
            output_buffer = io.BytesIO()
            img.save(output_buffer, format="PNG")
            out_bytes = output_buffer.getvalue()
        except Exception as e:
            logger.error(f"Failed to process output image: {e}")
            raise InferenceError(f"Failed to process output image: {e}")

        # 7. Cập nhật metadata
        meta["content_type"] = "image/png"
        meta.setdefault("model", "nitrosocke/Arcane-Diffusion")
        meta.setdefault("prompt", prompt)
        meta.setdefault("strength", strength)
        meta.setdefault("num_inference_steps", num_inference_steps)
        meta.setdefault("guidance_scale", guidance_scale)

        return out_bytes, meta

    except Exception as e:
        logger.error(f"Failed to parse HF response: {e}")
        raise InferenceError(f"Failed to parse HF response: {e}")
