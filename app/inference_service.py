import httpx, base64
from typing import Dict, Tuple
from .config import HF_TOKEN, HF_ENDPOINT_URL_STYLIZE, INFERENCE_TIMEOUT_SEC

STYLE_PRESETS = {
    # ----- GRADE-ONLY (giữ nguyên nội dung) -----
    "cinematic": {
        "prompt": (
            "color grading only, preserve subject and composition, same objects, "
            "cinematic film grade, soft contrast, warm highlights, gentle film grain, realistic colors, "
            "no style transfer, no new elements"
        ),
        "negative": (
            "person, people, human, face, hands, extra limbs, reshaping, morphing, mutation, "
            "cartoon, anime, illustration, painting, text, logo, watermark, oversaturated, blown highlights, "
            "subject change, object replacement"
        ),
        "recommended": {"strength": 0.22, "cfg": 1.3, "steps": 4},
    },
    "teal_orange": {
        "prompt": (
            "color grading only, preserve subject and composition, same objects, "
            "teal and orange cinematic color grade, clean skin tones, crisp contrast, "
            "no new content, no stylization"
        ),
        "negative": (
            "person, human, face, extra limbs, morphing, drawing, cartoon, text, logo, watermark, haloing, artifacts, "
            "subject change, object replacement"
        ),
        "recommended": {"strength": 0.22, "cfg": 1.3, "steps": 4},
    },
    "vintage_film": {
        "prompt": (
            "color grading only, preserve subject and composition, same objects, "
            "vintage film grade, soft fade, gentle grain, pastel tones, slight vignette, "
            "no new content"
        ),
        "negative": (
            "person, human, face, extra limbs, painting, illustration, harsh sharpening, neon colors, "
            "subject change, object replacement, text, logo, watermark"
        ),
        "recommended": {"strength": 0.24, "cfg": 1.3, "steps": 4},
    },
    "bw_matte": {
        "prompt": (
            "color grading only, preserve subject and composition, same objects, "
            "rich black and white, matte look, deep shadows, fine grain, no new elements"
        ),
        "negative": (
            "color tint, posterization, halos, banding, person, face, extra limbs, cartoon, text, logo, watermark, "
            "subject change, object replacement"
        ),
        "recommended": {"strength": 0.20, "cfg": 1.2, "steps": 4},
    },
    "hdr_pop": {
        "prompt": (
            "color grading only, preserve subject and composition, same objects, "
            "natural HDR pop, enhanced dynamic range, crisp detail, vibrant yet realistic, no new content"
        ),
        "negative": (
            "haloing, overprocessed look, neon, artifacts, person, human, face, text, logo, watermark, "
            "subject change, object replacement"
        ),
        "recommended": {"strength": 0.24, "cfg": 1.4, "steps": 4},
    },
    # ----- PAINT-OVER (có thể thay đổi chất liệu) -----
    "watercolor": {
        "prompt": (
            "watercolor painting style overlay, preserve subject identity and silhouette, same composition, "
            "paper texture, soft edges, gentle wash"
        ),
        "negative": (
            "subject change, object replacement, extra limbs, distorted anatomy, heavy outlines, pixelation, text, logo, watermark"
        ),
        "recommended": {"strength": 0.38, "cfg": 1.4, "steps": 5},
    },
    "anime_soft": {
        "prompt": (
            "soft anime render overlay, preserve subject identity and silhouette, same composition, "
            "pastel palette, smooth shading, clean lines"
        ),
        "negative": (
            "photorealistic skin pores, gritty texture, subject change, object replacement, extra limbs, text, logo, watermark"
        ),
        "recommended": {"strength": 0.42, "cfg": 1.5, "steps": 5},
    },
}


HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "*/*",
}


def _clip(v, lo, hi, cast=float):
    return max(lo, min(hi, cast(v)))


def _payload_inputs_params(
    image_url: str,
    prompt: str,
    strength: float,
    steps: int,
    cfg: float,
    negative: str | None,
):
    """Schema A: {inputs:{image,prompt}, parameters:{...}}"""
    p = {
        "inputs": {
            "image": image_url,
            "prompt": prompt,
        },
        "parameters": {
            "strength": _clip(strength, 0.1, 0.8),
            "num_inference_steps": _clip(steps, 2, 12, int),
            "guidance_scale": _clip(cfg, 0.5, 3.0),
        },
    }
    if negative:
        p["parameters"]["negative_prompt"] = negative
    return p


def _payload_flat(
    image_url: str,
    prompt: str,
    strength: float,
    steps: int,
    cfg: float,
    negative: str | None,
):
    """Schema B: {image,prompt,strength,...} (một số handler custom dùng kiểu này)"""
    p = {
        "image": image_url,
        "prompt": prompt,
        "strength": _clip(strength, 0.1, 0.8),
        "num_inference_steps": _clip(steps, 2, 12, int),
        "guidance_scale": _clip(cfg, 0.5, 3.0),
    }
    if negative:
        p["negative_prompt"] = negative
    return p


def _extract_bytes(resp: httpx.Response) -> Tuple[bytes, Dict]:
    ct = (resp.headers.get("content-type") or "").lower()

    # Lỗi HTTP: ném ra detail ngắn gọn
    if resp.status_code >= 400:
        try:
            # cố gắng rút gọn JSON lỗi nếu có
            j = resp.json()
            return_text = j if isinstance(j, str) else str(j)
        except Exception:
            return_text = resp.text[:300]
        raise RuntimeError(f"HF error {resp.status_code}: {return_text[:300]}")

    # 1) image/* -> bytes
    if ct.startswith("image/") or ct == "application/octet-stream":
        return resp.content, {"content_type": ct or "image/png", "hf_json": False}

    # 2) application/json -> có thể là dict/list/str
    if "application/json" in ct:
        try:
            data = resp.json()
        except Exception:
            # fallback: parse text thủ công
            txt = resp.text.strip()
            # có thể là base64 string trần (có/không dấu ngoặc kép)
            import re

            b64_guess = re.sub(r'(^"|"$)', "", txt)
            import base64

            try:
                return base64.b64decode(b64_guess), {
                    "content_type": "image/png",
                    "hf_json": True,
                }
            except Exception:
                raise RuntimeError(f"HF JSON parse failed: {txt[:200]}")

        import base64

        # 2a) dict chứa 'image' hoặc 'images'
        if isinstance(data, dict):
            if "image" in data and isinstance(data["image"], str):
                return base64.b64decode(data["image"]), {
                    "content_type": "image/png",
                    "hf_json": True,
                }
            if "images" in data:
                imgs = data["images"]
                if isinstance(imgs, list) and imgs:
                    first = imgs[0]
                    if isinstance(first, str):
                        return base64.b64decode(first), {
                            "content_type": "image/png",
                            "hf_json": True,
                        }
                    if (
                        isinstance(first, dict)
                        and "image" in first
                        and isinstance(first["image"], str)
                    ):
                        return base64.b64decode(first["image"]), {
                            "content_type": "image/png",
                            "hf_json": True,
                        }
            # một số handler dùng 'data' hoặc 'b64'
            for k in ("data", "b64", "b64_json"):
                v = data.get(k)
                if isinstance(v, str):
                    return base64.b64decode(v), {
                        "content_type": "image/png",
                        "hf_json": True,
                    }
                if isinstance(v, list) and v and isinstance(v[0], str):
                    return base64.b64decode(v[0]), {
                        "content_type": "image/png",
                        "hf_json": True,
                    }
            raise RuntimeError(
                f"HF JSON dict but no image field: {str(list(data.keys()))[:200]}"
            )

        # 2b) list các base64 string
        if isinstance(data, list) and data and isinstance(data[0], (str, bytes)):
            b64 = (
                data[0].decode() if isinstance(data[0], (bytes, bytearray)) else data[0]
            )
            return base64.b64decode(b64), {"content_type": "image/png", "hf_json": True}

        # 2c) string base64 trần
        if isinstance(data, str):
            return base64.b64decode(data), {
                "content_type": "image/png",
                "hf_json": True,
            }

        # kiểu lạ → báo lỗi
        raise RuntimeError(f"HF JSON unsupported type: {type(data).__name__}")

    # kiểu content-type lạ → thử coi như bytes
    return resp.content, {
        "content_type": ct or "application/octet-stream",
        "hf_json": False,
    }


async def call_sdxl_turbo(
    image_url: str,
    style: str,
    strength: float,
    steps: int,
    cfg: float,
    user_neg: str | None,
) -> Tuple[bytes, Dict]:
    preset = STYLE_PRESETS[style]
    neg = user_neg or preset["negative"]
    strength = strength or preset["recommended"]["strength"]
    steps = steps or preset["recommended"]["steps"]
    cfg = cfg or preset["recommended"]["cfg"]

    async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT_SEC) as client:
        # Try Schema A (inputs/parameters) — cái endpoint của bạn vừa báo yêu cầu "inputs"
        payload_a = _payload_inputs_params(
            image_url, preset["prompt"], strength, steps, cfg, neg
        )
        r = await client.post(HF_ENDPOINT_URL_STYLIZE, headers=HEADERS, json=payload_a)
        if r.status_code == 400:
            # Fallback sang Schema B nếu handler là custom
            payload_b = _payload_flat(
                image_url, preset["prompt"], strength, steps, cfg, neg
            )
            r = await client.post(
                HF_ENDPOINT_URL_STYLIZE, headers=HEADERS, json=payload_b
            )
        return _extract_bytes(r)
