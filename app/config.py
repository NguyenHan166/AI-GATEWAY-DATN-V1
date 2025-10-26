from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(), encoding="utf-8")


def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


# Cloudflare R2
CF_R2_ACCOUNT_ID = require_env("CF_R2_ACCOUNT_ID")
CF_R2_BUCKET = require_env("CF_R2_BUCKET")
CF_R2_ACCESS_KEY_ID = require_env("CF_R2_ACCESS_KEY_ID")
CF_R2_SECRET_ACCESS_KEY = require_env("CF_R2_SECRET_ACCESS_KEY")
CF_R2_ENDPOINT = (
    os.getenv("CF_R2_ENDPOINT")
    or f"https://{CF_R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
)

# API
API_KEY = os.getenv("API_KEY", "").strip()
ALLOWED_ORIGINS = [s.strip() for s in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

INDEXED_PREFIXES = [
    s.strip().strip("/")
    for s in os.getenv("INDEXED_PREFIXES", "").split(",")
    if s.strip()
]
MANIFEST_CACHE_TTL_SECONDS = int(os.getenv("MANIFEST_CACHE_TTL_SECONDS", "300"))
PRESIGN_EXPIRES_SECONDS = int(os.getenv("PRESIGN_EXPIRES_SECONDS", "900"))
STYLE_MAX_SIDE = int(os.getenv("STYLE_MAX_SIDE", "1024"))

# Hugging Face
HF_ENDPOINT_URL_RESTORE = os.getenv("HF_ENDPOINT_URL_RESTORE")
# Thêm dòng này cùng với các HF_ENDPOINT khác
HF_ENDPOINT_URL_REMOVE_BG = os.getenv("HF_ENDPOINT_URL_REMOVE_BG")
HF_ENDPOINT_URL_INSTRUCTPIX2PIX = os.getenv("HF_ENDPOINT_URL_INSTRUCTPIX2PIX")
# config.py (bổ sung vào phần Hugging Face)
# đã có: HF_ENDPOINT_URL_INSTRUCTPIX2PIX, HF_TOKEN, INFERENCE_TIMEOUT_SEC, ...
HF_ENDPOINT_URL_ARCANE_STYLE = os.getenv("HF_ENDPOINT_URL_ARCANE_STYLE")

HF_TOKEN = os.getenv("HF_TOKEN")
INFERENCE_TIMEOUT_SEC = int(os.getenv("INFERENCE_TIMEOUT_SEC", "90"))

# an toàn: các extension hợp lệ
ALLOWED_EXTS = {".xmp", ".cube", ".onpreset", ".onpreset.zip"}

# Shared key (HMAC)
SHARED_KEY = require_env("SHARED_KEY")
