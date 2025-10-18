import io
import os
import boto3
import hashlib
import json
from datetime import datetime, timezone
from botocore.client import Config
from urllib.parse import urljoin
from .config import (
    CF_R2_ENDPOINT,
    CF_R2_ACCESS_KEY_ID,
    CF_R2_SECRET_ACCESS_KEY,
    CF_R2_BUCKET,
    PRESIGN_EXPIRES_SECONDS,
)


# S3-compatible client → Cloudflare R2
s3 = boto3.client(
    "s3",
    endpoint_url=CF_R2_ENDPOINT,
    aws_access_key_id=CF_R2_ACCESS_KEY_ID,
    aws_secret_access_key=CF_R2_SECRET_ACCESS_KEY,
    config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
)


def list_objects(prefix: str, continuation_token: str | None = None):
    """List 1000 objects theo prefix (dùng đệ quy cho full)."""
    kwargs = {
        "Bucket": CF_R2_BUCKET,
        "Prefix": prefix,
        "MaxKeys": 1000,
    }
    if continuation_token:
        kwargs["ContinuationToken"] = continuation_token
    return s3.list_objects_v2(**kwargs)


def head_object(key: str):
    return s3.head_object(Bucket=CF_R2_BUCKET, Key=key)


def presign_get(key: str, expires: int) -> str:
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": CF_R2_BUCKET, "Key": key},
        ExpiresIn=expires,
    )


def upload_bytes_to_r2(
    data: bytes,
    *,
    key: str,
    content_type: str = "image/png",
    cache_control: str = "public, max-age=31536000, immutable",
    metadata: dict | None = None,
) -> str:
    put_kwargs = {
        "Bucket": CF_R2_BUCKET,
        "Key": key,
        "Body": io.BytesIO(data),
        "ContentType": content_type,
        "CacheControl": cache_control,
    }
    if metadata:
        put_kwargs["Metadata"] = {str(k): str(v) for k, v in metadata.items()}
    s3.put_object(**put_kwargs)
    return presign_get(key, PRESIGN_EXPIRES_SECONDS)


def make_inference_key(
    *,
    task: str,
    source_url: str,
    style: str | None,
    params: dict | None,
    ext: str = "png",
) -> str:
    payload = {"task": task, "src": source_url, "style": style, "params": params or {}}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    return f"inference/{task}/{prefix}/{style or 'default'}/{digest[:16]}.{ext}"


def get_or_put_cached(
    data: bytes,
    *,
    key: str,
    content_type: str = "image/png",
    metadata: dict | None = None,
):
    try:
        s3.head_object(Bucket=CF_R2_BUCKET, Key=key)
        return presign_get(key, PRESIGN_EXPIRES_SECONDS), key, True
    except Exception:
        url = upload_bytes_to_r2(
            data, key=key, content_type=content_type, metadata=metadata
        )
        return url, key, False
