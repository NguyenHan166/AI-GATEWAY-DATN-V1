import math
import os, time
from pathlib import PurePosixPath
from typing import Dict, List
from ..config import INDEXED_PREFIXES, ALLOWED_EXTS, MANIFEST_CACHE_TTL_SECONDS
from ..storage import list_objects, head_object

# cache đơn giản trong memory
_cache_manifest = None
_cache_time = 0.0


def _ext_of(key: str) -> str:
    p = PurePosixPath(key)
    return p.suffix.lower()


def _is_valid_file(key: str) -> bool:
    ext = _ext_of(key)
    return ext in ALLOWED_EXTS


def _iter_all_objects(prefix: str):
    token = None
    while True:
        r = list_objects(prefix, token)
        for it in r.get("Contents", []):
            yield it
        if r.get("IsTruncated"):
            token = r.get("NextContinuationToken")
        else:
            break


def build_manifest() -> Dict:
    """
    Nếu INDEXED_PREFIXES có giá trị -> chỉ crawl các prefix đó.
    Nếu rỗng -> crawl toàn bucket (prefix = "").
    Nhóm theo <Category>/<Target>/<File>.
    """
    packs: Dict[str, Dict] = {}

    prefixes = INDEXED_PREFIXES or [""]  # <— nếu rỗng, duyệt toàn bucket
    for base in prefixes:
        base_prefix = (base.strip("/") + "/") if base else ""
        for obj in _iter_all_objects(base_prefix):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if not _is_valid_file(key):
                continue

            parts = key.split("/")
            if len(parts) < 3:
                # kỳ vọng: <Category>/<Target>/<File>
                continue
            category, target = parts[0], parts[1]
            pack_id = f"{category}/{target}"
            pack_title = f"{category.replace('_',' ')} — {target.replace('_',' ')}"

            pack = packs.setdefault(
                pack_id,
                {
                    "id": pack_id,
                    "title": pack_title,
                    "category": category,
                    "target": target,
                    "files": [],
                },
            )

            size = obj.get("Size", 0)
            etag = obj.get("ETag", "").strip('"') if obj.get("ETag") else None

            pack["files"].append(
                {
                    "key": key,
                    "size": size,
                    "etag": etag,
                    "content_type": "application/octet-stream",
                }
            )

    manifest = {"version": "2025.10.0", "packs": []}
    for pack in packs.values():
        pack["count"] = len(pack["files"])
        manifest["packs"].append(pack)

    manifest["packs"].sort(key=lambda p: (p["category"], p["target"]))
    return manifest


def get_manifest_cached() -> Dict:
    global _cache_manifest, _cache_time
    now = time.time()
    if (not _cache_manifest) or (now - _cache_time > MANIFEST_CACHE_TTL_SECONDS):
        data = build_manifest()
        _cache_manifest = data
        _cache_time = now
        # log nhỏ để bạn thấy số pack
        print(f"[manifest] packs={len(data.get('packs', []))}")
    return _cache_manifest

    """
    Lưu ý:

    Mình dùng ETag như một “weak checksum” (đủ tốt cho file nhỏ 1-part như .xmp, .cube). 
    Nếu bạn muốn SHA256 chính xác, hãy chạy một script tính sha256 và lưu vào DB hoặc user-metadata của object (ví dụ x-amz-meta-sha256), 
    rồi sửa build_manifest() đọc metadata đó (gọi head_object(key) để lấy).
    """


def filter_packs(data: dict, category: str | None, target: str | None) -> list[dict]:
    """Lọc danh sách packs theo category/target (case-sensitive để khớp path)."""
    packs = data.get("packs", [])
    if category:
        packs = [p for p in packs if p.get("category") == category]
    if target:
        packs = [p for p in packs if p.get("target") == target]
    return packs


def paginate(items: list, page: int, page_size: int) -> tuple[list, int, int, int]:
    """Trả (items_page, total, page, total_pages). Page bắt đầu từ 1."""
    total = len(items)
    if page_size <= 0:
        page_size = 50
    total_pages = max(1, math.ceil(total / page_size)) if total else 1
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = start + page_size
    return items[start:end], total, page, total_pages
