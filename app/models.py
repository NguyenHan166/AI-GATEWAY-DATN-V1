from pydantic import BaseModel, Field, HttpUrl, AnyUrl
from typing import Optional, List, Dict, Literal, Any


class PresetFile(BaseModel):
    key: str
    size: int
    etag: Optional[str] = None  # md5 1-part; dùng như weak checksum
    sha256: Optional[str] = None  # nếu bạn có tính sẵn & gắn metadata
    content_type: str = "application/octet-stream"


class Pack(BaseModel):
    id: str  # ví dụ: "ON1_BW_LUTs/For_Lightroom"
    title: str  # tiêu đề đẹp hơn (render cho user)
    category: str  # "ON1_BW_LUTs"
    target: str  # "For_Lightroom"
    files: List[PresetFile]
    count: int


class Manifest(BaseModel):
    version: str = "2025.10.0"
    packs: List[Pack] = Field(default_factory=list)


class PresignReq(BaseModel):
    pack_id: str
    key: str


class PresignResp(BaseModel):
    url: str
    expires_in: int


class ManifestPaged(BaseModel):
    version: str
    packs: List[Pack] = Field(default_factory=list)
    total_packs: int
    page: int
    page_size: int
    total_pages: int


# Models cho Inference API

StyleName = Literal[
    "cinematic",
    "teal_orange",
    "vintage_film",
    "watercolor",
    "anime_soft",
    "bw_matte",
    "hdr_pop",
]


class StylizeReq(BaseModel):
    image_url: HttpUrl
    style: StyleName = "cinematic"
    strength: float = 0.30  # 0.2–0.5
    steps: int = 4  # 4–6 với Turbo
    cfg: float = 1.5  # 1.2–1.8
    negative_prompt: Optional[str] = None


class StylizeResp(BaseModel):
    output_url: HttpUrl
    meta: Dict


# ----- Restore (multipart route trả JSON này) -----
class RestoreResp(BaseModel):
    output_url: AnyUrl
    meta: Dict[str, Any]


# ----- Presign -----
class PresignReq(BaseModel):
    key: str = Field(..., description="Object key trong R2")


class PresignResp(BaseModel):
    url: AnyUrl
    expires_in: int
