from fastapi import Header, HTTPException, Depends
import os
from .config import API_KEY


def require_api_key(x_api_key: str = Header(None), authorization: str = Header(None)):
    """
    Cho phép 2 cách gửi:
    - Header: X-API-Key: <key>
    - Header: Authorization: ApiKey <key>
    """
    key = None
    if x_api_key:
        key = x_api_key.strip()
    elif authorization and authorization.lower().startswith("apikey "):
        key = authorization.split(" ", 1)[1].strip()

    if not key or key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True
