import hashlib
import json
import os
from typing import Any, Optional

try:
    import redis
except Exception:
    redis = None


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def cache_key(prefix: str, payload: Any) -> str:
    normalized = json.dumps(
        payload,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


class RedisCache:
    def __init__(self) -> None:
        self._client: Optional["redis.Redis"] = None
        if not _bool_env("REDIS_CACHE_ENABLED", True):
            return
        host = os.getenv("REDIS_HOST")
        if not host or redis is None:
            return
        port = _int_env("REDIS_PORT", 6379)
        db = _int_env("REDIS_DB", 0)
        password = os.getenv("REDIS_PASSWORD") or None
        ssl = _bool_env("REDIS_SSL", False)
        socket_timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT", "2"))
        self._client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            ssl=ssl,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_timeout,
            decode_responses=True,
        )

    def enabled(self) -> bool:
        return self._client is not None

    def get_json(self, key: str) -> Optional[Any]:
        if self._client is None:
            return None
        value = self._client.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        if self._client is None:
            return
        payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        if ttl_seconds and ttl_seconds > 0:
            self._client.setex(key, ttl_seconds, payload)
        else:
            self._client.set(key, payload)
