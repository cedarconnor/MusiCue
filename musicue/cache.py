from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path


def build_audio_cache_key(audio_path: Path, config_dict: dict) -> str:
    h = hashlib.sha256()
    h.update(audio_path.read_bytes())
    h.update(json.dumps(config_dict, sort_keys=True).encode())
    return h.hexdigest()


class Cache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _entry_path(self, key: str, suffix: str) -> Path:
        return self.root / key[:2] / key / suffix

    def get(self, key: str, suffix: str) -> Path | None:
        p = self._entry_path(key, suffix)
        return p if p.exists() else None

    def put(self, key: str, suffix: str, src: Path) -> Path:
        dest = self._entry_path(key, suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest

    def put_bytes(self, key: str, suffix: str, data: bytes) -> Path:
        dest = self._entry_path(key, suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest
