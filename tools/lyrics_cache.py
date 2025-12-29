import json
import os
from typing import Dict, Optional, Any


class LyricsCache:
    """Simple JSON-based cache for lyrics analysis results."""

    def __init__(self, cache_file: str = "lyrics_cache.json"):
        self.cache_file = cache_file
        self.cache: Dict[str, Any] = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except IOError:
            pass

    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)

    def set(self, key: str, value: Any):
        self.cache[key] = value
        self._save_cache()

    def get_analysis(self, artist: str, title: str) -> Optional[Dict[str, Any]]:
        key = f"{artist} - {title}"
        return self.get(key)

    def set_analysis(self, artist: str, title: str, analysis: Dict[str, Any]):
        key = f"{artist} - {title}"
        self.set(key, analysis)
