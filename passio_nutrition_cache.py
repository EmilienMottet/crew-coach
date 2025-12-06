"""Cache for Passio food nutritional data to minimize API calls."""
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

CACHE_FILE = Path(__file__).parent / ".passio_nutrition_cache.json"
CACHE_TTL_DAYS = 30  # Cache valid for 30 days


class PassioNutritionCache:
    """Persistent cache for Passio food nutritional data.

    Stores nutritional data (protein, carbs, fat, calories per 100g) keyed by
    Passio ref_code. Automatically expires entries after CACHE_TTL_DAYS.
    """

    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = cache_file
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cache from disk, filtering expired entries."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                    # Filter out expired entries
                    now = datetime.now().timestamp()
                    ttl_seconds = CACHE_TTL_DAYS * 86400
                    self._cache = {
                        k: v for k, v in data.items()
                        if now - v.get("cached_at", 0) < ttl_seconds
                    }
            except (json.JSONDecodeError, IOError):
                self._cache = {}

    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self._cache, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save Passio cache: {e}")

    def get(self, ref_code: str) -> Optional[Dict[str, float]]:
        """Get nutritional data from cache by ref_code.

        Args:
            ref_code: Passio ref_code (base64 encoded)

        Returns:
            Dict with protein_per_100g, carbs_per_100g, fat_per_100g, calories_per_100g
            or None if not in cache
        """
        entry = self._cache.get(ref_code)
        if entry:
            return {
                "protein_per_100g": entry.get("protein_per_100g"),
                "carbs_per_100g": entry.get("carbs_per_100g"),
                "fat_per_100g": entry.get("fat_per_100g"),
                "calories_per_100g": entry.get("calories_per_100g"),
            }
        return None

    def set(self, ref_code: str, nutrition: Dict[str, float]):
        """Store nutritional data in cache.

        Args:
            ref_code: Passio ref_code (base64 encoded)
            nutrition: Dict with protein_per_100g, carbs_per_100g, fat_per_100g, calories_per_100g
        """
        self._cache[ref_code] = {
            **nutrition,
            "cached_at": datetime.now().timestamp(),
        }
        self._save_cache()

    def get_by_food_name(self, food_name: str) -> Optional[Dict[str, float]]:
        """Get nutritional data by food name (secondary lookup).

        Useful when ref_code changes but food name stays the same.

        Args:
            food_name: Passio food name (e.g., "Chicken Breast, raw")

        Returns:
            Dict with macros or None if not found
        """
        food_name_lower = food_name.lower()
        for entry in self._cache.values():
            if entry.get("food_name", "").lower() == food_name_lower:
                return {
                    "protein_per_100g": entry.get("protein_per_100g"),
                    "carbs_per_100g": entry.get("carbs_per_100g"),
                    "fat_per_100g": entry.get("fat_per_100g"),
                    "calories_per_100g": entry.get("calories_per_100g"),
                }
        return None

    def set_with_name(self, ref_code: str, food_name: str, nutrition: Dict[str, float]):
        """Store nutritional data with food name for secondary lookup.

        Args:
            ref_code: Passio ref_code
            food_name: Passio food name
            nutrition: Dict with macros
        """
        self._cache[ref_code] = {
            **nutrition,
            "food_name": food_name,
            "cached_at": datetime.now().timestamp(),
        }
        self._save_cache()

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        file_size = 0
        if self.cache_file.exists():
            file_size = self.cache_file.stat().st_size
        return {
            "total_entries": len(self._cache),
            "file_size_bytes": file_size,
            "file_size_kb": file_size // 1024,
            "cache_file": str(self.cache_file),
        }

    def clear(self):
        """Clear all cache entries."""
        self._cache = {}
        self._save_cache()


# Singleton instance
_cache_instance: Optional[PassioNutritionCache] = None


def get_passio_cache() -> PassioNutritionCache:
    """Get the singleton cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = PassioNutritionCache()
    return _cache_instance
