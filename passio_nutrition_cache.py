"""Cache for Passio food nutritional data and search results to minimize API calls."""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

CACHE_FILE = Path(__file__).parent / ".passio_nutrition_cache.json"
SEARCH_CACHE_FILE = Path(__file__).parent / ".passio_search_cache.json"
CACHE_TTL_DAYS = 30  # Cache valid for 30 days
NEGATIVE_CACHE_TTL_DAYS = 7  # Negative cache expires faster (no data found)


class PassioSearchCache:
    """Persistent cache for Passio search results.

    Caches search query -> first result mapping to avoid redundant API calls.
    Common ingredients (chicken, rice, eggs, etc.) are searched repeatedly.
    """

    def __init__(self, cache_file: Path = SEARCH_CACHE_FILE):
        self.cache_file = cache_file
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent cache keys."""
        return query.lower().strip()

    def _load_cache(self):
        """Load cache from disk, filtering expired entries."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                    now = datetime.now().timestamp()
                    ttl_seconds = CACHE_TTL_DAYS * 86400
                    self._cache = {
                        k: v
                        for k, v in data.items()
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
            print(f"Warning: Could not save Passio search cache: {e}")

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached search result for a query.

        Args:
            query: Search query (e.g., "chicken breast")

        Returns:
            Dict with passio_food_id, passio_ref_code, passio_food_name,
            and nutrition data, or None if not cached.
            For negative cache entries, returns {"negative": True, "reason": ...}
        """
        key = self._normalize_query(query)
        entry = self._cache.get(key)
        if entry:
            # Check if this is a negative cache entry (no data found)
            if entry.get("negative"):
                ttl = entry.get("ttl_days", NEGATIVE_CACHE_TTL_DAYS) * 86400
                if datetime.now().timestamp() - entry.get("cached_at", 0) < ttl:
                    return {"negative": True, "reason": entry.get("reason", "no_data")}
                # Negative cache expired, return None to trigger new search
                return None
            # Normal cache entry
            return {
                "passio_food_id": entry.get("passio_food_id"),
                "passio_ref_code": entry.get("passio_ref_code"),
                "passio_food_name": entry.get("passio_food_name"),
                "protein_per_100g": entry.get("protein_per_100g"),
                "carbs_per_100g": entry.get("carbs_per_100g"),
                "fat_per_100g": entry.get("fat_per_100g"),
                "calories_per_100g": entry.get("calories_per_100g"),
            }
        return None

    def set(self, query: str, result: Dict[str, Any]):
        """Store search result in cache.

        Args:
            query: Original search query
            result: Dict with passio_food_id, passio_ref_code, passio_food_name,
                   and nutrition data (protein_per_100g, carbs_per_100g, etc.)
        """
        key = self._normalize_query(query)
        self._cache[key] = {
            "passio_food_id": result.get("passio_food_id"),
            "passio_ref_code": result.get("passio_ref_code"),
            "passio_food_name": result.get("passio_food_name"),
            "protein_per_100g": result.get("protein_per_100g"),
            "carbs_per_100g": result.get("carbs_per_100g"),
            "fat_per_100g": result.get("fat_per_100g"),
            "calories_per_100g": result.get("calories_per_100g"),
            "original_query": query,
            "cached_at": datetime.now().timestamp(),
        }
        self._save_cache()

    def set_negative(self, query: str, reason: str = "no_nutrition_data"):
        """Cache a negative result (no data found from Passio API).

        This prevents repeated API calls for items that don't have nutrition data.
        Uses a shorter TTL than positive cache entries.

        Args:
            query: Original search query
            reason: Reason for negative cache (e.g., "no_nutrition_data", "api_error")
        """
        key = self._normalize_query(query)
        self._cache[key] = {
            "negative": True,
            "reason": reason,
            "original_query": query,
            "cached_at": datetime.now().timestamp(),
            "ttl_days": NEGATIVE_CACHE_TTL_DAYS,
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

    def delete(self, query: str):
        """Delete a specific cache entry.

        Args:
            query: Search query to delete from cache
        """
        key = self._normalize_query(query)
        if key in self._cache:
            del self._cache[key]
            self._save_cache()

    def clear(self):
        """Clear all cache entries."""
        self._cache = {}
        self._save_cache()


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
                        k: v
                        for k, v in data.items()
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


# Singleton instances
_cache_instance: Optional[PassioNutritionCache] = None
_search_cache_instance: Optional[PassioSearchCache] = None


def get_passio_cache() -> PassioNutritionCache:
    """Get the singleton nutrition cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = PassioNutritionCache()
    return _cache_instance


def get_passio_search_cache() -> PassioSearchCache:
    """Get the singleton search cache instance."""
    global _search_cache_instance
    if _search_cache_instance is None:
        _search_cache_instance = PassioSearchCache()
    return _search_cache_instance
