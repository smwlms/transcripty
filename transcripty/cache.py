"""Thread-safe model cache with FIFO eviction."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ModelCache:
    """Thread-safe cache for heavy model objects with FIFO eviction.

    When the cache exceeds max_size, the oldest inserted entry is evicted.
    """

    def __init__(self, name: str, max_size: int = 2) -> None:
        self._name = name
        self._cache: dict[str, object] = {}
        self._lock = threading.Lock()
        self.max_size = max_size

    def get_or_load(self, key: str, loader: Callable[[], T]) -> T:
        """Get a cached model or load it using the provided factory.

        Args:
            key: Cache key (e.g. "small:int8:cpu").
            loader: Callable that creates the model if not cached.

        Returns:
            The cached or newly loaded model.
        """
        with self._lock:
            if key in self._cache:
                logger.debug("Using cached %s '%s'", self._name, key)
                return self._cache[key]  # type: ignore[return-value]

            # Evict oldest entry (FIFO) if cache is full
            if len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                logger.info("Evicting %s '%s' from cache", self._name, oldest_key)
                del self._cache[oldest_key]

            logger.info("Loading %s '%s'...", self._name, key)
            model = loader()
            self._cache[key] = model
            return model

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
        logger.info("%s cache cleared", self._name)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache
