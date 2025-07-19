"""
Thread-safe caching utilities for Comic Translate MT.

This module provides thread-safe caching implementations to replace plain dictionary
caches throughout the application. Includes LRU cache with size bounds, persistent
cache with msgpack serialization, and a cache manager for coordination.
"""

import threading
import time
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable, Iterator, Tuple
import msgpack
import psutil
import os


class CacheError(Exception):
    """Base exception for cache-related errors."""
    pass


class CacheFullError(CacheError):
    """Raised when cache is full and cannot accept new items."""
    pass


class CacheSerializationError(CacheError):
    """Raised when cache serialization/deserialization fails."""
    pass


class ThreadSafeLRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache with size bounds and eviction policies.
    
    This cache maintains items in order of access, automatically evicting the least
    recently used items when the cache reaches its maximum size. All operations are
    thread-safe using proper locking mechanisms.
    """
    
    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        """
        Initialize the LRU cache.
        
        Args:
            max_size: Maximum number of items to store in cache
            ttl: Time-to-live in seconds for cache entries (None for no expiration)
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        
        self._max_size = max_size
        self._ttl = ttl
        self._cache: OrderedDict[Any, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
    def get(self, key: Any, default: Any = None) -> Any:
        """
        Get an item from the cache.
        
        Args:
            key: The cache key
            default: Default value if key not found
            
        Returns:
            The cached value or default if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return default
                
            value, timestamp = self._cache[key]
            
            # Check TTL expiration
            if self._ttl is not None and time.time() - timestamp > self._ttl:
                del self._cache[key]
                self._misses += 1
                return default
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value
    
    def put(self, key: Any, value: Any) -> None:
        """
        Put an item into the cache.
        
        Args:
            key: The cache key
            value: The value to cache
        """
        with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                # Update existing item
                self._cache[key] = (value, current_time)
                self._cache.move_to_end(key)
            else:
                # Add new item
                if len(self._cache) >= self._max_size:
                    # Evict least recently used item
                    self._cache.popitem(last=False)
                    self._evictions += 1
                
                self._cache[key] = (value, current_time)
    
    def __setitem__(self, key: Any, value: Any) -> None:
        """Support dict-like assignment."""
        self.put(key, value)
    
    def __getitem__(self, key: Any) -> Any:
        """Support dict-like access."""
        result = self.get(key, None)
        if result is None and key not in self._cache:
            raise KeyError(key)
        return result
    
    def __contains__(self, key: Any) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            if key not in self._cache:
                return False
            
            # Check TTL expiration
            if self._ttl is not None:
                _, timestamp = self._cache[key]
                if time.time() - timestamp > self._ttl:
                    del self._cache[key]
                    return False
            
            return True
    
    def __len__(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            self._cache.clear()
    
    def pop(self, key: Any, default: Any = None) -> Any:
        """Remove and return an item from cache."""
        with self._lock:
            if key in self._cache:
                value, _ = self._cache.pop(key)
                return value
            return default
    
    def keys(self) -> Iterator[Any]:
        """Get cache keys iterator."""
        with self._lock:
            return iter(list(self._cache.keys()))
    
    def values(self) -> Iterator[Any]:
        """Get cache values iterator."""
        with self._lock:
            return iter([value for value, _ in self._cache.values()])
    
    def items(self) -> Iterator[Tuple[Any, Any]]:
        """Get cache items iterator."""
        with self._lock:
            return iter([(key, value) for key, (value, _) in self._cache.items()])
    
    def cleanup_expired(self) -> int:
        """
        Remove expired items from cache.
        
        Returns:
            Number of items removed
        """
        if self._ttl is None:
            return 0
        
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, (_, timestamp) in self._cache.items():
                if current_time - timestamp > self._ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'hit_rate': hit_rate,
                'ttl': self._ttl
            }
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0


class PersistentCache:
    """
    Persistent cache with msgpack serialization for cross-session persistence.
    
    This cache automatically saves and loads data to/from disk using msgpack
    for efficient serialization. Supports both in-memory caching and persistence.
    """
    
    def __init__(self, cache_file: Union[str, Path], max_size: int = 1000, 
                 auto_save: bool = True, save_interval: float = 300.0):
        """
        Initialize the persistent cache.
        
        Args:
            cache_file: Path to the cache file
            max_size: Maximum number of items in memory cache
            auto_save: Whether to automatically save changes
            save_interval: Interval in seconds for automatic saves
        """
        self._cache_file = Path(cache_file)
        self._memory_cache = ThreadSafeLRUCache(max_size)
        self._auto_save = auto_save
        self._save_interval = save_interval
        self._last_save = time.time()
        self._dirty = False
        self._lock = threading.RLock()
        
        # Ensure cache directory exists
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache
        self._load_from_disk()
        
        # Start auto-save thread if enabled
        if auto_save:
            self._save_thread = threading.Thread(target=self._auto_save_worker, daemon=True)
            self._save_thread.start()
    
    def _load_from_disk(self) -> None:
        """Load cache data from disk."""
        if not self._cache_file.exists():
            return
        
        try:
            with open(self._cache_file, 'rb') as f:
                data = msgpack.unpack(f, raw=False)
                
            with self._lock:
                for key, value in data.items():
                    self._memory_cache.put(key, value)
                    
        except (OSError, msgpack.exceptions.ExtraData, 
                msgpack.exceptions.UnpackException, ValueError) as e:
            raise CacheSerializationError(f"Failed to load cache from {self._cache_file}: {e}")
    
    def _save_to_disk(self) -> None:
        """Save cache data to disk."""
        try:
            # Create temporary file for atomic write
            temp_file = self._cache_file.with_suffix('.tmp')
            
            with self._lock:
                data = dict(self._memory_cache.items())
            
            with open(temp_file, 'wb') as f:
                msgpack.pack(data, f)
            
            # Atomic rename
            temp_file.replace(self._cache_file)
            self._dirty = False
            self._last_save = time.time()
            
        except (OSError, msgpack.exceptions.PackException) as e:
            raise CacheSerializationError(f"Failed to save cache to {self._cache_file}: {e}")
    
    def _auto_save_worker(self) -> None:
        """Worker thread for automatic saving."""
        while True:
            time.sleep(self._save_interval)
            if self._dirty and time.time() - self._last_save >= self._save_interval:
                try:
                    self._save_to_disk()
                except CacheSerializationError:
                    # Log error but continue
                    pass
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Get an item from the cache."""
        return self._memory_cache.get(key, default)
    
    def put(self, key: Any, value: Any) -> None:
        """Put an item into the cache."""
        self._memory_cache.put(key, value)
        self._dirty = True
        
        if self._auto_save and time.time() - self._last_save >= self._save_interval:
            try:
                self._save_to_disk()
            except CacheSerializationError:
                # Log error but continue
                pass
    
    def __setitem__(self, key: Any, value: Any) -> None:
        """Support dict-like assignment."""
        self.put(key, value)
    
    def __getitem__(self, key: Any) -> Any:
        """Support dict-like access."""
        return self._memory_cache[key]
    
    def __contains__(self, key: Any) -> bool:
        """Check if key exists in cache."""
        return key in self._memory_cache
    
    def __len__(self) -> int:
        """Get current cache size."""
        return len(self._memory_cache)
    
    def clear(self) -> None:
        """Clear all items from cache."""
        self._memory_cache.clear()
        self._dirty = True
    
    def pop(self, key: Any, default: Any = None) -> Any:
        """Remove and return an item from cache."""
        result = self._memory_cache.pop(key, default)
        if result != default:
            self._dirty = True
        return result
    
    def keys(self) -> Iterator[Any]:
        """Get cache keys iterator."""
        return self._memory_cache.keys()
    
    def values(self) -> Iterator[Any]:
        """Get cache values iterator."""
        return self._memory_cache.values()
    
    def items(self) -> Iterator[Tuple[Any, Any]]:
        """Get cache items iterator."""
        return self._memory_cache.items()
    
    def save(self) -> None:
        """Manually save cache to disk."""
        self._save_to_disk()
    
    def reload(self) -> None:
        """Reload cache from disk."""
        self._memory_cache.clear()
        self._load_from_disk()
        self._dirty = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self._memory_cache.get_stats()
        stats.update({
            'cache_file': str(self._cache_file),
            'file_exists': self._cache_file.exists(),
            'file_size': self._cache_file.stat().st_size if self._cache_file.exists() else 0,
            'dirty': self._dirty,
            'last_save': self._last_save,
            'auto_save': self._auto_save
        })
        return stats


class CacheManager:
    """
    Cache manager to coordinate multiple cache instances and provide unified
    cache statistics, monitoring, and cleanup operations.
    """
    
    def __init__(self):
        """Initialize the cache manager."""
        self._caches: Dict[str, Union[ThreadSafeLRUCache, PersistentCache]] = {}
        self._cache_configs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._memory_monitor = MemoryMonitor()
        
        # Weak references to track cache instances
        self._cache_refs: Dict[str, weakref.ref] = {}
    
    def register_cache(self, name: str, cache: Union[ThreadSafeLRUCache, PersistentCache],
                      config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a cache instance with the manager.
        
        Args:
            name: Unique name for the cache
            cache: Cache instance to register
            config: Optional configuration for the cache
        """
        with self._lock:
            self._caches[name] = cache
            self._cache_configs[name] = config or {}
            
            # Create weak reference for cleanup tracking
            def cleanup_callback(ref):
                with self._lock:
                    self._cache_refs.pop(name, None)
            
            self._cache_refs[name] = weakref.ref(cache, cleanup_callback)
    
    def get_cache(self, name: str) -> Optional[Union[ThreadSafeLRUCache, PersistentCache]]:
        """Get a registered cache by name."""
        with self._lock:
            return self._caches.get(name)
    
    def create_lru_cache(self, name: str, max_size: int = 1000, 
                        ttl: Optional[float] = None) -> ThreadSafeLRUCache:
        """
        Create and register a new LRU cache.
        
        Args:
            name: Unique name for the cache
            max_size: Maximum cache size
            ttl: Time-to-live for cache entries
            
        Returns:
            The created cache instance
        """
        cache = ThreadSafeLRUCache(max_size=max_size, ttl=ttl)
        config = {'type': 'lru', 'max_size': max_size, 'ttl': ttl}
        self.register_cache(name, cache, config)
        return cache
    
    def create_persistent_cache(self, name: str, cache_file: Union[str, Path],
                               max_size: int = 1000, auto_save: bool = True,
                               save_interval: float = 300.0) -> PersistentCache:
        """
        Create and register a new persistent cache.
        
        Args:
            name: Unique name for the cache
            cache_file: Path to cache file
            max_size: Maximum cache size
            auto_save: Enable automatic saving
            save_interval: Save interval in seconds
            
        Returns:
            The created cache instance
        """
        cache = PersistentCache(
            cache_file=cache_file,
            max_size=max_size,
            auto_save=auto_save,
            save_interval=save_interval
        )
        config = {
            'type': 'persistent',
            'cache_file': str(cache_file),
            'max_size': max_size,
            'auto_save': auto_save,
            'save_interval': save_interval
        }
        self.register_cache(name, cache, config)
        return cache
    
    def clear_all(self) -> None:
        """Clear all registered caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
    
    def cleanup_expired(self) -> Dict[str, int]:
        """
        Clean up expired items from all caches.
        
        Returns:
            Dictionary mapping cache names to number of expired items removed
        """
        results = {}
        with self._lock:
            for name, cache in self._caches.items():
                if hasattr(cache, 'cleanup_expired'):
                    results[name] = cache.cleanup_expired()
                else:
                    results[name] = 0
        return results
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global cache statistics."""
        with self._lock:
            total_size = 0
            total_hits = 0
            total_misses = 0
            total_evictions = 0
            cache_stats = {}
            
            for name, cache in self._caches.items():
                stats = cache.get_stats()
                cache_stats[name] = stats
                
                total_size += stats.get('size', 0)
                total_hits += stats.get('hits', 0)
                total_misses += stats.get('misses', 0)
                total_evictions += stats.get('evictions', 0)
            
            total_requests = total_hits + total_misses
            global_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'cache_count': len(self._caches),
                'total_size': total_size,
                'total_hits': total_hits,
                'total_misses': total_misses,
                'total_evictions': total_evictions,
                'global_hit_rate': global_hit_rate,
                'memory_usage': self._memory_monitor.get_memory_usage(),
                'caches': cache_stats
            }
    
    def save_all_persistent(self) -> None:
        """Save all persistent caches to disk."""
        with self._lock:
            for cache in self._caches.values():
                if isinstance(cache, PersistentCache):
                    try:
                        cache.save()
                    except CacheSerializationError:
                        # Log error but continue with other caches
                        pass
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match against cache keys
            
        Returns:
            Total number of items invalidated
        """
        import fnmatch
        
        total_invalidated = 0
        with self._lock:
            for cache in self._caches.values():
                keys_to_remove = []
                for key in cache.keys():
                    if isinstance(key, str) and fnmatch.fnmatch(key, pattern):
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    cache.pop(key, None)
                    total_invalidated += 1
        
        return total_invalidated
    
    def get_cache_names(self) -> list[str]:
        """Get list of registered cache names."""
        with self._lock:
            return list(self._caches.keys())
    
    def remove_cache(self, name: str) -> bool:
        """
        Remove a cache from the manager.
        
        Args:
            name: Name of cache to remove
            
        Returns:
            True if cache was removed, False if not found
        """
        with self._lock:
            if name in self._caches:
                del self._caches[name]
                self._cache_configs.pop(name, None)
                self._cache_refs.pop(name, None)
                return True
            return False


class MemoryMonitor:
    """Monitor memory usage for cache management."""
    
    def __init__(self):
        """Initialize memory monitor."""
        self._process = psutil.Process()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with memory usage statistics in MB
        """
        try:
            memory_info = self._process.memory_info()
            memory_percent = self._process.memory_percent()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'percent': memory_percent,
                'available_mb': psutil.virtual_memory().available / 1024 / 1024
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {
                'rss_mb': 0.0,
                'vms_mb': 0.0,
                'percent': 0.0,
                'available_mb': 0.0
            }
    
    def is_memory_pressure(self, threshold_percent: float = 80.0) -> bool:
        """
        Check if system is under memory pressure.
        
        Args:
            threshold_percent: Memory usage threshold percentage
            
        Returns:
            True if memory usage is above threshold
        """
        try:
            return psutil.virtual_memory().percent > threshold_percent
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False


# Global cache manager instance
cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return cache_manager


def create_cache_key(*args, **kwargs) -> str:
    """
    Create a consistent cache key from arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        String cache key
    """
    import hashlib
    
    # Convert arguments to string representation
    key_parts = []
    
    for arg in args:
        if hasattr(arg, '__dict__'):
            # For objects, use their string representation
            key_parts.append(str(arg))
        else:
            key_parts.append(repr(arg))
    
    # Sort kwargs for consistent ordering
    for key, value in sorted(kwargs.items()):
        key_parts.append(f"{key}={repr(value)}")
    
    # Create hash of the combined key
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()


def cache_result(cache_name: str, ttl: Optional[float] = None):
    """
    Decorator to cache function results.
    
    Args:
        cache_name: Name of cache to use
        ttl: Time-to-live for cached results
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Get or create cache
            cache = cache_manager.get_cache(cache_name)
            if cache is None:
                cache = cache_manager.create_lru_cache(cache_name, ttl=ttl)
            
            # Create cache key
            cache_key = f"{func.__name__}:{create_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            return result
        
        return wrapper
    return decorator