import threading
import importlib
try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

from .base import DetectionEngine
from .rtdetr_v2 import RTDetrV2Detection
from ..utils.cache import get_cache_manager, ThreadSafeLRUCache
from ..utils.exceptions import DetectionError
from ..utils.logging_config import get_detection_logger


class DetectionEngineFactory:
    """Factory for creating appropriate detection engines based on settings."""

    _lock = threading.RLock()
    _engine_factories = {}
    _initialized = False
    _default_model = 'RT-DETR-v2'
    _cache_name = 'detection_engines'

    @classmethod
    def register_engine(cls, model_name: str, factory_method):
        """
        Register a custom detection engine factory method.
        Args:
            model_name: Unique name for the detection model
            factory_method: Callable(settings) -> DetectionEngine
        """
        with cls._lock:
            cls._engine_factories[model_name] = factory_method

    @classmethod
    def discover_plugins(cls):
        """
        Discover and register detection engine plugins via entry points.
        Entry point group: 'comic_translate.detection_engines'
        """
        with cls._lock:
            try:
                entry_points = importlib_metadata.entry_points()
                # For importlib_metadata >= 3.10
                if hasattr(entry_points, "select"):
                    eps = entry_points.select(group='comic_translate.detection_engines')
                else:
                    eps = entry_points.get('comic_translate.detection_engines', [])
                for ep in eps:
                    try:
                        factory = ep.load()
                        cls._engine_factories[ep.name] = factory
                        get_detection_logger().info(
                            f"Discovered detection plugin: {ep.name}"
                        )
                    except Exception as e:
                        get_detection_logger().warning(
                            f"Failed to load detection plugin '{ep.name}'", exception=e
                        )
            except Exception as e:
                # Plugin discovery non-fatal
                get_detection_logger().warning(
                    "Plugin discovery failed for detection engines", exception=e
                )

    @classmethod
    def _ensure_factories_initialized(cls):
        """Initialize builtin factories and discover plugins once."""
        with cls._lock:
            if not cls._initialized:
                # Register built-in engines
                cls._engine_factories[cls._default_model] = cls._create_rtdetr_v2
                # Discover external plugins
                cls.discover_plugins()
                cls._initialized = True

    @classmethod
    def create_engine(cls, settings, model_name: str = None) -> DetectionEngine:
        """
        Create or retrieve an appropriate detection engine.
        
        Args:
            settings: Settings object with detection configuration
            model_name: Name of the detection model to use
        Returns:
            DetectionEngine instance
        Raises:
            DetectionError on failure
        """
        model = model_name or cls._default_model
        logger = get_detection_logger()

        with cls._lock:
            cls._ensure_factories_initialized()

            # Initialize or get cache
            cache_manager = get_cache_manager()
            cache = cache_manager.get_cache(cls._cache_name)
            max_size = getattr(settings, 'detection_cache_max_engines', 5)
            if cache is None:
                cache = cache_manager.create_lru_cache(
                    cls._cache_name, max_size=max_size
                )

            # Try to retrieve from cache
            try:
                engine = cache.get(model)
            except Exception:
                engine = None

            if engine:
                # Health check
                try:
                    if hasattr(engine, 'health_check') and not engine.health_check():
                        raise DetectionError(
                            f"Health check failed for engine '{model}'",
                            model_name=model
                        )
                    logger.debug(f"Using cached detection engine", model_name=model)
                    return engine
                except DetectionError as de:
                    logger.warning(
                        f"Cached engine '{model}' failed health check, recreating",
                        exception=de, model_name=model
                    )
                    cache.pop(model)
                    engine = None

            # Create new engine with retry and fallback
            factory = cls._engine_factories.get(model)
            if factory is None:
                logger.warning(
                    f"No factory found for model '{model}', falling back to default",
                    model_name=model
                )
                model = cls._default_model
                factory = cls._engine_factories.get(model)
            last_exception = None
            engine = None
            for attempt in range(2):
                try:
                    engine = factory(settings)
                    # Post-init health check
                    if hasattr(engine, 'health_check') and not engine.health_check():
                        raise DetectionError(
                            f"Post-initialization health check failed for '{model}'",
                            model_name=model
                        )
                    logger.info(f"Initialized detection engine '{model}'", model_name=model)
                    break
                except DetectionError as de:
                    last_exception = de
                    logger.error(
                        f"Attempt {attempt+1} failed initializing '{model}'",
                        exception=de, model_name=model
                    )
                    if model != cls._default_model:
                        logger.info(
                            f"Falling back to default model '{cls._default_model}'"
                        )
                        model = cls._default_model
                        factory = cls._engine_factories.get(model)
                    else:
                        break
                except Exception as e:
                    last_exception = e
                    logger.error(
                        f"Unexpected error initializing '{model}'",
                        exception=e, model_name=model
                    )
                    if model != cls._default_model:
                        logger.info(
                            f"Falling back to default model '{cls._default_model}'"
                        )
                        model = cls._default_model
                        factory = cls._engine_factories.get(model)
                    else:
                        break

            if engine is None:
                raise DetectionError(
                    f"Unable to initialize detection engine for model '{model}'",
                    cause=last_exception,
                    model_name=model
                )

            # Evict LRU engines if cache is full
            try:
                while len(cache) >= cache._max_size:
                    lru_key = next(iter(cache.keys()))
                    lru_engine = cache.pop(lru_key)
                    logger.info(
                        f"Evicting least-recently-used engine '{lru_key}'",
                        evicted_model=lru_key
                    )
                    if hasattr(lru_engine, 'cleanup_resources'):
                        try:
                            lru_engine.cleanup_resources()
                        except Exception as e:
                            logger.warning(
                                f"Error cleaning up resources for '{lru_key}'",
                                exception=e, evicted_model=lru_key
                            )
            except Exception:
                # Non-fatal eviction errors
                pass

            # Cache the newly created engine
            cache.put(model, engine)
            logger.debug(f"Cached detection engine '{model}'", model_name=model)
            return engine

    @staticmethod
    def _create_rtdetr_v2(settings):
        """Create and initialize RT-DETR-V2 detection engine."""
        logger = get_detection_logger()
        engine = RTDetrV2Detection()
        device = 'cuda' if getattr(settings, 'is_gpu_enabled', lambda: False)() else 'cpu'
        try:
            engine.initialize(device=device)
        except Exception as e:
            raise DetectionError(
                f"RT-DETR-v2 initialization failed on device '{device}': {e}",
                model_name='RT-DETR-v2',
                cause=e
            )
        logger.debug("RT-DETR-v2 engine initialized", model_name='RT-DETR-v2', device=device)
        return engine