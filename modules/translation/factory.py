import json
import hashlib
import threading
import time

from .base import TranslationEngine
from .google import GoogleTranslation
from .microsoft import MicrosoftTranslation
from .deepl import DeepLTranslation
from .yandex import YandexTranslation
from .llm.gpt import GPTTranslation
from .llm.claude import ClaudeTranslation
from .llm.gemini import GeminiTranslation
from .llm.deepseek import DeepseekTranslation
from .llm.custom import CustomTranslation

from ..utils.cache import ThreadSafeLRUCache
from ..utils.exceptions import TranslationError, handle_exception_chain
from ..utils.logging_config import get_translation_logger


class TranslationFactory:
    """Factory for creating appropriate translation engines based on settings."""

    # Thread-safe LRU cache for engine instances
    _cache = ThreadSafeLRUCache(max_size=20)
    _lock = threading.RLock()

    # Map traditional translation services to their engine classes
    TRADITIONAL_ENGINES = {
        "Google Translate": GoogleTranslation,
        "Microsoft Translator": MicrosoftTranslation,
        "DeepL": DeepLTranslation,
        "Yandex": YandexTranslation
    }

    # Map LLM identifiers to their engine classes
    LLM_ENGINE_IDENTIFIERS = {
        "GPT": GPTTranslation,
        "Claude": ClaudeTranslation,
        "Gemini": GeminiTranslation,
        "Deepseek": DeepseekTranslation,
        "Custom": CustomTranslation
    }

    # Default engines for fallback
    DEFAULT_TRADITIONAL_ENGINE = GoogleTranslation
    DEFAULT_LLM_ENGINE = GPTTranslation
    DEFAULT_TRADITIONAL_KEY = "Google Translate"
    DEFAULT_LLM_KEY = "GPT"

    @classmethod
    def register_traditional_engine(cls, key: str, engine_class: type[TranslationEngine]) -> None:
        """Register a custom traditional translation engine."""
        with cls._lock:
            cls.TRADITIONAL_ENGINES[key] = engine_class

    @classmethod
    def register_llm_engine(cls, identifier: str, engine_class: type[TranslationEngine]) -> None:
        """Register a custom LLM translation engine."""
        with cls._lock:
            cls.LLM_ENGINE_IDENTIFIERS[identifier] = engine_class

    @classmethod
    def create_engine(
        cls,
        settings,
        source_lang: str,
        target_lang: str,
        translator_key: str
    ) -> TranslationEngine:
        """
        Create or retrieve an appropriate translation engine based on settings.

        Args:
            settings: Settings object with translation configuration
            source_lang: Source language name
            target_lang: Target language name
            translator_key: Key identifying which translator to use

        Returns:
            Appropriate translation engine instance
        """
        logger = get_translation_logger()
        cache_key = cls._create_cache_key(translator_key, source_lang, target_lang, settings)

        try:
            with cls._lock:
                # Try to retrieve from cache
                try:
                    engine = cls._cache.get(cache_key)
                except Exception as e:
                    logger.error(f"Cache retrieval failed for key {cache_key}: {e}")
                    cls._cache.pop(cache_key, None)
                    engine = None

                # Validate cached engine health
                if engine and hasattr(engine, "health_check"):
                    try:
                        if not engine.health_check():
                            logger.warning(f"Health check failed for cached engine {cache_key}, evicting.")
                            cls._cache.pop(cache_key, None)
                            engine = None
                    except Exception as e:
                        logger.error(f"Error during health check for {cache_key}: {e}")
                        cls._cache.pop(cache_key, None)
                        engine = None

                if engine:
                    return engine

                # Not cached or unhealthy: initialize new engine
                engine = cls._initialize_with_retry(settings, source_lang, target_lang, translator_key, logger)

                # Cache the new engine
                cls._cache.put(cache_key, engine)
                return engine

        except TranslationError:
            # Propagate known translation errors
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            raise handle_exception_chain(e)

    @classmethod
    def _initialize_with_retry(
        cls,
        settings,
        source_lang: str,
        target_lang: str,
        translator_key: str,
        logger
    ) -> TranslationEngine:
        """
        Initialize translation engine with retries and fallback.

        Args:
            settings: Settings object
            source_lang: Source language
            target_lang: Target language
            translator_key: Primary translator key
            logger: Logger instance

        Returns:
            Initialized TranslationEngine instance
        """
        # Build list of translator keys to try: primary then fallback
        keys_to_try = [translator_key]
        if translator_key in cls.TRADITIONAL_ENGINES:
            keys_to_try.append(cls.DEFAULT_TRADITIONAL_KEY)
        else:
            keys_to_try.append(cls.DEFAULT_LLM_KEY)

        last_exception: Exception | None = None

        for key in keys_to_try:
            engine_class = cls._get_engine_class(key)
            engine = engine_class()
            # Exponential backoff delays
            delays = [1, 2, 4]
            for attempt, delay in enumerate(delays, start=1):
                try:
                    # Initialize engine
                    if key in cls.TRADITIONAL_ENGINES:
                        engine.initialize(settings, source_lang, target_lang)
                    else:
                        engine.initialize(settings, source_lang, target_lang, key)

                    # Post-initialization health check
                    if hasattr(engine, "health_check"):
                        healthy = engine.health_check()
                        if not healthy:
                            raise TranslationError(
                                "Engine health check failed",
                                engine_name=key,
                                source_language=source_lang,
                                target_language=target_lang
                            )

                    return engine
                except Exception as e:
                    tex = e if isinstance(e, TranslationError) else TranslationError(
                        f"Initialization error for engine '{key}': {e}",
                        engine_name=key,
                        source_language=source_lang,
                        target_language=target_lang,
                        cause=e
                    )
                    logger.error(f"Attempt {attempt} for engine '{key}' failed: {tex}")
                    last_exception = tex
                    if attempt < len(delays):
                        time.sleep(delay)
            logger.warning(f"All retries exhausted for engine '{key}', trying next fallback.")

        # If we reach here, initialization failed for all keys
        logger.critical("All translation engine initialization attempts failed.")
        if last_exception:
            raise last_exception
        raise TranslationError("Unable to initialize any translation engine.")

    @classmethod
    def _get_engine_class(cls, translator_key: str):
        """Get the appropriate engine class based on translator key."""
        if translator_key in cls.TRADITIONAL_ENGINES:
            return cls.TRADITIONAL_ENGINES[translator_key]
        for identifier, engine_class in cls.LLM_ENGINE_IDENTIFIERS.items():
            if identifier in translator_key:
                return engine_class
        # Default fallback
        return cls.DEFAULT_LLM_ENGINE

    @classmethod
    def _create_cache_key(
        cls,
        translator_key: str,
        source_lang: str,
        target_lang: str,
        settings
    ) -> str:
        """
        Build a cache key for all translation engines.

        - Includes per-translator credentials so changes trigger new engine.
        - For LLM engines, includes all LLM-specific settings.
        - The cache key is a hash of these dynamic values plus translator and language pair.
        """
        base = f"{translator_key}_{source_lang}_{target_lang}"

        extras: dict[str, object] = {}

        # Include credentials if present
        try:
            creds = settings.get_credentials(translator_key)
        except Exception:
            creds = None
        if creds:
            extras["credentials"] = creds

        # Include LLM-specific settings
        is_llm = any(identifier in translator_key for identifier in cls.LLM_ENGINE_IDENTIFIERS)
        if is_llm:
            try:
                extras["llm"] = settings.get_llm_settings()
            except Exception:
                # ignore if llm settings cannot be retrieved
                pass

        if not extras:
            return base

        extras_json = json.dumps(extras, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(extras_json.encode("utf-8")).hexdigest()
        return f"{base}_{digest}"