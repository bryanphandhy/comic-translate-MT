import json
import hashlib
import threading
from typing import Any, Dict, Type, Optional

from ..utils.cache import ThreadSafeLRUCache
from ..utils.exceptions import OCRError, ErrorCode, ErrorSeverity
from ..utils.logging_config import get_logger

from .base import OCREngine
from .microsoft_ocr import MicrosoftOCR
from .google_ocr import GoogleOCR
from .gpt_ocr import GPTOCR
from .paddle_ocr import PaddleOCREngine
from .manga_ocr.engine import MangaOCREngine
from .pororo.engine import PororoOCREngine
from .doctr_ocr import DocTROCR
from .gemini_ocr import GeminiOCR

logger = get_logger().bind(component="OCRFactory")


class OCRFactory:
    """Factory for creating appropriate OCR engines based on settings."""

    _lock = threading.RLock()
    _cache = ThreadSafeLRUCache(max_size=50)

    # Identifiers for LLM-based OCR engines
    LLM_ENGINE_IDENTIFIERS: Dict[str, Type[OCREngine]] = {
        "GPT": GPTOCR,
        "Gemini": GeminiOCR,
    }

    # Dynamic plugin engines registry
    plugin_engines: Dict[str, Type[OCREngine]] = {}

    @classmethod
    def register_engine(cls, name: str, engine_cls: Type[OCREngine]) -> None:
        """Register a custom OCR engine under a given name."""
        with cls._lock:
            cls.plugin_engines[name] = engine_cls
            logger.info(f"Registered custom OCR engine: {name}")

    @classmethod
    def create_engine(cls, settings: Any, source_lang_english: str, ocr_model: str) -> OCREngine:
        """
        Create or retrieve an appropriate OCR engine based on settings.
        
        Args:
            settings: Settings object with OCR configuration
            source_lang_english: Source language in English
            ocr_model: OCR model to use
        
        Returns:
            Appropriate OCR engine instance
        
        Raises:
            OCRError: If engine creation fails and fallback also fails.
        """
        # Generate cache key
        try:
            cache_key = cls._create_cache_key(ocr_model, source_lang_english, settings)
        except Exception as e:
            logger.error("Failed to generate cache key", error=str(e))
            raise OCRError(
                message=f"Cache key generation failed: {e}",
                error_code=ErrorCode.CACHE_READ_FAILED,
                severity=ErrorSeverity.ERROR,
                engine_name=ocr_model,
                language=source_lang_english,
                cause=e
            )

        # Attempt to get from cache
        with cls._lock:
            cached = cls._cache.get(cache_key, default=None)
            if cached:
                # Perform health check if supported
                healthy = True
                if hasattr(cached, "health_check"):
                    try:
                        healthy = cached.health_check()
                    except Exception as he:
                        healthy = False
                        logger.warning("Health check exception", engine=ocr_model, error=str(he))
                if healthy:
                    logger.info("Reusing cached OCR engine", engine=ocr_model, cache_key=cache_key)
                    return cached
                else:
                    logger.warning("Removing unhealthy engine from cache", engine=ocr_model)
                    cls._cache.pop(cache_key, default=None)

        # Create a new engine instance
        try:
            engine = cls._create_new_engine(settings, source_lang_english, ocr_model)
            if not isinstance(engine, OCREngine):
                raise OCRError(
                    message="Factory returned non-OCREngine instance",
                    error_code=ErrorCode.OCR_ENGINE_INIT_FAILED,
                    severity=ErrorSeverity.CRITICAL,
                    engine_name=ocr_model
                )
            # Set source language if supported
            try:
                if hasattr(engine, "set_source_language"):
                    engine.set_source_language([ ], source_lang_english)
            except Exception as le:
                logger.warning("Failed to set source language", engine=ocr_model, error=str(le))
            # Cache the engine
            with cls._lock:
                cls._cache.put(cache_key, engine)
                logger.info("Cached new OCR engine", engine=ocr_model, cache_key=cache_key)
            return engine
        except OCRError:
            # Already structured, re-raise
            raise
        except Exception as ce:
            logger.error("Error creating OCR engine", engine=ocr_model, error=str(ce))
            # Attempt graceful fallback to default
            try:
                fallback = cls._create_doctr_ocr(settings)
                with cls._lock:
                    cls._cache.put(cache_key, fallback)
                    logger.info("Cached fallback DoctrOCR engine", cache_key=cache_key)
                return fallback
            except Exception as fe:
                logger.critical("Fallback engine creation failed", engine=ocr_model, error=str(fe))
                raise OCRError(
                    message=f"Failed to create OCR engine: {fe}",
                    error_code=ErrorCode.OCR_ENGINE_INIT_FAILED,
                    severity=ErrorSeverity.CRITICAL,
                    engine_name=ocr_model,
                    language=source_lang_english,
                    cause=fe
                )

    @classmethod
    def _create_cache_key(cls, ocr_key: str, source_lang: str, settings: Any) -> str:
        """
        Build a cache key for OCR engines that includes credentials and dynamic settings.
        """
        base = f"{ocr_key}_{source_lang}"
        extras: Dict[str, Any] = {}

        # Include API credentials if available
        try:
            creds = settings.get_credentials(ocr_key)
            if creds:
                extras["credentials"] = creds
        except Exception:
            # Credentials unavailable or invalid
            pass

        # Include LLM settings if an LLM engine
        if any(identifier in ocr_key for identifier in cls.LLM_ENGINE_IDENTIFIERS):
            try:
                llm_settings = settings.get_llm_settings()
                if llm_settings:
                    extras["llm_settings"] = llm_settings
            except Exception:
                pass

        if not extras:
            return base

        extras_json = json.dumps(extras, sort_keys=True, separators=(",", ":"), default=str)
        fingerprint = hashlib.sha256(extras_json.encode("utf-8")).hexdigest()
        return f"{base}_{fingerprint}"

    @classmethod
    def _create_new_engine(cls, settings: Any, source_lang_english: str, ocr_model: str) -> OCREngine:
        """Internal: create a fresh OCR engine instance without caching."""
        # Check for dynamic plugin
        with cls._lock:
            if ocr_model in cls.plugin_engines:
                try:
                    engine_cls = cls.plugin_engines[ocr_model]
                    engine = engine_cls()
                    engine.initialize(settings=settings, model=ocr_model)
                    return engine
                except Exception as e:
                    raise OCRError(
                        message=f"Plugin engine '{ocr_model}' init failed: {e}",
                        error_code=ErrorCode.OCR_ENGINE_INIT_FAILED,
                        severity=ErrorSeverity.ERROR,
                        engine_name=ocr_model,
                        cause=e
                    )

        # Built-in model-specific factories
        try:
            if ocr_model == "Microsoft OCR":
                return cls._create_microsoft_ocr(settings)
            if ocr_model == "Google Cloud Vision":
                return cls._create_google_ocr(settings)
            if ocr_model in cls.LLM_ENGINE_IDENTIFIERS:
                return cls._create_llm_ocr(settings, ocr_model)
            # Default + language-specific
            if ocr_model == "Default":
                lang_map = {
                    "Japanese": cls._create_manga_ocr,
                    "Korean": cls._create_pororo_ocr,
                    "Chinese": cls._create_paddle_ocr,
                    "Russian": lambda s: cls._create_llm_ocr(s, "GPT-4.1-mini"),
                }
                if source_lang_english in lang_map:
                    return lang_map[source_lang_english](settings)
            # Fallback to DocTR
            return cls._create_doctr_ocr(settings)
        except OCRError:
            raise
        except Exception as e:
            raise OCRError(
                message=f"Engine initialization failed: {e}",
                error_code=ErrorCode.OCR_ENGINE_INIT_FAILED,
                severity=ErrorSeverity.ERROR,
                engine_name=ocr_model,
                language=source_lang_english,
                cause=e
            )

    @staticmethod
    def _create_microsoft_ocr(settings: Any) -> OCREngine:
        """Instantiate and initialize Microsoft OCR engine."""
        creds = settings.get_credentials(settings.ui.tr("Microsoft Azure"))
        if not creds or "api_key_ocr" not in creds or "endpoint" not in creds:
            raise OCRError(
                message="Missing Microsoft OCR credentials",
                error_code=ErrorCode.OCR_API_ERROR,
                severity=ErrorSeverity.ERROR,
                engine_name="Microsoft OCR"
            )
        engine = MicrosoftOCR()
        engine.initialize(api_key=creds["api_key_ocr"], endpoint=creds["endpoint"])
        return engine

    @staticmethod
    def _create_google_ocr(settings: Any) -> OCREngine:
        """Instantiate and initialize Google Cloud Vision OCR engine."""
        creds = settings.get_credentials(settings.ui.tr("Google Cloud"))
        if not creds or "api_key" not in creds:
            raise OCRError(
                message="Missing Google OCR API key",
                error_code=ErrorCode.OCR_API_ERROR,
                severity=ErrorSeverity.ERROR,
                engine_name="Google Cloud Vision"
            )
        engine = GoogleOCR()
        engine.initialize(api_key=creds["api_key"])
        return engine

    @staticmethod
    def _create_llm_ocr(settings: Any, model: str) -> OCREngine:
        """Instantiate and initialize an LLM-based OCR engine."""
        creds = settings.get_credentials(settings.ui.tr("Open AI GPT"))
        api_key = creds.get("api_key", "") if creds else ""
        if not api_key:
            raise OCRError(
                message="Missing LLM OCR API key",
                error_code=ErrorCode.OCR_API_ERROR,
                severity=ErrorSeverity.ERROR,
                engine_name=model
            )
        engine_cls = OCRFactory.LLM_ENGINE_IDENTIFIERS.get(model)
        if not engine_cls:
            raise OCRError(
                message=f"Unknown LLM OCR model: {model}",
                error_code=ErrorCode.OCR_INVALID_IMAGE,
                severity=ErrorSeverity.ERROR,
                engine_name=model
            )
        engine = engine_cls()
        engine.initialize(api_key=api_key, model=model)
        return engine

    @staticmethod
    def _create_manga_ocr(settings: Any) -> OCREngine:
        """Instantiate and initialize Manga OCR engine."""
        device = "cuda" if settings.is_gpu_enabled() else "cpu"
        engine = MangaOCREngine()
        engine.initialize(device=device)
        return engine

    @staticmethod
    def _create_pororo_ocr(settings: Any) -> OCREngine:
        """Instantiate and initialize Pororo OCR engine."""
        engine = PororoOCREngine()
        engine.initialize()
        return engine

    @staticmethod
    def _create_paddle_ocr(settings: Any) -> OCREngine:
        """Instantiate and initialize PaddleOCR engine."""
        engine = PaddleOCREngine()
        engine.initialize()
        return engine

    @staticmethod
    def _create_doctr_ocr(settings: Any) -> OCREngine:
        """Instantiate and initialize DocTR OCR engine."""
        device = "cuda" if settings.is_gpu_enabled() else "cpu"
        engine = DocTROCR()
        engine.initialize(device=device)
        return engine

    @staticmethod
    def _create_gemini_ocr(settings: Any, model: str) -> OCREngine:
        """Instantiate and initialize Gemini OCR engine."""
        engine = GeminiOCR()
        engine.initialize(settings, model)
        return engine