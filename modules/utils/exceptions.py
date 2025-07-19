"""
Comprehensive exception hierarchy for Comic Translate application.

This module provides a structured exception system with error codes, context preservation,
severity levels, and integration with logging systems.
"""

import traceback
from enum import Enum
from typing import Any, Dict, Optional, Union
from pathlib import Path


class ErrorSeverity(Enum):
    """Error severity levels for categorizing exceptions."""
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ErrorCode(Enum):
    """Standardized error codes for different types of failures."""
    # General errors
    UNKNOWN_ERROR = "CT_0000"
    CONFIGURATION_ERROR = "CT_0001"
    VALIDATION_ERROR = "CT_0002"
    RESOURCE_ERROR = "CT_0003"
    
    # Detection errors
    DETECTION_MODEL_LOAD_FAILED = "CT_1001"
    DETECTION_INFERENCE_FAILED = "CT_1002"
    DETECTION_INVALID_INPUT = "CT_1003"
    DETECTION_NO_TEXT_FOUND = "CT_1004"
    DETECTION_GPU_ERROR = "CT_1005"
    
    # OCR errors
    OCR_ENGINE_INIT_FAILED = "CT_2001"
    OCR_PROCESSING_FAILED = "CT_2002"
    OCR_INVALID_IMAGE = "CT_2003"
    OCR_LANGUAGE_NOT_SUPPORTED = "CT_2004"
    OCR_API_ERROR = "CT_2005"
    OCR_TIMEOUT = "CT_2006"
    
    # Translation errors
    TRANSLATION_ENGINE_INIT_FAILED = "CT_3001"
    TRANSLATION_FAILED = "CT_3002"
    TRANSLATION_API_ERROR = "CT_3003"
    TRANSLATION_QUOTA_EXCEEDED = "CT_3004"
    TRANSLATION_LANGUAGE_PAIR_NOT_SUPPORTED = "CT_3005"
    TRANSLATION_TIMEOUT = "CT_3006"
    
    # Inpainting errors
    INPAINTING_MODEL_LOAD_FAILED = "CT_4001"
    INPAINTING_PROCESSING_FAILED = "CT_4002"
    INPAINTING_INVALID_MASK = "CT_4003"
    INPAINTING_GPU_ERROR = "CT_4004"
    
    # Cache errors
    CACHE_WRITE_FAILED = "CT_5001"
    CACHE_READ_FAILED = "CT_5002"
    CACHE_CORRUPTION = "CT_5003"
    CACHE_SIZE_EXCEEDED = "CT_5004"
    CACHE_LOCK_TIMEOUT = "CT_5005"
    
    # Configuration errors
    CONFIG_FILE_NOT_FOUND = "CT_6001"
    CONFIG_INVALID_FORMAT = "CT_6002"
    CONFIG_MISSING_REQUIRED_FIELD = "CT_6003"
    CONFIG_INVALID_VALUE = "CT_6004"
    CONFIG_PERMISSION_DENIED = "CT_6005"


class ComicTranslateException(Exception):
    """
    Base exception class for all Comic Translate application errors.
    
    Provides structured error information including error codes, user-friendly messages,
    context preservation, and severity levels.
    """
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        user_message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize the exception with structured error information.
        
        Args:
            message: Technical error message for logging and debugging
            error_code: Standardized error code for programmatic handling
            severity: Error severity level
            user_message: User-friendly message for UI display
            context: Additional context information (file paths, model names, etc.)
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.error_code = error_code
        self.severity = severity
        self.user_message = user_message or self._generate_user_message()
        self.context = context or {}
        self.cause = cause
        self.timestamp = self._get_timestamp()
        
    def _generate_user_message(self) -> str:
        """Generate a user-friendly message based on the error code."""
        user_messages = {
            ErrorCode.UNKNOWN_ERROR: "An unexpected error occurred. Please try again.",
            ErrorCode.CONFIGURATION_ERROR: "Configuration error. Please check your settings.",
            ErrorCode.VALIDATION_ERROR: "Invalid input provided. Please check your data.",
            ErrorCode.RESOURCE_ERROR: "Resource unavailable. Please try again later.",
            
            # Detection errors
            ErrorCode.DETECTION_MODEL_LOAD_FAILED: "Failed to load text detection model. Please check your installation.",
            ErrorCode.DETECTION_INFERENCE_FAILED: "Text detection failed. The image may be corrupted or unsupported.",
            ErrorCode.DETECTION_INVALID_INPUT: "Invalid image provided for text detection.",
            ErrorCode.DETECTION_NO_TEXT_FOUND: "No text was detected in the image.",
            ErrorCode.DETECTION_GPU_ERROR: "GPU error during text detection. Try switching to CPU mode.",
            
            # OCR errors
            ErrorCode.OCR_ENGINE_INIT_FAILED: "Failed to initialize OCR engine. Please check your configuration.",
            ErrorCode.OCR_PROCESSING_FAILED: "OCR processing failed. The text may be unclear or in an unsupported format.",
            ErrorCode.OCR_INVALID_IMAGE: "Invalid image provided for OCR processing.",
            ErrorCode.OCR_LANGUAGE_NOT_SUPPORTED: "The selected language is not supported by the OCR engine.",
            ErrorCode.OCR_API_ERROR: "OCR service error. Please check your API credentials and try again.",
            ErrorCode.OCR_TIMEOUT: "OCR processing timed out. Please try again with a smaller image.",
            
            # Translation errors
            ErrorCode.TRANSLATION_ENGINE_INIT_FAILED: "Failed to initialize translation engine. Please check your configuration.",
            ErrorCode.TRANSLATION_FAILED: "Translation failed. Please try again or use a different translation service.",
            ErrorCode.TRANSLATION_API_ERROR: "Translation service error. Please check your API credentials.",
            ErrorCode.TRANSLATION_QUOTA_EXCEEDED: "Translation quota exceeded. Please check your API limits.",
            ErrorCode.TRANSLATION_LANGUAGE_PAIR_NOT_SUPPORTED: "The selected language pair is not supported.",
            ErrorCode.TRANSLATION_TIMEOUT: "Translation timed out. Please try again.",
            
            # Inpainting errors
            ErrorCode.INPAINTING_MODEL_LOAD_FAILED: "Failed to load inpainting model. Please check your installation.",
            ErrorCode.INPAINTING_PROCESSING_FAILED: "Inpainting failed. The image may be too complex or corrupted.",
            ErrorCode.INPAINTING_INVALID_MASK: "Invalid mask provided for inpainting.",
            ErrorCode.INPAINTING_GPU_ERROR: "GPU error during inpainting. Try switching to CPU mode.",
            
            # Cache errors
            ErrorCode.CACHE_WRITE_FAILED: "Failed to write to cache. Please check disk space and permissions.",
            ErrorCode.CACHE_READ_FAILED: "Failed to read from cache. Cache may be corrupted.",
            ErrorCode.CACHE_CORRUPTION: "Cache corruption detected. Cache will be rebuilt.",
            ErrorCode.CACHE_SIZE_EXCEEDED: "Cache size limit exceeded. Old entries will be removed.",
            ErrorCode.CACHE_LOCK_TIMEOUT: "Cache operation timed out. Please try again.",
            
            # Configuration errors
            ErrorCode.CONFIG_FILE_NOT_FOUND: "Configuration file not found. Using default settings.",
            ErrorCode.CONFIG_INVALID_FORMAT: "Invalid configuration file format. Please check the file.",
            ErrorCode.CONFIG_MISSING_REQUIRED_FIELD: "Required configuration field is missing.",
            ErrorCode.CONFIG_INVALID_VALUE: "Invalid configuration value provided.",
            ErrorCode.CONFIG_PERMISSION_DENIED: "Permission denied accessing configuration file.",
        }
        
        return user_messages.get(self.error_code, "An error occurred. Please try again.")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for error tracking."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def add_context(self, key: str, value: Any) -> 'ComicTranslateException':
        """Add context information to the exception."""
        self.context[key] = value
        return self
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context information from the exception."""
        return self.context.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and serialization."""
        return {
            'error_code': self.error_code.value,
            'severity': self.severity.value,
            'message': str(self),
            'user_message': self.user_message,
            'context': self.context,
            'timestamp': self.timestamp,
            'traceback': traceback.format_exc() if self.__traceback__ else None,
            'cause': str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        """String representation including error code and context."""
        context_str = ""
        if self.context:
            context_items = [f"{k}={v}" for k, v in self.context.items()]
            context_str = f" [{', '.join(context_items)}]"
        
        return f"[{self.error_code.value}] {super().__str__()}{context_str}"


class DetectionError(ComicTranslateException):
    """Exception for text detection related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.DETECTION_INFERENCE_FAILED,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        user_message: Optional[str] = None,
        model_name: Optional[str] = None,
        image_path: Optional[Union[str, Path]] = None,
        image_size: Optional[tuple] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize detection error with relevant context.
        
        Args:
            message: Technical error message
            error_code: Specific detection error code
            severity: Error severity level
            user_message: User-friendly message
            model_name: Name of the detection model
            image_path: Path to the image being processed
            image_size: Size of the image (width, height)
            context: Additional context information
            cause: Original exception
        """
        context = context or {}
        if model_name:
            context['model_name'] = model_name
        if image_path:
            context['image_path'] = str(image_path)
        if image_size:
            context['image_size'] = image_size
            
        super().__init__(message, error_code, severity, user_message, context, cause)


class OCRError(ComicTranslateException):
    """Exception for OCR processing related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.OCR_PROCESSING_FAILED,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        user_message: Optional[str] = None,
        engine_name: Optional[str] = None,
        language: Optional[str] = None,
        image_path: Optional[Union[str, Path]] = None,
        text_block_count: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize OCR error with relevant context.
        
        Args:
            message: Technical error message
            error_code: Specific OCR error code
            severity: Error severity level
            user_message: User-friendly message
            engine_name: Name of the OCR engine
            language: Language being processed
            image_path: Path to the image being processed
            text_block_count: Number of text blocks being processed
            context: Additional context information
            cause: Original exception
        """
        context = context or {}
        if engine_name:
            context['engine_name'] = engine_name
        if language:
            context['language'] = language
        if image_path:
            context['image_path'] = str(image_path)
        if text_block_count is not None:
            context['text_block_count'] = text_block_count
            
        super().__init__(message, error_code, severity, user_message, context, cause)


class TranslationError(ComicTranslateException):
    """Exception for translation related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.TRANSLATION_FAILED,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        user_message: Optional[str] = None,
        engine_name: Optional[str] = None,
        source_language: Optional[str] = None,
        target_language: Optional[str] = None,
        text_length: Optional[int] = None,
        api_endpoint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize translation error with relevant context.
        
        Args:
            message: Technical error message
            error_code: Specific translation error code
            severity: Error severity level
            user_message: User-friendly message
            engine_name: Name of the translation engine
            source_language: Source language code
            target_language: Target language code
            text_length: Length of text being translated
            api_endpoint: API endpoint being used
            context: Additional context information
            cause: Original exception
        """
        context = context or {}
        if engine_name:
            context['engine_name'] = engine_name
        if source_language:
            context['source_language'] = source_language
        if target_language:
            context['target_language'] = target_language
        if text_length is not None:
            context['text_length'] = text_length
        if api_endpoint:
            context['api_endpoint'] = api_endpoint
            
        super().__init__(message, error_code, severity, user_message, context, cause)


class InpaintingError(ComicTranslateException):
    """Exception for inpainting related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.INPAINTING_PROCESSING_FAILED,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        user_message: Optional[str] = None,
        model_name: Optional[str] = None,
        image_path: Optional[Union[str, Path]] = None,
        mask_path: Optional[Union[str, Path]] = None,
        image_size: Optional[tuple] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize inpainting error with relevant context.
        
        Args:
            message: Technical error message
            error_code: Specific inpainting error code
            severity: Error severity level
            user_message: User-friendly message
            model_name: Name of the inpainting model
            image_path: Path to the image being processed
            mask_path: Path to the mask being used
            image_size: Size of the image (width, height)
            context: Additional context information
            cause: Original exception
        """
        context = context or {}
        if model_name:
            context['model_name'] = model_name
        if image_path:
            context['image_path'] = str(image_path)
        if mask_path:
            context['mask_path'] = str(mask_path)
        if image_size:
            context['image_size'] = image_size
            
        super().__init__(message, error_code, severity, user_message, context, cause)


class CacheError(ComicTranslateException):
    """Exception for cache related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.CACHE_WRITE_FAILED,
        severity: ErrorSeverity = ErrorSeverity.WARNING,
        user_message: Optional[str] = None,
        cache_type: Optional[str] = None,
        cache_key: Optional[str] = None,
        cache_size: Optional[int] = None,
        cache_path: Optional[Union[str, Path]] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize cache error with relevant context.
        
        Args:
            message: Technical error message
            error_code: Specific cache error code
            severity: Error severity level
            user_message: User-friendly message
            cache_type: Type of cache (LRU, persistent, etc.)
            cache_key: Cache key that caused the error
            cache_size: Current cache size
            cache_path: Path to cache file/directory
            context: Additional context information
            cause: Original exception
        """
        context = context or {}
        if cache_type:
            context['cache_type'] = cache_type
        if cache_key:
            context['cache_key'] = cache_key
        if cache_size is not None:
            context['cache_size'] = cache_size
        if cache_path:
            context['cache_path'] = str(cache_path)
            
        super().__init__(message, error_code, severity, user_message, context, cause)


class ConfigurationError(ComicTranslateException):
    """Exception for configuration related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.CONFIGURATION_ERROR,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        user_message: Optional[str] = None,
        config_file: Optional[Union[str, Path]] = None,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        expected_type: Optional[type] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize configuration error with relevant context.
        
        Args:
            message: Technical error message
            error_code: Specific configuration error code
            severity: Error severity level
            user_message: User-friendly message
            config_file: Path to configuration file
            config_key: Configuration key that caused the error
            config_value: Invalid configuration value
            expected_type: Expected type for the configuration value
            context: Additional context information
            cause: Original exception
        """
        context = context or {}
        if config_file:
            context['config_file'] = str(config_file)
        if config_key:
            context['config_key'] = config_key
        if config_value is not None:
            context['config_value'] = str(config_value)
        if expected_type:
            context['expected_type'] = expected_type.__name__
            
        super().__init__(message, error_code, severity, user_message, context, cause)


# Helper functions for error reporting and context extraction

def create_error_report(exception: ComicTranslateException) -> Dict[str, Any]:
    """
    Create a comprehensive error report from an exception.
    
    Args:
        exception: The exception to create a report for
        
    Returns:
        Dictionary containing structured error information
    """
    report = exception.to_dict()
    
    # Add additional diagnostic information
    report['exception_type'] = type(exception).__name__
    report['python_version'] = __import__('sys').version
    
    # Add stack trace if available
    if exception.__traceback__:
        import traceback
        report['full_traceback'] = traceback.format_exception(
            type(exception), exception, exception.__traceback__
        )
    
    return report


def extract_user_context(exception: ComicTranslateException) -> Dict[str, str]:
    """
    Extract user-relevant context from an exception for UI display.
    
    Args:
        exception: The exception to extract context from
        
    Returns:
        Dictionary containing user-friendly context information
    """
    user_context = {}
    
    # Map technical context keys to user-friendly labels
    context_mapping = {
        'image_path': 'File',
        'model_name': 'Model',
        'engine_name': 'Engine',
        'language': 'Language',
        'source_language': 'From',
        'target_language': 'To',
        'config_file': 'Configuration File',
        'cache_type': 'Cache Type'
    }
    
    for tech_key, user_label in context_mapping.items():
        if tech_key in exception.context:
            value = exception.context[tech_key]
            # Format file paths to show only filename
            if 'path' in tech_key or 'file' in tech_key:
                value = Path(value).name if value else value
            user_context[user_label] = str(value)
    
    return user_context


def log_exception(exception: ComicTranslateException, logger=None):
    """
    Log an exception with appropriate level based on severity.
    
    Args:
        exception: The exception to log
        logger: Logger instance (optional, will use default if not provided)
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    # Create log message with context
    log_message = f"{exception.error_code.value}: {str(exception)}"
    
    # Add context information
    if exception.context:
        context_str = ", ".join(f"{k}={v}" for k, v in exception.context.items())
        log_message += f" | Context: {context_str}"
    
    # Log with appropriate level
    if exception.severity == ErrorSeverity.CRITICAL:
        logger.critical(log_message, exc_info=exception)
    elif exception.severity == ErrorSeverity.ERROR:
        logger.error(log_message, exc_info=exception)
    else:  # WARNING
        logger.warning(log_message)


def handle_exception_chain(exception: Exception) -> ComicTranslateException:
    """
    Convert a generic exception to a ComicTranslateException, preserving the chain.
    
    Args:
        exception: The original exception
        
    Returns:
        ComicTranslateException with preserved context
    """
    if isinstance(exception, ComicTranslateException):
        return exception
    
    # Try to infer the appropriate exception type based on the original exception
    exception_type = type(exception).__name__.lower()
    
    if 'config' in exception_type or 'setting' in exception_type:
        return ConfigurationError(
            f"Configuration error: {str(exception)}",
            cause=exception
        )
    elif 'cache' in exception_type or 'memory' in exception_type:
        return CacheError(
            f"Cache error: {str(exception)}",
            cause=exception
        )
    elif 'ocr' in exception_type or 'text' in exception_type:
        return OCRError(
            f"OCR error: {str(exception)}",
            cause=exception
        )
    elif 'translation' in exception_type or 'translate' in exception_type:
        return TranslationError(
            f"Translation error: {str(exception)}",
            cause=exception
        )
    elif 'detection' in exception_type or 'detect' in exception_type:
        return DetectionError(
            f"Detection error: {str(exception)}",
            cause=exception
        )
    else:
        return ComicTranslateException(
            f"Unexpected error: {str(exception)}",
            cause=exception
        )