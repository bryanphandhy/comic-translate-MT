"""
Structured logging configuration for Comic Translate MT.

This module provides a comprehensive logging system using loguru to replace
scattered print statements throughout the codebase. It includes context-aware
logging, performance monitoring, and log analysis utilities.
"""

import sys
import os
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import traceback

try:
    from loguru import logger
except ImportError:
    raise ImportError("loguru is required for logging. Install with: pip install loguru")


@dataclass
class LogContext:
    """Context information for structured logging."""
    component: str = ""
    operation: str = ""
    image_path: str = ""
    model_name: str = ""
    language: str = ""
    engine_type: str = ""
    batch_id: str = ""
    user_id: str = ""
    session_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging."""
        context = {
            "component": self.component,
            "operation": self.operation,
            "image_path": self.image_path,
            "model_name": self.model_name,
            "language": self.language,
            "engine_type": self.engine_type,
            "batch_id": self.batch_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }
        # Add extra fields
        context.update(self.extra)
        # Remove empty values
        return {k: v for k, v in context.items() if v}


@dataclass
class PerformanceMetrics:
    """Performance metrics for logging."""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_before: Optional[int] = None
    memory_after: Optional[int] = None
    memory_peak: Optional[int] = None
    cache_hits: int = 0
    cache_misses: int = 0
    items_processed: int = 0
    errors_count: int = 0

    def finish(self):
        """Mark the end of the operation and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "duration_ms": round(self.duration * 1000, 2) if self.duration else None,
            "memory_before_mb": round(self.memory_before / 1024 / 1024, 2) if self.memory_before else None,
            "memory_after_mb": round(self.memory_after / 1024 / 1024, 2) if self.memory_after else None,
            "memory_peak_mb": round(self.memory_peak / 1024 / 1024, 2) if self.memory_peak else None,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hits / (self.cache_hits + self.cache_misses), 3) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "items_processed": self.items_processed,
            "errors_count": self.errors_count,
            "throughput_items_per_sec": round(self.items_processed / self.duration, 2) if self.duration and self.duration > 0 else 0,
        }


class ContextualLogger:
    """Thread-safe contextual logger with structured logging capabilities."""
    
    def __init__(self):
        self._local = threading.local()
        self._session_id = self._generate_session_id()
        
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    
    @property
    def context(self) -> LogContext:
        """Get the current thread's log context."""
        if not hasattr(self._local, 'context'):
            self._local.context = LogContext(session_id=self._session_id)
        return self._local.context
    
    @context.setter
    def context(self, value: LogContext):
        """Set the current thread's log context."""
        if not hasattr(self._local, 'context'):
            self._local.context = LogContext(session_id=self._session_id)
        self._local.context = value
        if not value.session_id:
            value.session_id = self._session_id
    
    def _format_message(self, message: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format message with context information."""
        log_data = {
            "message": message,
            "context": self.context.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "thread_id": threading.get_ident(),
        }
        
        if extra:
            log_data["extra"] = extra
            
        return log_data
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        log_data = self._format_message(message, kwargs)
        logger.debug(json.dumps(log_data, ensure_ascii=False))
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        log_data = self._format_message(message, kwargs)
        logger.info(json.dumps(log_data, ensure_ascii=False))
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        log_data = self._format_message(message, kwargs)
        logger.warning(json.dumps(log_data, ensure_ascii=False))
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with context and optional exception."""
        log_data = self._format_message(message, kwargs)
        
        if exception:
            log_data["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc(),
            }
        
        logger.error(json.dumps(log_data, ensure_ascii=False))
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message with context and optional exception."""
        log_data = self._format_message(message, kwargs)
        
        if exception:
            log_data["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc(),
            }
        
        logger.critical(json.dumps(log_data, ensure_ascii=False))
    
    def performance(self, message: str, metrics: PerformanceMetrics, **kwargs):
        """Log performance metrics with context."""
        log_data = self._format_message(message, kwargs)
        log_data["performance"] = metrics.to_dict()
        logger.info(json.dumps(log_data, ensure_ascii=False))


class LoggingConfig:
    """Main logging configuration class."""
    
    def __init__(self, 
                 log_dir: Optional[Union[str, Path]] = None,
                 log_level: str = "INFO",
                 max_file_size: str = "10 MB",
                 retention: str = "30 days",
                 compression: str = "gz",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 structured_format: bool = True):
        """
        Initialize logging configuration.
        
        Args:
            log_dir: Directory for log files (default: ./logs)
            log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            max_file_size: Maximum size per log file
            retention: How long to keep log files
            compression: Compression format for rotated logs
            enable_console: Whether to log to console
            enable_file: Whether to log to files
            structured_format: Whether to use structured JSON format
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_level = log_level.upper()
        self.max_file_size = max_file_size
        self.retention = retention
        self.compression = compression
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.structured_format = structured_format
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize logger
        self._setup_logger()
        
        # Create contextual logger instance
        self.contextual = ContextualLogger()
    
    def _setup_logger(self):
        """Setup loguru logger with appropriate handlers."""
        # Remove default handler
        logger.remove()
        
        # Console handler
        if self.enable_console:
            if self.structured_format:
                console_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
            else:
                console_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
            
            logger.add(
                sys.stderr,
                format=console_format,
                level=self.log_level,
                colorize=True,
                backtrace=True,
                diagnose=True
            )
        
        # File handlers
        if self.enable_file:
            # Main log file
            logger.add(
                self.log_dir / "comic_translate.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                level=self.log_level,
                rotation=self.max_file_size,
                retention=self.retention,
                compression=self.compression,
                backtrace=True,
                diagnose=True
            )
            
            # Error-only log file
            logger.add(
                self.log_dir / "errors.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                level="ERROR",
                rotation=self.max_file_size,
                retention=self.retention,
                compression=self.compression,
                backtrace=True,
                diagnose=True
            )
            
            # Performance log file
            logger.add(
                self.log_dir / "performance.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
                level="INFO",
                rotation=self.max_file_size,
                retention=self.retention,
                compression=self.compression,
                filter=lambda record: "performance" in record["message"].lower()
            )
    
    @contextmanager
    def component_context(self, component: str, **kwargs):
        """Context manager for component-specific logging."""
        old_context = LogContext(**self.contextual.context.__dict__)
        
        # Update context
        self.contextual.context.component = component
        for key, value in kwargs.items():
            if hasattr(self.contextual.context, key):
                setattr(self.contextual.context, key, value)
            else:
                self.contextual.context.extra[key] = value
        
        try:
            yield self.contextual
        finally:
            # Restore old context
            self.contextual.context = old_context
    
    @contextmanager
    def operation_context(self, operation: str, **kwargs):
        """Context manager for operation-specific logging."""
        old_context = LogContext(**self.contextual.context.__dict__)
        
        # Update context
        self.contextual.context.operation = operation
        for key, value in kwargs.items():
            if hasattr(self.contextual.context, key):
                setattr(self.contextual.context, key, value)
            else:
                self.contextual.context.extra[key] = value
        
        try:
            yield self.contextual
        finally:
            # Restore old context
            self.contextual.context = old_context
    
    @contextmanager
    def performance_tracking(self, operation: str, **kwargs):
        """Context manager for performance tracking."""
        metrics = PerformanceMetrics()
        
        # Get memory usage if psutil is available
        try:
            import psutil
            process = psutil.Process()
            metrics.memory_before = process.memory_info().rss
        except ImportError:
            pass
        
        with self.operation_context(operation, **kwargs) as log:
            try:
                yield metrics
            finally:
                metrics.finish()
                
                # Get final memory usage
                try:
                    import psutil
                    process = psutil.Process()
                    metrics.memory_after = process.memory_info().rss
                except ImportError:
                    pass
                
                log.performance(f"Operation completed: {operation}", metrics)


def performance_monitor(operation_name: Optional[str] = None):
    """Decorator for automatic performance monitoring."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with get_logger().performance_tracking(op_name) as metrics:
                try:
                    result = func(*args, **kwargs)
                    metrics.items_processed = 1
                    return result
                except Exception as e:
                    metrics.errors_count = 1
                    raise
        
        return wrapper
    return decorator


def log_exceptions(logger_instance: Optional[ContextualLogger] = None):
    """Decorator for automatic exception logging."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logger_instance or get_logger().contextual
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log.error(f"Exception in {func.__name__}", exception=e)
                raise
        
        return wrapper
    return decorator


class LogAnalyzer:
    """Utilities for log analysis and monitoring."""
    
    def __init__(self, log_dir: Union[str, Path]):
        self.log_dir = Path(log_dir)
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours."""
        error_log = self.log_dir / "errors.log"
        if not error_log.exists():
            return {"total_errors": 0, "error_types": {}, "components": {}}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        errors = []
        
        try:
            with open(error_log, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # Parse structured log entry
                        if line.strip().startswith('{'):
                            log_entry = json.loads(line.strip())
                            timestamp = datetime.fromisoformat(log_entry.get('timestamp', ''))
                            if timestamp >= cutoff_time:
                                errors.append(log_entry)
                    except (json.JSONDecodeError, ValueError):
                        # Handle non-JSON log entries
                        continue
        except FileNotFoundError:
            pass
        
        # Analyze errors
        error_types = {}
        components = {}
        
        for error in errors:
            # Count by exception type
            if 'exception' in error:
                exc_type = error['exception'].get('type', 'Unknown')
                error_types[exc_type] = error_types.get(exc_type, 0) + 1
            
            # Count by component
            component = error.get('context', {}).get('component', 'Unknown')
            components[component] = components.get(component, 0) + 1
        
        return {
            "total_errors": len(errors),
            "error_types": error_types,
            "components": components,
            "time_range_hours": hours
        }
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        perf_log = self.log_dir / "performance.log"
        if not perf_log.exists():
            return {"operations": {}, "average_duration": 0, "total_operations": 0}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        operations = {}
        
        try:
            with open(perf_log, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # Parse structured log entry
                        if line.strip().startswith('{'):
                            log_entry = json.loads(line.strip())
                            timestamp = datetime.fromisoformat(log_entry.get('timestamp', ''))
                            if timestamp >= cutoff_time:
                                operation = log_entry.get('context', {}).get('operation', 'Unknown')
                                perf_data = log_entry.get('performance', {})
                                
                                if operation not in operations:
                                    operations[operation] = {
                                        "count": 0,
                                        "total_duration": 0,
                                        "total_items": 0,
                                        "total_errors": 0
                                    }
                                
                                operations[operation]["count"] += 1
                                operations[operation]["total_duration"] += perf_data.get('duration_ms', 0)
                                operations[operation]["total_items"] += perf_data.get('items_processed', 0)
                                operations[operation]["total_errors"] += perf_data.get('errors_count', 0)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except FileNotFoundError:
            pass
        
        # Calculate averages
        for op_data in operations.values():
            if op_data["count"] > 0:
                op_data["avg_duration_ms"] = op_data["total_duration"] / op_data["count"]
                op_data["avg_throughput"] = op_data["total_items"] / op_data["count"] if op_data["count"] > 0 else 0
                op_data["error_rate"] = op_data["total_errors"] / op_data["count"] if op_data["count"] > 0 else 0
        
        total_operations = sum(op["count"] for op in operations.values())
        avg_duration = sum(op["total_duration"] for op in operations.values()) / total_operations if total_operations > 0 else 0
        
        return {
            "operations": operations,
            "average_duration_ms": avg_duration,
            "total_operations": total_operations,
            "time_range_hours": hours
        }


# Global logger instance
_global_logger: Optional[LoggingConfig] = None


def setup_logging(log_dir: Optional[Union[str, Path]] = None,
                  log_level: str = "INFO",
                  **kwargs) -> LoggingConfig:
    """
    Setup global logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Minimum log level
        **kwargs: Additional configuration options
    
    Returns:
        LoggingConfig instance
    """
    global _global_logger
    _global_logger = LoggingConfig(log_dir=log_dir, log_level=log_level, **kwargs)
    return _global_logger


def get_logger() -> LoggingConfig:
    """
    Get the global logger instance.
    
    Returns:
        LoggingConfig instance
    
    Raises:
        RuntimeError: If logging hasn't been setup
    """
    global _global_logger
    if _global_logger is None:
        # Auto-setup with defaults
        _global_logger = LoggingConfig()
    return _global_logger


# Component-specific logger factories
def get_detection_logger():
    """Get logger configured for detection operations."""
    return get_logger().contextual


def get_ocr_logger():
    """Get logger configured for OCR operations."""
    return get_logger().contextual


def get_translation_logger():
    """Get logger configured for translation operations."""
    return get_logger().contextual


def get_ui_logger():
    """Get logger configured for UI operations."""
    return get_logger().contextual


def get_pipeline_logger():
    """Get logger configured for pipeline operations."""
    return get_logger().contextual


def get_cache_logger():
    """Get logger configured for cache operations."""
    return get_logger().contextual


# Convenience functions for quick logging
def log_info(message: str, component: str = "", **kwargs):
    """Quick info logging with optional component context."""
    with get_logger().component_context(component) as log:
        log.info(message, **kwargs)


def log_error(message: str, exception: Optional[Exception] = None, component: str = "", **kwargs):
    """Quick error logging with optional component context."""
    with get_logger().component_context(component) as log:
        log.error(message, exception=exception, **kwargs)


def log_warning(message: str, component: str = "", **kwargs):
    """Quick warning logging with optional component context."""
    with get_logger().component_context(component) as log:
        log.warning(message, **kwargs)


def log_debug(message: str, component: str = "", **kwargs):
    """Quick debug logging with optional component context."""
    with get_logger().component_context(component) as log:
        log.debug(message, **kwargs)


# Export main classes and functions
__all__ = [
    'LoggingConfig',
    'ContextualLogger', 
    'LogContext',
    'PerformanceMetrics',
    'LogAnalyzer',
    'setup_logging',
    'get_logger',
    'get_detection_logger',
    'get_ocr_logger', 
    'get_translation_logger',
    'get_ui_logger',
    'get_pipeline_logger',
    'get_cache_logger',
    'performance_monitor',
    'log_exceptions',
    'log_info',
    'log_error',
    'log_warning',
    'log_debug',
]