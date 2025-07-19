"""
Pytest configuration and shared fixtures for Comic Translate MT testing infrastructure.

This module provides comprehensive test fixtures including mock engines, test data,
temporary directories, Qt application testing, performance monitoring, and utilities
for test data generation and validation.
"""

import os
import sys
import tempfile
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator, Callable
from unittest.mock import Mock, MagicMock, patch
import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from modules.utils.cache import ThreadSafeLRUCache, PersistentCache, CacheManager
from modules.utils.exceptions import (
    ComicTranslateException, DetectionError, OCRError, TranslationError,
    InpaintingError, CacheError, ConfigurationError, ErrorCode, ErrorSeverity
)

# Qt imports for GUI testing
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    from PySide6.QtTest import QTest
    import pytest_qt
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ============================================================================
# Test Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that require API access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Mark slow tests
        if "slow" in item.name or "performance" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


# ============================================================================
# Logging Configuration
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def configure_test_logging():
    """Configure logging for tests."""
    import logging
    
    # Create test-specific logger
    logger = logging.getLogger("comic_translate_test")
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler for test output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# Temporary Directory and File Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_cache_dir(temp_dir):
    """Create a temporary directory specifically for cache files."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


@pytest.fixture
def temp_config_dir(temp_dir):
    """Create a temporary directory for configuration files."""
    config_dir = temp_dir / "config"
    config_dir.mkdir(exist_ok=True)
    return config_dir


@pytest.fixture
def temp_image_dir(temp_dir):
    """Create a temporary directory for test images."""
    image_dir = temp_dir / "images"
    image_dir.mkdir(exist_ok=True)
    return image_dir


# ============================================================================
# Test Image Generation
# ============================================================================

class TestImageGenerator:
    """Utility class for generating test images with various characteristics."""
    
    @staticmethod
    def create_simple_image(width: int = 800, height: int = 600, 
                          color: tuple = (255, 255, 255)) -> Image.Image:
        """Create a simple solid color image."""
        return Image.new('RGB', (width, height), color)
    
    @staticmethod
    def create_text_image(text: str = "Sample Text", width: int = 800, 
                         height: int = 600, font_size: int = 48) -> Image.Image:
        """Create an image with text for OCR testing."""
        img = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to use a system font
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), text, fill=(0, 0, 0), font=font)
        return img
    
    @staticmethod
    def create_comic_panel_image(width: int = 800, height: int = 600) -> Image.Image:
        """Create a mock comic panel with speech bubbles."""
        img = Image.new('RGB', (width, height), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Draw speech bubbles
        bubbles = [
            ((50, 50, 300, 150), "Hello World!"),
            ((450, 200, 750, 300), "こんにちは"),
            ((100, 400, 400, 500), "Test Text")
        ]
        
        for (x1, y1, x2, y2), text in bubbles:
            # Draw bubble background
            draw.ellipse([x1, y1, x2, y2], fill=(255, 255, 255), outline=(0, 0, 0))
            
            # Add text
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except (OSError, IOError):
                font = ImageFont.load_default()
            
            # Center text in bubble
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = x1 + (x2 - x1 - text_width) // 2
            text_y = y1 + (y2 - y1 - text_height) // 2
            
            draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
        
        return img
    
    @staticmethod
    def create_noisy_image(width: int = 800, height: int = 600, 
                          noise_level: float = 0.1) -> Image.Image:
        """Create an image with noise for robustness testing."""
        img = TestImageGenerator.create_text_image("Noisy Text", width, height)
        img_array = np.array(img)
        
        # Add random noise
        noise = np.random.randint(0, int(255 * noise_level), img_array.shape, dtype=np.uint8)
        noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_array)


@pytest.fixture
def image_generator():
    """Provide test image generator."""
    return TestImageGenerator()


@pytest.fixture
def sample_images(temp_image_dir, image_generator):
    """Create a set of sample test images."""
    images = {}
    
    # Simple test image
    simple_img = image_generator.create_simple_image()
    simple_path = temp_image_dir / "simple.png"
    simple_img.save(simple_path)
    images['simple'] = simple_path
    
    # Text image for OCR
    text_img = image_generator.create_text_image("Hello World")
    text_path = temp_image_dir / "text.png"
    text_img.save(text_path)
    images['text'] = text_path
    
    # Comic panel
    comic_img = image_generator.create_comic_panel_image()
    comic_path = temp_image_dir / "comic.png"
    comic_img.save(comic_path)
    images['comic'] = comic_path
    
    # Noisy image
    noisy_img = image_generator.create_noisy_image()
    noisy_path = temp_image_dir / "noisy.png"
    noisy_img.save(noisy_path)
    images['noisy'] = noisy_path
    
    return images


# ============================================================================
# Mock Engine Fixtures
# ============================================================================

class MockDetectionEngine:
    """Mock detection engine for testing."""
    
    def __init__(self, name: str = "mock_detector"):
        self.name = name
        self.is_loaded = True
        self.call_count = 0
    
    def detect_text_blocks(self, image_path: str) -> List[Dict[str, Any]]:
        """Mock text detection that returns predefined blocks."""
        self.call_count += 1
        
        # Return mock text blocks
        return [
            {
                'bbox': [50, 50, 300, 150],
                'confidence': 0.95,
                'text_region': True
            },
            {
                'bbox': [450, 200, 750, 300],
                'confidence': 0.88,
                'text_region': True
            }
        ]
    
    def cleanup(self):
        """Mock cleanup method."""
        self.is_loaded = False


class MockOCREngine:
    """Mock OCR engine for testing."""
    
    def __init__(self, name: str = "mock_ocr", language: str = "en"):
        self.name = name
        self.language = language
        self.call_count = 0
        self.mock_results = {
            "Hello World": "Hello World",
            "こんにちは": "Hello",
            "Test Text": "Test Text"
        }
    
    def extract_text(self, image_path: str, text_blocks: List[Dict]) -> List[str]:
        """Mock OCR that returns predefined text."""
        self.call_count += 1
        
        # Return mock text based on number of blocks
        results = []
        mock_texts = list(self.mock_results.keys())
        
        for i, block in enumerate(text_blocks):
            if i < len(mock_texts):
                results.append(mock_texts[i])
            else:
                results.append(f"Mock text {i}")
        
        return results
    
    def set_language(self, language: str):
        """Mock language setting."""
        self.language = language


class MockTranslationEngine:
    """Mock translation engine for testing."""
    
    def __init__(self, name: str = "mock_translator"):
        self.name = name
        self.call_count = 0
        self.mock_translations = {
            "Hello World": "こんにちは世界",
            "Test Text": "テストテキスト",
            "Sample": "サンプル"
        }
    
    def translate_texts(self, texts: List[str], source_lang: str, 
                       target_lang: str) -> List[str]:
        """Mock translation that returns predefined translations."""
        self.call_count += 1
        
        results = []
        for text in texts:
            translated = self.mock_translations.get(text, f"Translated: {text}")
            results.append(translated)
        
        return results


class MockInpaintingEngine:
    """Mock inpainting engine for testing."""
    
    def __init__(self, name: str = "mock_inpainter"):
        self.name = name
        self.call_count = 0
    
    def inpaint_image(self, image_path: str, mask_path: str, 
                     output_path: str) -> str:
        """Mock inpainting that copies the original image."""
        self.call_count += 1
        
        # Simply copy the original image for testing
        import shutil
        shutil.copy2(image_path, output_path)
        return output_path


@pytest.fixture
def mock_detection_engine():
    """Provide mock detection engine."""
    return MockDetectionEngine()


@pytest.fixture
def mock_ocr_engine():
    """Provide mock OCR engine."""
    return MockOCREngine()


@pytest.fixture
def mock_translation_engine():
    """Provide mock translation engine."""
    return MockTranslationEngine()


@pytest.fixture
def mock_inpainting_engine():
    """Provide mock inpainting engine."""
    return MockInpaintingEngine()


@pytest.fixture
def mock_engines(mock_detection_engine, mock_ocr_engine, 
                mock_translation_engine, mock_inpainting_engine):
    """Provide all mock engines together."""
    return {
        'detection': mock_detection_engine,
        'ocr': mock_ocr_engine,
        'translation': mock_translation_engine,
        'inpainting': mock_inpainting_engine
    }


# ============================================================================
# Cache and Database Fixtures
# ============================================================================

@pytest.fixture
def mock_cache():
    """Provide a mock cache for testing."""
    cache = ThreadSafeLRUCache(max_size=100)
    yield cache
    cache.clear()


@pytest.fixture
def mock_persistent_cache(temp_cache_dir):
    """Provide a mock persistent cache for testing."""
    cache_file = temp_cache_dir / "test_cache.msgpack"
    cache = PersistentCache(cache_file, max_size=100, auto_save=False)
    yield cache
    cache.clear()


@pytest.fixture
def mock_cache_manager(temp_cache_dir):
    """Provide a mock cache manager for testing."""
    manager = CacheManager()
    
    # Create test caches
    lru_cache = manager.create_lru_cache("test_lru", max_size=50)
    persistent_cache = manager.create_persistent_cache(
        "test_persistent", 
        temp_cache_dir / "manager_cache.msgpack",
        max_size=50,
        auto_save=False
    )
    
    yield manager
    
    # Cleanup
    manager.clear_all()


# ============================================================================
# Settings and Configuration Fixtures
# ============================================================================

@pytest.fixture
def mock_settings():
    """Provide mock application settings."""
    return {
        'detection': {
            'model': 'mock_detector',
            'confidence_threshold': 0.5,
            'gpu_enabled': False
        },
        'ocr': {
            'engine': 'mock_ocr',
            'language': 'en',
            'api_key': 'test_key'
        },
        'translation': {
            'engine': 'mock_translator',
            'source_language': 'en',
            'target_language': 'ja',
            'api_key': 'test_key'
        },
        'inpainting': {
            'model': 'mock_inpainter',
            'gpu_enabled': False
        },
        'cache': {
            'enabled': True,
            'max_size': 1000,
            'ttl': 3600
        },
        'ui': {
            'theme': 'light',
            'language': 'en',
            'max_images_in_memory': 10
        }
    }


@pytest.fixture
def mock_config_file(temp_config_dir, mock_settings):
    """Create a mock configuration file."""
    import json
    
    config_file = temp_config_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(mock_settings, f, indent=2)
    
    return config_file


# ============================================================================
# Qt Application Fixtures
# ============================================================================

if QT_AVAILABLE:
    @pytest.fixture(scope="session")
    def qapp():
        """Create QApplication for Qt tests."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app
        # Don't quit the app as it might be used by other tests
    
    @pytest.fixture
    def qtbot(qapp, qtbot):
        """Provide qtbot with application context."""
        return qtbot
    
    @pytest.fixture
    def mock_main_window(qapp):
        """Create a mock main window for testing."""
        from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel
        
        window = QMainWindow()
        central_widget = QWidget()
        layout = QVBoxLayout()
        label = QLabel("Test Window")
        layout.addWidget(label)
        central_widget.setLayout(layout)
        window.setCentralWidget(central_widget)
        
        yield window
        window.close()

else:
    @pytest.fixture
    def qapp():
        """Dummy fixture when Qt is not available."""
        pytest.skip("Qt not available")
    
    @pytest.fixture
    def qtbot():
        """Dummy fixture when Qt is not available."""
        pytest.skip("Qt not available")
    
    @pytest.fixture
    def mock_main_window():
        """Dummy fixture when Qt is not available."""
        pytest.skip("Qt not available")


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

class PerformanceMonitor:
    """Monitor performance metrics during tests."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None
        
    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory
    
    def update_peak_memory(self):
        """Update peak memory usage."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            if current_memory > self.peak_memory:
                self.peak_memory = current_memory
    
    def stop(self):
        """Stop performance monitoring."""
        self.end_time = time.time()
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            self.end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        metrics = {}
        
        if self.start_time and self.end_time:
            metrics['execution_time'] = self.end_time - self.start_time
        
        if PSUTIL_AVAILABLE and self.start_memory and self.end_memory:
            metrics['memory_start_mb'] = self.start_memory
            metrics['memory_end_mb'] = self.end_memory
            metrics['memory_peak_mb'] = self.peak_memory
            metrics['memory_delta_mb'] = self.end_memory - self.start_memory
        
        return metrics


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring for tests."""
    monitor = PerformanceMonitor()
    yield monitor


@pytest.fixture
def benchmark_timer():
    """Simple timer for benchmarking test operations."""
    times = {}
    
    def timer(name: str):
        """Context manager for timing operations."""
        class Timer:
            def __enter__(self):
                self.start = time.time()
                return self
            
            def __exit__(self, *args):
                self.end = time.time()
                times[name] = self.end - self.start
        
        return Timer()
    
    timer.times = times
    return timer


# ============================================================================
# Exception Testing Fixtures
# ============================================================================

@pytest.fixture
def sample_exceptions():
    """Provide sample exceptions for testing."""
    return {
        'base': ComicTranslateException("Base error"),
        'detection': DetectionError(
            "Detection failed",
            model_name="test_model",
            image_path="/test/image.jpg"
        ),
        'ocr': OCRError(
            "OCR failed",
            engine_name="test_ocr",
            language="en"
        ),
        'translation': TranslationError(
            "Translation failed",
            engine_name="test_translator",
            source_language="en",
            target_language="ja"
        ),
        'inpainting': InpaintingError(
            "Inpainting failed",
            model_name="test_inpainter"
        ),
        'cache': CacheError(
            "Cache error",
            cache_type="LRU"
        ),
        'config': ConfigurationError(
            "Config error",
            config_key="test_key"
        )
    }


# ============================================================================
# Test Data Validation Utilities
# ============================================================================

class TestDataValidator:
    """Utility class for validating test data and results."""
    
    @staticmethod
    def validate_image_file(image_path: Path) -> bool:
        """Validate that a file is a valid image."""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_text_blocks(text_blocks: List[Dict]) -> bool:
        """Validate text block structure."""
        required_keys = ['bbox', 'confidence']
        
        for block in text_blocks:
            if not isinstance(block, dict):
                return False
            
            for key in required_keys:
                if key not in block:
                    return False
            
            # Validate bbox format
            bbox = block['bbox']
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return False
            
            # Validate confidence
            confidence = block['confidence']
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                return False
        
        return True
    
    @staticmethod
    def validate_cache_stats(stats: Dict[str, Any]) -> bool:
        """Validate cache statistics structure."""
        required_keys = ['size', 'max_size', 'hits', 'misses', 'hit_rate']
        
        for key in required_keys:
            if key not in stats:
                return False
        
        # Validate numeric values
        for key in ['size', 'max_size', 'hits', 'misses']:
            if not isinstance(stats[key], int) or stats[key] < 0:
                return False
        
        # Validate hit rate
        hit_rate = stats['hit_rate']
        if not isinstance(hit_rate, (int, float)) or not 0 <= hit_rate <= 1:
            return False
        
        return True


@pytest.fixture
def test_validator():
    """Provide test data validator."""
    return TestDataValidator()


# ============================================================================
# Cleanup and Teardown
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test."""
    yield
    
    # Clear any global state
    import gc
    gc.collect()
    
    # Reset mock call counts if they exist
    for obj_name in dir():
        obj = locals().get(obj_name)
        if hasattr(obj, 'call_count'):
            obj.call_count = 0


@pytest.fixture(scope="session", autouse=True)
def session_cleanup():
    """Cleanup at the end of test session."""
    yield
    
    # Final cleanup
    import gc
    gc.collect()


# ============================================================================
# Utility Functions
# ============================================================================

def create_test_project_structure(base_dir: Path) -> Dict[str, Path]:
    """Create a test project directory structure."""
    structure = {
        'root': base_dir,
        'images': base_dir / 'images',
        'cache': base_dir / 'cache',
        'config': base_dir / 'config',
        'output': base_dir / 'output',
        'temp': base_dir / 'temp'
    }
    
    for path in structure.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return structure


def wait_for_condition(condition: Callable[[], bool], timeout: float = 5.0, 
                      interval: float = 0.1) -> bool:
    """Wait for a condition to become true with timeout."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if condition():
            return True
        time.sleep(interval)
    
    return False


def assert_performance_within_limits(metrics: Dict[str, float], 
                                    max_time: float = None,
                                    max_memory_mb: float = None):
    """Assert that performance metrics are within acceptable limits."""
    if max_time and 'execution_time' in metrics:
        assert metrics['execution_time'] <= max_time, \
            f"Execution time {metrics['execution_time']:.3f}s exceeds limit {max_time}s"
    
    if max_memory_mb and 'memory_peak_mb' in metrics:
        assert metrics['memory_peak_mb'] <= max_memory_mb, \
            f"Peak memory {metrics['memory_peak_mb']:.1f}MB exceeds limit {max_memory_mb}MB"


# Export commonly used fixtures and utilities
__all__ = [
    'temp_dir', 'temp_cache_dir', 'temp_config_dir', 'temp_image_dir',
    'image_generator', 'sample_images',
    'mock_detection_engine', 'mock_ocr_engine', 'mock_translation_engine', 
    'mock_inpainting_engine', 'mock_engines',
    'mock_cache', 'mock_persistent_cache', 'mock_cache_manager',
    'mock_settings', 'mock_config_file',
    'qapp', 'qtbot', 'mock_main_window',
    'performance_monitor', 'benchmark_timer',
    'sample_exceptions', 'test_validator',
    'create_test_project_structure', 'wait_for_condition', 
    'assert_performance_within_limits'
]