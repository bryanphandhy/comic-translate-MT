# Contributing to Comic Translate MT

Thank you for your interest in contributing to Comic Translate MT! This document provides comprehensive guidelines for setting up your development environment, understanding the project architecture, and contributing effectively to the project.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Architecture](#project-architecture)
- [Coding Standards](#coding-standards)
- [Adding New Engines](#adding-new-engines)
- [Testing](#testing)
- [Code Review Process](#code-review-process)
- [Documentation](#documentation)
- [Issue Templates](#issue-templates)
- [Release Process](#release-process)

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) CUDA-compatible GPU for deep learning models

### Basic Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/comic-translate-mt/comic-translate-mt.git
   cd comic-translate-mt
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install core dependencies:**
   ```bash
   pip install -e .
   ```

4. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

### Optional Dependencies

The project uses optional dependency groups for different features:

#### GPU Support
For deep learning models with CUDA acceleration:
```bash
pip install -e ".[gpu]"
```

#### Chinese OCR Support
For PaddleOCR and Chinese text recognition:
```bash
pip install -e ".[ocr-chinese]"
```

#### LLM Translation Support
For transformer-based translation models:
```bash
pip install -e ".[llm-translation]"
```

#### Cloud Services
For cloud-based OCR and translation APIs:
```bash
pip install -e ".[cloud-services]"
```

#### All Features
To install all optional dependencies:
```bash
pip install -e ".[all]"
```

### Development Tools Setup

1. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

2. **Configure your IDE:**
   - Set up Python path to use the virtual environment
   - Configure linting with ruff
   - Set up type checking with mypy
   - Configure formatting with black

### Environment Variables

Create a `.env` file in the project root for API keys and configuration:
```bash
# Translation APIs
DEEPL_API_KEY=your_deepl_key
GOOGLE_TRANSLATE_API_KEY=your_google_key
AZURE_TRANSLATOR_KEY=your_azure_key

# OCR APIs
AZURE_VISION_KEY=your_azure_vision_key
AZURE_VISION_ENDPOINT=your_azure_endpoint

# Development settings
DEBUG=true
LOG_LEVEL=DEBUG
```

## Project Architecture

### Overview

Comic Translate MT follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   UI Layer      │    │  Controller     │    │   Pipeline      │
│   (app/ui/)     │◄──►│  (controller.py)│◄──►│  (pipeline.py)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │    Modules      │
                                               │  (modules/)     │
                                               └─────────────────┘
```

### Key Design Patterns

#### Factory Pattern
All processing engines (detection, OCR, translation, inpainting) use the factory pattern:

```python
# Example: OCR Factory
class OCRFactory:
    @staticmethod
    def create_engine(engine_type: str, **kwargs) -> OCREngine:
        if engine_type == "manga_ocr":
            return MangaOCREngine(**kwargs)
        elif engine_type == "paddle_ocr":
            return PaddleOCREngine(**kwargs)
        # ... other engines
```

#### Abstract Base Classes
Each engine type has an abstract base class defining the interface:

```python
class OCREngine(ABC):
    @abstractmethod
    def encode_image(self, image: np.ndarray, text_blocks: List[TextBlock]) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
```

#### Pipeline Orchestration
The main pipeline coordinates all processing steps:

1. **Detection**: Find text regions in images
2. **OCR**: Extract text from detected regions
3. **Translation**: Translate extracted text
4. **Inpainting**: Remove original text from images
5. **Rendering**: Render translated text back onto images

#### Thread Safety
- All caches use `ThreadSafeLRUCache` with proper locking
- Factory instances are thread-safe with concurrent access protection
- Pipeline operations support cancellation and cleanup

### Module Structure

#### `modules/detection/`
- Text and speech bubble detection using deep learning models
- Supports RT-DETR-V2 and other detection architectures
- Extensible through the factory pattern

#### `modules/ocr/`
- Multiple OCR backends: Manga OCR, PaddleOCR, DocTR, GPT-4V
- Language-specific optimizations
- Confidence scoring and result validation

#### `modules/translation/`
- Traditional APIs: Google, DeepL, Microsoft, Yandex
- LLM-based translation with context awareness
- Batch processing and rate limiting

#### `modules/inpainting/`
- Text removal using LaMa, AOT, MI-GAN models
- GPU acceleration support
- Quality assessment and fallback mechanisms

#### `modules/utils/`
- Shared utilities: caching, logging, exceptions
- File handling and archive support
- Configuration management

## Coding Standards

### Code Style

We use the following tools for code quality:

- **Formatter**: Black with 88-character line length
- **Linter**: Ruff for fast Python linting
- **Type Checker**: MyPy for static type analysis
- **Import Sorting**: isort with black-compatible settings

### Code Quality Rules

1. **Type Hints**: All public functions must have type hints
   ```python
   def process_image(image: np.ndarray, config: Dict[str, Any]) -> ProcessResult:
       pass
   ```

2. **Error Handling**: Use the custom exception hierarchy
   ```python
   from modules.utils.exceptions import OCRError
   
   try:
       result = ocr_engine.process(image)
   except Exception as e:
       raise OCRError(f"OCR processing failed: {e}", context={"image_shape": image.shape})
   ```

3. **Logging**: Use structured logging with context
   ```python
   from modules.utils.logging_config import get_logger
   
   logger = get_logger(__name__)
   logger.info("Processing image", extra={"image_path": path, "model": model_name})
   ```

4. **Documentation**: All public APIs must have docstrings
   ```python
   def detect_text_blocks(self, image: np.ndarray) -> List[TextBlock]:
       """Detect text blocks in the given image.
       
       Args:
           image: Input image as numpy array in BGR format
           
       Returns:
           List of detected text blocks with bounding boxes
           
       Raises:
           DetectionError: If detection fails or image is invalid
       """
   ```

### Performance Guidelines

1. **Memory Management**: Use bounded caches and proper cleanup
2. **GPU Resources**: Always clean up GPU memory after use
3. **Threading**: Use thread-safe collections and proper locking
4. **Caching**: Cache expensive operations with appropriate invalidation

## Adding New Engines

### Detection Engines

1. **Create the engine class:**
   ```python
   # modules/detection/my_detector.py
   from .base import DetectionEngine
   
   class MyDetectionEngine(DetectionEngine):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           # Initialize your model
           
       def detect(self, image: np.ndarray) -> List[Dict]:
           # Implement detection logic
           pass
           
       def is_available(self) -> bool:
           # Check if dependencies are available
           return True
   ```

2. **Register in the factory:**
   ```python
   # modules/detection/factory.py
   def create_detection_engine(engine_type: str, **kwargs) -> DetectionEngine:
       if engine_type == "my_detector":
           from .my_detector import MyDetectionEngine
           return MyDetectionEngine(**kwargs)
   ```

3. **Add configuration schema:**
   ```python
   # Add to detection configuration
   MY_DETECTOR_CONFIG = {
       "confidence_threshold": 0.5,
       "model_path": "path/to/model",
       # ... other config options
   }
   ```

### OCR Engines

1. **Implement the base class:**
   ```python
   # modules/ocr/my_ocr.py
   from .base import OCREngine
   
   class MyOCREngine(OCREngine):
       def encode_image(self, image: np.ndarray, text_blocks: List[TextBlock]) -> str:
           # Implement OCR logic
           pass
           
       def is_available(self) -> bool:
           # Check dependencies
           return True
   ```

2. **Add language support:**
   ```python
   SUPPORTED_LANGUAGES = ["en", "ja", "ko", "zh"]
   
   def supports_language(self, language: str) -> bool:
       return language in SUPPORTED_LANGUAGES
   ```

### Translation Engines

1. **Implement base classes:**
   ```python
   # modules/translation/my_translator.py
   from .base import TranslationEngine
   
   class MyTranslationEngine(TranslationEngine):
       def translate(self, text: str, source_lang: str, target_lang: str) -> str:
           # Implement translation logic
           pass
           
       def is_available(self) -> bool:
           return True
   ```

2. **Add batch processing support:**
   ```python
   def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
       # Implement efficient batch translation
       pass
   ```

### Plugin Registration

For external plugins, create an entry point in `setup.py`:

```python
entry_points={
    "comic_translate.detection": [
        "my_detector = my_plugin.detection:MyDetectionEngine",
    ],
    "comic_translate.ocr": [
        "my_ocr = my_plugin.ocr:MyOCREngine",
    ],
    "comic_translate.translation": [
        "my_translator = my_plugin.translation:MyTranslationEngine",
    ],
}
```

## Testing

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests for individual components
│   ├── test_cache.py
│   ├── test_exceptions.py
│   ├── test_factories.py
│   └── ...
├── integration/             # Integration tests
│   ├── test_pipeline.py
│   ├── test_ui_integration.py
│   └── ...
└── performance/             # Performance and stress tests
    ├── test_memory_usage.py
    └── test_throughput.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=modules --cov=app --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest -m "not slow"        # Skip slow tests

# Run tests in parallel
pytest -n auto
```

### Writing Tests

1. **Use fixtures for common setup:**
   ```python
   def test_ocr_engine(mock_ocr_engine, sample_image):
       result = mock_ocr_engine.encode_image(sample_image, [])
       assert result is not None
   ```

2. **Mock external dependencies:**
   ```python
   @patch('modules.ocr.paddle_ocr.PaddleOCR')
   def test_paddle_ocr_initialization(mock_paddle):
       engine = PaddleOCREngine()
       assert engine.is_available()
   ```

3. **Test error conditions:**
   ```python
   def test_ocr_invalid_image():
       engine = OCREngine()
       with pytest.raises(OCRError):
           engine.encode_image(None, [])
   ```

### Test Requirements

- **Unit tests**: All public methods must have unit tests
- **Integration tests**: Critical workflows must have integration tests
- **Performance tests**: Memory usage and throughput tests for core operations
- **UI tests**: Use pytest-qt for GUI component testing
- **Coverage**: Maintain >90% code coverage for core modules

## Code Review Process

### Pull Request Guidelines

1. **Branch naming:**
   - `feature/description` for new features
   - `bugfix/description` for bug fixes
   - `refactor/description` for refactoring
   - `docs/description` for documentation

2. **Commit messages:**
   ```
   type(scope): brief description
   
   Longer description if needed
   
   Fixes #123
   ```

3. **PR description template:**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests added/updated
   - [ ] Manual testing completed
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No new warnings introduced
   ```

### Review Criteria

1. **Code Quality:**
   - Follows coding standards
   - Proper error handling
   - Adequate test coverage
   - Clear documentation

2. **Architecture:**
   - Follows established patterns
   - Maintains modularity
   - Thread-safe implementation
   - Proper resource management

3. **Performance:**
   - No memory leaks
   - Efficient algorithms
   - Appropriate caching
   - GPU resource cleanup

### Review Process

1. **Automated checks:** All CI checks must pass
2. **Peer review:** At least one maintainer approval required
3. **Testing:** Manual testing for UI changes
4. **Documentation:** Updates to relevant documentation

## Documentation

### Types of Documentation

1. **Code Documentation:**
   - Docstrings for all public APIs
   - Inline comments for complex logic
   - Type hints for all functions

2. **User Documentation:**
   - README.md for basic usage
   - docs/ directory for detailed guides
   - Multi-language support

3. **Developer Documentation:**
   - Architecture overview
   - API reference
   - Contributing guidelines

### Documentation Standards

1. **Docstring format:**
   ```python
   def function_name(param1: Type1, param2: Type2) -> ReturnType:
       """Brief description.
       
       Longer description if needed.
       
       Args:
           param1: Description of param1
           param2: Description of param2
           
       Returns:
           Description of return value
           
       Raises:
           ExceptionType: Description of when this exception is raised
           
       Example:
           >>> result = function_name("value1", "value2")
           >>> print(result)
           expected_output
       """
   ```

2. **Markdown standards:**
   - Use clear headings and structure
   - Include code examples
   - Add screenshots for UI features
   - Keep language simple and clear

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[dev]"

# Build documentation
cd docs
make html

# Serve documentation locally
python -m http.server 8000 -d _build/html
```

## Issue Templates

### Bug Report Template

```markdown
**Bug Description**
A clear and concise description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
A clear description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g. Windows 10, Ubuntu 20.04]
 - Python Version: [e.g. 3.9.7]
 - Comic Translate Version: [e.g. 1.2.3]
 - GPU: [e.g. NVIDIA RTX 3080, None]

**Additional Context**
Add any other context about the problem here.

**Log Output**
```
Paste relevant log output here
```

### Feature Request Template

```markdown
**Feature Description**
A clear and concise description of the feature you'd like to see.

**Problem Statement**
Describe the problem this feature would solve.

**Proposed Solution**
Describe the solution you'd like to see implemented.

**Alternatives Considered**
Describe any alternative solutions you've considered.

**Additional Context**
Add any other context, mockups, or examples about the feature request.

**Implementation Notes**
If you have ideas about how this could be implemented, please share them.
```

### Engine Addition Template

```markdown
**Engine Type**
- [ ] Detection Engine
- [ ] OCR Engine
- [ ] Translation Engine
- [ ] Inpainting Engine

**Engine Name**
Name of the engine/model/service

**Description**
Brief description of the engine and its capabilities.

**Languages Supported**
List of supported languages (if applicable).

**Dependencies**
List of required dependencies and their versions.

**License**
License of the engine/model (if known).

**Performance Characteristics**
- Speed: [Fast/Medium/Slow]
- Accuracy: [High/Medium/Low]
- Memory Usage: [High/Medium/Low]
- GPU Required: [Yes/No]

**Implementation Plan**
High-level plan for implementing this engine.

**References**
Links to documentation, papers, or repositories.
```

## Release Process

### Versioning Strategy

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Types

1. **Alpha releases**: `1.2.3a1` - Early development versions
2. **Beta releases**: `1.2.3b1` - Feature-complete, testing phase
3. **Release candidates**: `1.2.3rc1` - Final testing before release
4. **Stable releases**: `1.2.3` - Production-ready versions

### Release Workflow

1. **Preparation:**
   ```bash
   # Update version in VERSION file
   echo "1.2.3" > VERSION
   
   # Update CHANGELOG.md
   # Add release notes and migration guide if needed
   ```

2. **Testing:**
   ```bash
   # Run full test suite
   pytest tests/
   
   # Run performance tests
   pytest tests/performance/
   
   # Manual testing on different platforms
   ```

3. **Documentation:**
   ```bash
   # Update documentation
   # Build and verify documentation
   cd docs && make html
   ```

4. **Release:**
   ```bash
   # Create release branch
   git checkout -b release/1.2.3
   
   # Commit version changes
   git commit -am "Release 1.2.3"
   
   # Create tag
   git tag -a v1.2.3 -m "Release 1.2.3"
   
   # Push to repository
   git push origin release/1.2.3
   git push origin v1.2.3
   ```

5. **Distribution:**
   - GitHub Actions automatically builds and publishes releases
   - PyPI package is automatically updated
   - Documentation is deployed to GitHub Pages

### Changelog Maintenance

Keep `CHANGELOG.md` updated with:

```markdown
## [1.2.3] - 2024-01-15

### Added
- New OCR engine support for language X
- Batch processing improvements

### Changed
- Updated UI layout for better usability
- Improved error messages

### Fixed
- Fixed memory leak in image processing
- Resolved crash when loading large images

### Deprecated
- Old API methods (will be removed in 2.0.0)

### Security
- Updated dependencies to fix security vulnerabilities
```

### Migration Guides

For breaking changes, provide migration guides:

```markdown
## Migration Guide: v1.x to v2.0

### Breaking Changes

1. **Configuration Format Changed**
   ```python
   # Old format
   config = {"ocr_engine": "manga_ocr"}
   
   # New format
   config = {"engines": {"ocr": {"type": "manga_ocr"}}}
   ```

2. **API Method Renamed**
   ```python
   # Old API
   result = pipeline.process_image(image)
   
   # New API
   result = pipeline.process_single_image(image)
   ```
```

---

## Getting Help

- **Documentation**: Check the [docs/](docs/) directory
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Discord**: Join our community Discord server (link in README)

## Code of Conduct

Please note that this project is released with a [Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

---

Thank you for contributing to Comic Translate MT! Your contributions help make comic translation accessible to everyone.