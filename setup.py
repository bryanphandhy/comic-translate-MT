#!/usr/bin/env python3
"""
Setup script for Comic Translate MT - A comprehensive comic translation application.
"""

from setuptools import setup, find_packages
import os
import sys

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "A comprehensive comic translation application with OCR, translation, and inpainting capabilities."

# Read version from a version file or set default
def get_version():
    version_path = os.path.join(os.path.dirname(__file__), 'VERSION')
    if os.path.exists(version_path):
        with open(version_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return "1.0.0"

# Core dependencies that are always required
CORE_REQUIREMENTS = [
    "PySide6>=6.8.0,<7.0.0",
    "opencv-python>=4.8.0,<5.0.0",
    "requests>=2.31.0,<3.0.0",
    "msgpack>=1.1.0,<2.0.0",
    "pdfplumber>=0.11.5,<1.0.0",
    "py7zr>=0.20.8,<1.0.0",
    "rarfile>=4.1,<5.0.0",
    "img2pdf>=0.5.1,<1.0.0",
    "wget>=3.2,<4.0.0",
    "loguru>=0.7.2,<1.0.0",
    "pyperclip>=1.9.0,<2.0.0",
    "dayu-path>=0.5.2,<1.0.0",
    "largestinteriorrectangle>=0.2.0,<1.0.0",
    "setuptools>=65.0.0",
]

# Optional dependencies organized by feature
EXTRAS_REQUIRE = {
    # GPU support for deep learning models
    "gpu": [
        "torch>=2.6.0,<3.0.0",
        "torchvision>=0.21.0,<1.0.0",
        "python-doctr[torch]>=0.11.0,<1.0.0",
    ],
    
    # Chinese OCR support
    "ocr-chinese": [
        "paddleocr>=2.8.1,<3.0.0",
        "paddlepaddle>=2.6.1,<3.0.0",
    ],
    
    # LLM-based translation support
    "llm-translation": [
        "transformers>=4.49.0,<5.0.0",
        "stanza>=1.7.0,<2.0.0",
        "jaconv>=0.3.4,<1.0.0",
        "fugashi>=1.4.0,<2.0.0",
        "unidic-lite>=1.0.8,<2.0.0",
    ],
    
    # Cloud services for OCR and translation
    "cloud-services": [
        "azure-ai-vision-imageanalysis>=1.0.0b1,<2.0.0",
        "deepl>=1.16.1,<2.0.0",
        "deep-translator>=1.11.4,<2.0.0",
    ],
    
    # Development and testing tools
    "dev": [
        "pytest>=7.0.0,<8.0.0",
        "pytest-qt>=4.0.0,<5.0.0",
        "pytest-cov>=4.0.0,<5.0.0",
        "pytest-mock>=3.10.0,<4.0.0",
        "pytest-asyncio>=0.21.0,<1.0.0",
        "ruff>=0.1.0,<1.0.0",
        "mypy>=1.5.0,<2.0.0",
        "black>=23.0.0,<24.0.0",
        "isort>=5.12.0,<6.0.0",
        "pre-commit>=3.0.0,<4.0.0",
        "sphinx>=7.0.0,<8.0.0",
        "sphinx-rtd-theme>=1.3.0,<2.0.0",
        "coverage>=7.0.0,<8.0.0",
        "bandit>=1.7.0,<2.0.0",
        "safety>=2.3.0,<3.0.0",
    ],
    
    # All optional features combined
    "all": [
        # GPU support
        "torch>=2.6.0,<3.0.0",
        "torchvision>=0.21.0,<1.0.0",
        "python-doctr[torch]>=0.11.0,<1.0.0",
        # Chinese OCR
        "paddleocr>=2.8.1,<3.0.0",
        "paddlepaddle>=2.6.1,<3.0.0",
        # LLM translation
        "transformers>=4.49.0,<5.0.0",
        "stanza>=1.7.0,<2.0.0",
        "jaconv>=0.3.4,<1.0.0",
        "fugashi>=1.4.0,<2.0.0",
        "unidic-lite>=1.0.8,<2.0.0",
        # Cloud services
        "azure-ai-vision-imageanalysis>=1.0.0b1,<2.0.0",
        "deepl>=1.16.1,<2.0.0",
        "deep-translator>=1.11.4,<2.0.0",
    ],
}

# Entry points for console scripts and GUI launchers
ENTRY_POINTS = {
    "console_scripts": [
        "comic-translate=comic_translate.main:main",
        "comic-translate-cli=comic_translate.cli:main",
        "comic-translate-batch=comic_translate.batch:main",
    ],
    "gui_scripts": [
        "comic-translate-gui=comic_translate.gui:main",
    ],
}

# Package classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: End Users/Desktop",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Multimedia :: Graphics",
    "Topic :: Text Processing :: Linguistic",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Environment :: X11 Applications :: Qt",
    "Environment :: Win32 (MS Windows)",
    "Environment :: MacOS X",
]

# Keywords for PyPI search
KEYWORDS = [
    "comic", "manga", "translation", "ocr", "image-processing",
    "text-detection", "machine-translation", "gui", "pyside6",
    "computer-vision", "deep-learning", "nlp", "inpainting"
]

# Python version requirement
PYTHON_REQUIRES = ">=3.8"

setup(
    name="comic-translate-mt",
    version=get_version(),
    author="Comic Translate MT Team",
    author_email="contact@comic-translate-mt.org",
    description="A comprehensive comic translation application with OCR, translation, and inpainting capabilities",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/comic-translate-mt/comic-translate-mt",
    project_urls={
        "Bug Reports": "https://github.com/comic-translate-mt/comic-translate-mt/issues",
        "Source": "https://github.com/comic-translate-mt/comic-translate-mt",
        "Documentation": "https://comic-translate-mt.readthedocs.io/",
        "Changelog": "https://github.com/comic-translate-mt/comic-translate-mt/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    include_package_data=True,
    package_data={
        "comic_translate": [
            "assets/*",
            "assets/**/*",
            "resources/*",
            "resources/**/*",
            "models/*",
            "models/**/*",
            "translations/*",
            "translations/**/*",
        ],
    },
    install_requires=CORE_REQUIREMENTS,
    extras_require=EXTRAS_REQUIRE,
    entry_points=ENTRY_POINTS,
    python_requires=PYTHON_REQUIRES,
    classifiers=CLASSIFIERS,
    keywords=", ".join(KEYWORDS),
    license="MIT",
    platforms=["any"],
    zip_safe=False,
    
    # Additional metadata
    options={
        "bdist_wheel": {
            "universal": False,
        },
    },
    
    # Command classes for custom setup commands
    cmdclass={},
    
    # Additional setup configuration
    setup_requires=[
        "setuptools>=65.0.0",
        "wheel>=0.37.0",
    ],
    
    # Test suite configuration
    test_suite="tests",
    tests_require=EXTRAS_REQUIRE["dev"],
)