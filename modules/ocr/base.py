from abc import ABC, abstractmethod
import numpy as np
import cv2
import base64
from typing import List, Optional, Dict, Any

from ..utils.textblock import TextBlock
from ..utils.exceptions import OCRError, ErrorCode, ErrorSeverity
from ..utils.logging_config import get_ocr_logger


class OCREngine(ABC):
    """
    Abstract base class for all OCR engines.
    Each OCR implementation should inherit from this class, implement the required methods,
    and use run() to execute OCR with validation, error handling, and logging.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the OCR engine with configuration validation.
        """
        self.logger = get_ocr_logger()
        self.config = kwargs or {}
        try:
            self.validate_config()
            self.logger.debug("OCR engine configuration validated", config=self.config)
        except OCRError:
            raise
        except Exception as e:
            raise OCRError(
                "OCR engine configuration validation failed",
                error_code=ErrorCode.OCR_ENGINE_INIT_FAILED,
                severity=ErrorSeverity.CRITICAL,
                engine_name=self.__class__.__name__,
                context={'config': self.config},
                cause=e
            )

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize the OCR engine with necessary parameters.
        
        Args:
            **kwargs: Engine-specific initialization parameters
        """
        pass

    @abstractmethod
    def process_image(self, img: np.ndarray, blk_list: List[TextBlock]) -> List[TextBlock]:
        """
        Process an image with OCR and update text blocks with recognized text.
        
        Args:
            img: Input image as numpy array
            blk_list: List of TextBlock objects to update with OCR text
            
        Returns:
            List of updated TextBlock objects with recognized text, with confidence scores
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check the health/status of the OCR engine (e.g., model loaded, GPU available).
        
        Returns:
            True if engine is healthy, False otherwise.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup resources used by the OCR engine (e.g., GPU context, temporary files).
        """
        pass

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate engine-specific configuration parameters.
        
        Raises:
            OCRError if configuration is invalid.
        """
        pass

    @abstractmethod
    def optimize_batch(self, blk_list: List[TextBlock]) -> List[TextBlock]:
        """
        Optimize text blocks for batch OCR processing (e.g., merge small blocks).
        
        Args:
            blk_list: List of TextBlock objects to optimize
        
        Returns:
            Optimized list of TextBlock objects.
        """
        pass

    @abstractmethod
    def manage_memory(self) -> None:
        """
        Perform memory management tasks, such as clearing caches or limiting GPU memory usage.
        """
        pass

    def run(
        self,
        img: np.ndarray,
        blk_list: List[TextBlock],
        confidence_threshold: float = 0.0
    ) -> List[TextBlock]:
        """
        Execute OCR on the given image and text blocks with validation, filtering, and logging.
        
        Args:
            img: Input image as numpy array
            blk_list: List of TextBlock objects with bounding boxes
            confidence_threshold: Minimum confidence score to keep a text block
        
        Returns:
            List of TextBlock objects with recognized text and confidence >= threshold.
        
        Raises:
            OCRError on validation or processing failures.
        """
        self.logger.info(
            "Starting OCR run",
            engine=self.__class__.__name__,
            image_shape=img.shape if isinstance(img, np.ndarray) else None
        )

        # Input validation
        if not isinstance(img, np.ndarray):
            raise OCRError(
                "Invalid image type provided",
                error_code=ErrorCode.OCR_INVALID_IMAGE,
                severity=ErrorSeverity.ERROR,
                engine_name=self.__class__.__name__
            )
        if not isinstance(blk_list, list) or not all(isinstance(b, TextBlock) for b in blk_list):
            raise OCRError(
                "Invalid block list provided",
                error_code=ErrorCode.OCR_INVALID_IMAGE,
                severity=ErrorSeverity.ERROR,
                engine_name=self.__class__.__name__
            )

        # Ensure source language is set
        for blk in blk_list:
            if not getattr(blk, "source_lang", None):
                raise OCRError(
                    "Source language not set for text block",
                    error_code=ErrorCode.OCR_INVALID_IMAGE,
                    severity=ErrorSeverity.ERROR,
                    engine_name=self.__class__.__name__,
                    context={'block_index': blk_list.index(blk)}
                )

        try:
            # Optimize blocks for batch processing
            optimized_blocks = self.optimize_batch(blk_list)
            # Perform OCR
            results = self.process_image(img, optimized_blocks)
            # Filter by confidence
            filtered = self.filter_by_confidence(results, confidence_threshold)
            self.logger.info(
                "OCR run completed",
                engine=self.__class__.__name__,
                items_processed=len(filtered)
            )
            return filtered
        except OCRError:
            # Already a structured exception
            raise
        except Exception as e:
            raise OCRError(
                f"OCR processing failed: {e}",
                error_code=ErrorCode.OCR_PROCESSING_FAILED,
                severity=ErrorSeverity.ERROR,
                engine_name=self.__class__.__name__,
                cause=e
            )

    @staticmethod
    def filter_by_confidence(
        blk_list: List[TextBlock],
        threshold: float
    ) -> List[TextBlock]:
        """
        Filter text blocks by confidence score.
        
        Args:
            blk_list: List of TextBlock objects with 'confidence' attribute
            threshold: Minimum confidence to keep
        
        Returns:
            Filtered list of TextBlock objects.
        """
        return [
            blk for blk in blk_list
            if getattr(blk, "confidence", 0.0) >= threshold
        ]

    @staticmethod
    def set_source_language(blk_list: List[TextBlock], lang_code: str) -> None:
        """
        Set source language code for all text blocks.
        
        Args:
            blk_list: List of TextBlock objects
            lang_code: Language code to set for source language
        """
        for blk in blk_list:
            blk.source_lang = lang_code

    @staticmethod
    def encode_image(image: np.ndarray, ext: str = '.jpg') -> str:
        """
        Encode an image as base64 string.
        
        Args:
            image: Image as numpy array
            ext: Image format extension (default is .jpg)
        
        Returns:
            Base64 encoded image string
        
        Raises:
            OCRError if encoding fails or invalid input.
        """
        if not isinstance(image, np.ndarray):
            raise OCRError(
                "Invalid image type for encoding",
                error_code=ErrorCode.OCR_INVALID_IMAGE,
                severity=ErrorSeverity.ERROR
            )
        if not isinstance(ext, str) or not ext.startswith('.'):
            raise OCRError(
                f"Invalid image extension '{ext}'",
                error_code=ErrorCode.OCR_INVALID_IMAGE,
                severity=ErrorSeverity.ERROR
            )
        try:
            success, img_buffer = cv2.imencode(ext, image)
            if not success or img_buffer is None:
                raise ValueError("cv2.imencode returned failure")
            return base64.b64encode(img_buffer).decode('utf-8')
        except Exception as e:
            raise OCRError(
                f"Failed to encode image: {e}",
                error_code=ErrorCode.OCR_PROCESSING_FAILED,
                severity=ErrorSeverity.ERROR,
                cause=e
            )