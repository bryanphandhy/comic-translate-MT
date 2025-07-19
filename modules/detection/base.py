from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from ..utils.textblock import TextBlock
from .utils.general import does_rectangle_fit, do_rectangles_overlap, \
      filter_and_fix_bboxes, merge_overlapping_boxes

from ..utils.exceptions import DetectionError, ErrorCode
from ..utils.logging_config import get_detection_logger, performance_monitor
from ..utils.cache import cache_result


class DetectionEngine(ABC):
    """
    Abstract base class for all detection engines.
    Each model implementation should inherit from this class.
    """
    
    # Default thresholds and limits
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for considering text boxes
    DEFAULT_MAX_PIXELS = 50_000_000     # Maximum allowed pixels for processing
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize the detection model with necessary parameters.
        
        Args:
            **kwargs: Engine-specific initialization parameters
        """
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[TextBlock]:
        """
        Detect text blocks in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of TextBlock objects with detected regions
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check the health of the detection model (e.g., GPU memory, model loaded).
        
        Returns:
            True if the model is healthy, False otherwise.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup resources used by the detection model (e.g., GPU memory).
        """
        pass
        
    @cache_result("detection_text_blocks")
    @performance_monitor("detection.create_text_blocks")
    def create_text_blocks(self, image: np.ndarray, 
                           text_boxes: np.ndarray,
                           bubble_boxes: Optional[np.ndarray] = None) -> list[TextBlock]:
        """
        Convert raw detection boxes into structured TextBlock objects,
        matching text with bubbles and classifying free text.
        
        This method includes validation, filtering by confidence, and caching.
        """
        logger = get_detection_logger()
        context = {"model_name": getattr(self, "model_name", None)}
        
        # Validate image input
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            message = "Invalid or empty image provided to create_text_blocks"
            logger.error(message, **context)
            raise DetectionError(
                message,
                error_code=ErrorCode.DETECTION_INVALID_INPUT,
                context=context
            )
        height, width = image.shape[:2]
        if height * width > self.DEFAULT_MAX_PIXELS:
            message = f"Image too large ({width}x{height}). Exceeds maximum allowed pixels."
            logger.warning(message, **context)
            raise DetectionError(
                message,
                error_code=ErrorCode.DETECTION_INVALID_INPUT,
                context=context
            )
        
        # Validate and process text_boxes
        if text_boxes is None or not isinstance(text_boxes, np.ndarray):
            message = "Invalid text_boxes input to create_text_blocks"
            logger.error(message, **context)
            raise DetectionError(
                message,
                error_code=ErrorCode.DETECTION_INVALID_INPUT,
                context=context
            )
        try:
            # Filter based on confidence if available (assuming score in 5th column)
            if text_boxes.ndim == 2 and text_boxes.shape[1] >= 5:
                scores = text_boxes[:, 4]
                mask = scores >= self.DEFAULT_CONFIDENCE_THRESHOLD
                if not np.any(mask):
                    message = "No text boxes above confidence threshold"
                    logger.warning(message, **context)
                    raise DetectionError(
                        message,
                        error_code=ErrorCode.DETECTION_NO_TEXT_FOUND,
                        context=context
                    )
                text_boxes = text_boxes[mask][:, :4]
            else:
                text_boxes = text_boxes[:, :4]
        except DetectionError:
            # Re-raise known detection errors
            raise
        except Exception as e:
            logger.error("Error filtering text_boxes", exception=e, **context)
            raise DetectionError(
                "Error filtering text boxes",
                error_code=ErrorCode.DETECTION_INVALID_INPUT,
                context=context,
                cause=e
            )
        
        # Validate and process bubble_boxes
        if bubble_boxes is not None:
            if not isinstance(bubble_boxes, np.ndarray):
                logger.warning("Invalid bubble_boxes input; setting to empty", **context)
                bubble_boxes = np.array([])
            else:
                bubble_boxes = bubble_boxes[:, :4] if bubble_boxes.ndim == 2 else np.array([])
        else:
            bubble_boxes = np.array([])
        
        # Preprocessing of boxes
        try:
            text_boxes = filter_and_fix_bboxes(text_boxes, image.shape)
            bubble_boxes = filter_and_fix_bboxes(bubble_boxes, image.shape)
            text_boxes = merge_overlapping_boxes(text_boxes)
        except Exception as e:
            logger.error("Error preprocessing bounding boxes", exception=e, **context)
            raise DetectionError(
                "Error preprocessing bounding boxes",
                error_code=ErrorCode.DETECTION_INVALID_INPUT,
                context=context,
                cause=e
            )
        
        text_blocks: list[TextBlock] = []
        text_matched = [False] * len(text_boxes)
        
        # Create TextBlock objects
        try:
            for txt_idx, txt_box in enumerate(text_boxes):
                # If no bubble boxes, classify as free text
                if len(bubble_boxes) == 0:
                    text_blocks.append(
                        TextBlock(
                            text_bbox=txt_box,
                            text_class='text_free',
                        )
                    )
                    continue
                # Otherwise, match to bubbles
                matched = False
                for bble_box in bubble_boxes:
                    if bble_box is None:
                        continue
                    if does_rectangle_fit(bble_box, txt_box) or do_rectangles_overlap(bble_box, txt_box):
                        text_blocks.append(
                            TextBlock(
                                text_bbox=txt_box,
                                bubble_bbox=bble_box,
                                text_class='text_bubble',
                            )
                        )
                        text_matched[txt_idx] = True
                        matched = True
                        break
                if not matched:
                    text_blocks.append(
                        TextBlock(
                            text_bbox=txt_box,
                            text_class='text_free',
                        )
                    )
        except Exception as e:
            logger.error("Error creating text blocks", exception=e, **context)
            raise DetectionError(
                "Error creating text blocks",
                error_code=ErrorCode.DETECTION_INFERENCE_FAILED,
                context=context,
                cause=e
            )
        
        if not text_blocks:
            logger.info("No text blocks created", **context)
        else:
            logger.debug(f"Created {len(text_blocks)} text blocks", **context)
        
        return text_blocks