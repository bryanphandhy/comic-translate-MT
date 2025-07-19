from abc import ABC, abstractmethod
from typing import Any, List
import numpy as np
import json

from ..utils.textblock import TextBlock
from ..utils.exceptions import TranslationError
from ..utils.logging_config import get_translation_logger


class TranslationEngine(ABC):
    """
    Abstract base class for all translation engines.
    Defines common interface and utility methods.
    """
    
    def __init__(self) -> None:
        """
        Base constructor for translation engines.
        Sets up logger and initialization flags.
        """
        self.logger = get_translation_logger()
        self._initialized: bool = False
        self._source_lang: str = ""
        self._target_lang: str = ""
    
    @abstractmethod
    def initialize(self, settings: Any, source_lang: str, target_lang: str, **kwargs) -> None:
        """
        Initialize the translation engine with necessary parameters.
        
        Args:
            settings: Settings object with credentials and config
            source_lang: Source language name
            target_lang: Target language name
            **kwargs: Engine-specific initialization parameters
        Raises:
            TranslationError: If initialization fails or parameters are invalid
        """
        try:
            # Validate settings presence
            if settings is None:
                raise ValueError("Settings object is required for initialization")
            # Normalize language codes
            self._source_lang = self.get_language_code(source_lang)
            self._target_lang = self.get_language_code(target_lang)
            # Validate language pair support
            self.validate_language_pair(self._source_lang, self._target_lang)
            # Log initialization
            self.logger.info(
                "Initializing translation engine",
                engine=self.__class__.__name__,
                source_language=self._source_lang,
                target_language=self._target_lang
            )
        except Exception as e:
            raise TranslationError(
                f"Initialization failed: {e}",
                engine_name=self.__class__.__name__,
                source_language=source_lang,
                target_language=target_lang,
                cause=e
            )
        self._initialized = True
    
    def get_language_code(self, language: str) -> str:
        """
        Get standardized language code from language name.
        
        Args:
            language: Language name
            
        Returns:
            Standardized language code
        """
        from ..utils.pipeline_utils import get_language_code
        try:
            code = get_language_code(language)
            return code
        except Exception as e:
            raise TranslationError(
                f"Invalid language '{language}': {e}",
                engine_name=self.__class__.__name__,
                cause=e
            )
    
    def validate_language_pair(self, source_lang: str, target_lang: str) -> None:
        """
        Validate that the language pair is supported by this engine.
        Default implementation allows all pairs.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
        
        Raises:
            TranslationError: If the language pair is not supported
        """
        # Subclasses may override for specific support
        return
    
    def preprocess_text(self, blk_text: str, source_lang_code: str) -> str:
        """
        Preprocess text based on language:
        - Remove spaces for Chinese and Japanese languages
        - Remove all newline/carriage-return characters
        - Keep original text for other languages (aside from newline removal)
        - Preserve special characters
        
        Args:
            blk_text: The input text to process
            source_lang_code: Language code of the source text
        
        Returns:
            Processed text
        """
        if blk_text is None:
            raise TranslationError(
                "Cannot preprocess None text",
                engine_name=self.__class__.__name__,
                source_language=self._source_lang,
                target_language=self._target_lang
            )
        # Remove newline and carriage‐return characters
        text = blk_text.replace('\r', '').replace('\n', '')
        source = source_lang_code.lower()
        # Remove spaces for Chinese/Japanese
        if 'zh' in source or source == 'ja':
            return text.replace(' ', '')
        return text
    
    def serialize(self, blk_list: List[TextBlock]) -> str:
        """
        Serialize list of TextBlocks to JSON string.
        
        Args:
            blk_list: List of TextBlock objects
        
        Returns:
            JSON string of serialized blocks
        
        Raises:
            TranslationError: On serialization failure
        """
        try:
            data = [blk.to_dict() for blk in blk_list]
            return json.dumps(data, ensure_ascii=False)
        except Exception as e:
            raise TranslationError(
                f"Serialization failed: {e}",
                engine_name=self.__class__.__name__,
                cause=e
            )
    
    def deserialize(self, serialized: str) -> List[TextBlock]:
        """
        Deserialize JSON string to list of TextBlocks.
        
        Args:
            serialized: JSON string representing a list of TextBlock dicts
        
        Returns:
            List of TextBlock objects
        
        Raises:
            TranslationError: On deserialization failure
        """
        try:
            data = json.loads(serialized)
            blocks: List[TextBlock] = []
            for item in data:
                blk = TextBlock.from_dict(item)
                blocks.append(blk)
            return blocks
        except Exception as e:
            raise TranslationError(
                f"Deserialization failed: {e}",
                engine_name=self.__class__.__name__,
                cause=e
            )
    
    def evaluate_confidence(self, blk: TextBlock) -> float:
        """
        Evaluate confidence of a translated text block.
        
        Default implementation returns existing confidence or 1.0.
        
        Args:
            blk: TextBlock after translation
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        return getattr(blk, 'confidence', 1.0)
    
    @abstractmethod
    def translate(self, *args, **kwargs) -> List[TextBlock]:
        """
        Abstract translate method to be implemented by subclasses.
        
        The signature may vary: TraditionalTranslation uses translate(blk_list),
        LLMTranslation uses translate(blk_list, image, extra_context).
        
        Returns:
            List of updated TextBlock objects with translations
        
        Raises:
            TranslationError: On translation failure
        """
        pass
    
    def translate_blocks(self, blk_list: List[TextBlock], *args, **kwargs) -> List[TextBlock]:
        """
        Template method for translation that wraps actual translate call with
        validation, rate limiting, logging, and confidence evaluation.
        
        Args:
            blk_list: List of TextBlock objects to translate
            *args, **kwargs: Additional parameters for specific engines
        
        Returns:
            List of TextBlock objects with translations
        
        Raises:
            TranslationError: On any failure during translation process
        """
        if not self._initialized:
            raise TranslationError(
                "Engine not initialized",
                engine_name=self.__class__.__name__
            )
        # Validate input list
        if not isinstance(blk_list, list):
            raise TranslationError(
                "blk_list must be a list of TextBlock",
                engine_name=self.__class__.__name__
            )
        for blk in blk_list:
            if not isinstance(blk, TextBlock):
                raise TranslationError(
                    "Invalid item in blk_list; expected TextBlock instances",
                    engine_name=self.__class__.__name__
                )
        try:
            # Preprocess and apply rate limiting
            total_length = sum(
                len(self.preprocess_text(blk.text, self._source_lang))
                for blk in blk_list
            )
            self.rate_limit(total_length)
            # Log start
            self.logger.info(
                "Starting translation",
                engine=self.__class__.__name__,
                items=len(blk_list)
            )
            # Call subclass translate
            translated = self.translate(blk_list, *args, **kwargs)
            # Validate output
            if not isinstance(translated, list):
                raise TranslationError(
                    "Translate method must return a list of TextBlock",
                    engine_name=self.__class__.__name__
                )
            for blk in translated:
                if not isinstance(blk, TextBlock):
                    raise TranslationError(
                        "Translate output contains non-TextBlock items",
                        engine_name=self.__class__.__name__
                    )
                blk.confidence = self.evaluate_confidence(blk)
            # Log completion
            self.logger.info(
                "Translation completed",
                engine=self.__class__.__name__,
                items=len(translated)
            )
            return translated
        except TranslationError:
            raise
        except Exception as e:
            raise TranslationError(
                f"Translation failed: {e}",
                engine_name=self.__class__.__name__,
                cause=e
            )
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the engine is healthy and ready for translation.
        
        Returns:
            True if healthy, False otherwise
        
        Raises:
            TranslationError: On health check failure
        """
        pass
    
    @abstractmethod
    def rate_limit(self, text_length: int) -> None:
        """
        Enforce rate limit based on text length or API quotas.
        
        Args:
            text_length: Number of characters to translate
        
        Raises:
            TranslationError: If rate limit exceeded
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up engine resources (e.g., close connections, free GPU memory).
        
        Raises:
            TranslationError: On cleanup failure
        """
        pass
    
    @abstractmethod
    def optimize_batch(self, blk_list: List[TextBlock]) -> List[TextBlock]:
        """
        Optimize batch of TextBlocks for translation (e.g., grouping or chunking).
        
        Args:
            blk_list: Original list of TextBlock objects
        
        Returns:
            Optimized list of TextBlock objects for translation
        
        Raises:
            TranslationError: On optimization failure
        """
        pass


class TraditionalTranslation(TranslationEngine):
    """Base class for traditional translation engines (non-LLM)."""
    
    @abstractmethod
    def translate(self, blk_list: List[TextBlock]) -> List[TextBlock]:
        """
        Translate text blocks using non-LLM translators.
        
        Args:
            blk_list: List of TextBlock objects containing text to translate
            
        Returns:
            List of updated TextBlock objects with translations
        
        Raises:
            TranslationError: On translation failure
        """
        pass

    def preprocess_language_code(self, lang_code: str) -> str:
        """
        Preprocess language codes to match the specific translation API requirements.
        By default, returns the original language code.
        
        Args:
            lang_code: The language code to preprocess
            
        Returns:
            Preprocessed language code supported by the translation API
        """
        return lang_code  # Default implementation just returns the original code


class LLMTranslation(TranslationEngine):
    """Base class for LLM-based translation engines."""
    
    @abstractmethod
    def translate(self, blk_list: List[TextBlock], image: np.ndarray, extra_context: str) -> List[TextBlock]:
        """
        Translate text blocks using LLM.
        
        Args:
            blk_list: List of TextBlock objects containing text to translate
            image: Image as numpy array (for context)
            extra_context: Additional context information for translation
            
        Returns:
            List of updated TextBlock objects with translations
        
        Raises:
            TranslationError: On translation failure
        """
        pass
    
    def get_system_prompt(self, source_lang: str, target_lang: str) -> str:
        """
        Get system prompt for LLM translation.
        
        Args:
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Formatted system prompt
        """
        return (
            f"You are an expert translator who translates {source_lang} to {target_lang}. "
            "You pay attention to style, formality, idioms, slang etc and try to convey it "
            f"in the way a {target_lang} speaker would understand. "
            "BE MORE NATURAL. NEVER USE 당신, 그녀, 그 or its Japanese equivalents. "
            "Specifically, you will be translating text OCR'd from a comic. The OCR is not "
            "perfect and as such you may receive text with typos or other mistakes. "
            "To aid you and provide context, You may be given the image of the page and/or "
            "extra context about the comic. You will be given a json string of the detected "
            "text blocks and the text to translate. Return the json string with the texts "
            "translated. DO NOT translate the keys of the json. For each block: "
            f"- If it's already in {target_lang} or looks like gibberish, OUTPUT IT AS IT IS instead "
            "- DO NOT give explanations. Do Your Best! I'm really counting on you."
        )