import logging
import re
import string
from collections import Counter
import os
from pathlib import Path

# Try to import langdetect, but provide fallback options
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    logging.warning("langdetect not installed. Using basic language detection.")
    LANGDETECT_AVAILABLE = False

class LanguageIdentifier:
    """Language identification for text"""
    
    def __init__(self, fallback_lang='en'):
        """
        Initialize language identifier
        
        Args:
            fallback_lang: Fallback language code if detection fails
        """
        self.logger = logging.getLogger(__name__)
        self.fallback_lang = fallback_lang
        
        # Common language codes and their names
        self.language_names = {
            'en': 'English',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'ru': 'Russian',
            'ar': 'Arabic',
            'zh-cn': 'Chinese (Simplified)',
            'ja': 'Japanese',
            'ko': 'Korean',
            'hi': 'Hindi',
            'sw': 'Swahili'
        }
        
        self.logger.info("Language Identifier initialized")
    
    def _get_language_name(self, lang_code):
        """
        Convert language code to language name
        
        Args:
            lang_code: ISO language code
            
        Returns:
            Language name as string
        """
        return self.language_names.get(lang_code, f"Unknown ({lang_code})")
    
    def _basic_detect(self, text):
        """
        Basic language detection using character frequency analysis
        Used as fallback when langdetect is not available
        
        Args:
            text: Input text
            
        Returns:
            Predicted language code
        """
        # Very basic heuristics for common languages
        text = text.lower()
        
        # Check for language-specific characters
        if re.search(r'[а-яА-Я]', text):  # Cyrillic characters
            return 'ru'  # Russian
        elif re.search(r'[\u0600-\u06FF]', text):  # Arabic characters
            return 'ar'  # Arabic
        elif re.search(r'[\u3040-\u30FF]', text):  # Japanese characters
            return 'ja'  # Japanese
        elif re.search(r'[\u4E00-\u9FFF]', text):  # Chinese characters
            return 'zh-cn'  # Chinese
        elif re.search(r'[\uAC00-\uD7A3]', text):  # Korean characters
            return 'ko'  # Korean
        
        # For western languages, use common character frequencies and patterns
        if len(text) < 10:
            return self.fallback_lang  # Too short to be reliable
        
        # Count character frequencies to distinguish European languages
        char_count = Counter(c for c in text.lower() if c in string.ascii_lowercase)
        total = sum(char_count.values())
        
        if total == 0:
            return self.fallback_lang
        
        # Create frequency distribution
        freq = {char: count/total for char, count in char_count.items()}
        
        # Basic indicators for different languages
        if freq.get('ñ', 0) > 0.001:
            return 'es'  # Spanish
        elif freq.get('ß', 0) > 0.001:
            return 'de'  # German
        elif freq.get('q', 0) > 0.01 and freq.get('u', 0) > 0.05:
            return 'fr'  # French
        elif freq.get('j', 0) > 0.01:
            return 'nl'  # Dutch
        
        # Default to English
        return self.fallback_lang
    
    def identify_language(self, text):
        """
        Identify the language of a text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with language information:
            {
                'lang_code': ISO language code,
                'lang_name': Language name,
                'confidence': Confidence score (if available)
            }
        """
        self.logger.info("Identifying language")
        
        # Check if text is long enough
        if not text or len(text) < 5:
            self.logger.warning("Text too short for reliable language detection")
            return {
                'lang_code': self.fallback_lang,
                'lang_name': self._get_language_name(self.fallback_lang),
                'confidence': 0.0
            }
        
        try:
            if LANGDETECT_AVAILABLE:
                # Use langdetect if available
                lang_code = detect(text)
                self.logger.info(f"Detected language: {lang_code}")
                return {
                    'lang_code': lang_code,
                    'lang_name': self._get_language_name(lang_code),
                    'confidence': 0.8  # langdetect doesn't provide confidence scores
                }
            else:
                # Use basic detection as fallback
                lang_code = self._basic_detect(text)
                self.logger.info(f"Basic detected language: {lang_code}")
                return {
                    'lang_code': lang_code,
                    'lang_name': self._get_language_name(lang_code),
                    'confidence': 0.5  # Lower confidence for basic detection
                }
        except LangDetectException as e:
            self.logger.error(f"Language detection error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in language detection: {e}")
        
        # Fallback in case of error
        return {
            'lang_code': self.fallback_lang,
            'lang_name': self._get_language_name(self.fallback_lang),
            'confidence': 0.0
        }
    
    def get_supported_languages(self):
        """
        Get list of supported languages
        
        Returns:
            List of dictionaries with language information
        """
        return [
            {'code': code, 'name': name}
            for code, name in self.language_names.items()
        ]