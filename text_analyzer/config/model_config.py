class ModelConfig:
    """Configuration for OCR models and parameters"""
    
    def __init__(self):
        # Text detection parameters
        self.detection_confidence = 0.5
        self.east_model_path = None  # Will use the default path
        
        # Text recognition parameters
        self.languages = ['en']  # Default to English
        self.use_gpu = False  # Default to CPU for better compatibility
        self.recognition_min_confidence = 0.4  # Minimum confidence for text recognition
        
        # Language identification parameters
        self.langid_min_text_length = 10  # Minimum text length for language identification
        self.fallback_language = 'en'  # Default language if detection fails
    
    def set_languages(self, languages):
        """
        Set languages for OCR
        
        Args:
            languages: List of language codes supported by EasyOCR
        """
        self.languages = languages
    
    def enable_gpu(self, enable=True):
        """
        Enable or disable GPU acceleration
        
        Args:
            enable: Boolean to enable/disable GPU
        """
        self.use_gpu = enable
    
    def set_detection_confidence(self, confidence):
        """
        Set confidence threshold for text detection
        
        Args:
            confidence: Float value between 0 and 1
        """
        if 0 <= confidence <= 1:
            self.detection_confidence = confidence
    
    def set_recognition_confidence(self, confidence):
        """
        Set confidence threshold for text recognition
        
        Args:
            confidence: Float value between 0 and 1
        """
        if 0 <= confidence <= 1:
            self.recognition_min_confidence = confidence