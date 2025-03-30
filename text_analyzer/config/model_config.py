class ModelConfig:
  
    def __init__(self):
        # Text detection parameters
        self.detection_confidence = 0.5
        self.east_model_path = None  # Will use the default path
        
        # Text recognition parameters
        self.languages = ['en']  
        self.use_gpu = False 
        self.recognition_min_confidence = 0.4 
        
        # Language identification parameters
        self.langid_min_text_length = 10  
        self.fallback_language = 'en' 
    def set_languages(self, languages):
        self.languages = languages
    
    def enable_gpu(self, enable=True):
        self.use_gpu = enable
    
    def set_detection_confidence(self, confidence):
        if 0 <= confidence <= 1:
            self.detection_confidence = confidence
    
    def set_recognition_confidence(self, confidence):
        if 0 <= confidence <= 1:
            self.recognition_min_confidence = confidence