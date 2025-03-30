import cv2
import numpy as np
import logging
from config.preprocessing_config import PreprocessingConfig

class ImagePreprocessor:
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.config = config or PreprocessingConfig()
        self.logger.info("Image Preprocessor initialized")
    
    def _convert_to_grayscale(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def _apply_gaussian_blur(self, image):
        return cv2.GaussianBlur(
            image, 
            (self.config.gaussian_kernel_size, self.config.gaussian_kernel_size), 
            self.config.gaussian_sigma
        )
    
    def _apply_clahe(self, image):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit, 
            tileGridSize=self.config.clahe_grid_size
        )
        return clahe.apply(image)
    
    def _apply_binarization(self, image):
        """Apply adaptive thresholding for binarization"""
        if self.config.binarization_method == 'otsu':
            _, binary = cv2.threshold(
                image, 
                0, 
                255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:  # adaptive
            binary = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.config.adaptive_block_size,
                self.config.adaptive_c
            )
        return binary
    
    def _correct_skew(self, image):
        """Detect and correct text skew"""
        # Find all contours
        contours, _ = cv2.findContours(
            image, 
            cv2.RETR_LIST, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        # Filter contours by size
        contours = [c for c in contours if cv2.contourArea(c) > 50]
        
        if not contours:
            return image
        
        # Find minimum area rectangles
        angles = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            angle = rect[2]
            
            # Normalize angle
            if angle < -45:
                angle += 90
            
            angles.append(angle)
        
        # Calculate median angle
        if not angles:
            return image
            
        median_angle = np.median(angles)
        
        # Only correct if the skew is significant
        if abs(median_angle) < 0.5:
            return image
        
        # Rotate image to correct skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (w, h), 
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def process(self, image):
        """
        Apply preprocessing pipeline to an image
        
        Args:
            image: Input image (can be color or grayscale)
            
        Returns:
            Preprocessed image
        """
        self.logger.info("Preprocessing image")
        
        # Convert to grayscale
        gray = self._convert_to_grayscale(image)
        self.logger.debug("Converted to grayscale")
        
        # Apply Gaussian blur for noise reduction
        if self.config.use_gaussian:
            blurred = self._apply_gaussian_blur(gray)
            self.logger.debug("Applied Gaussian blur")
        else:
            blurred = gray
        
        # Apply CLAHE for contrast enhancement
        if self.config.use_clahe:
            enhanced = self._apply_clahe(blurred)
            self.logger.debug("Applied CLAHE")
        else:
            enhanced = blurred
        
        # Apply binarization
        if self.config.use_binarization:
            binary = self._apply_binarization(enhanced)
            self.logger.debug("Applied binarization")
        else:
            binary = enhanced
        
        # Correct skew if enabled
        if self.config.correct_skew:
            result = self._correct_skew(binary)
            self.logger.debug("Corrected skew")
        else:
            result = binary
        
        self.logger.info("Preprocessing complete")
        return result