import logging
import cv2
import numpy as np
import os
from pathlib import Path
import sys

# Add try-except to handle potential import errors
try:
    import easyocr
except ImportError:
    logging.error("EasyOCR not installed. Please install it using: pip install easyocr")
    easyocr = None

class TextRecognizer:
    """Text recognition class using EasyOCR"""
    
    def __init__(self, languages=None, gpu=False):
        """
        Initialize text recognizer
        
        Args:
            languages: List of languages to recognize (default: ['en'])
            gpu: Whether to use GPU acceleration
        """
        self.logger = logging.getLogger(__name__)
        self.languages = languages or ['en']
        self.gpu = gpu
        self.reader = None
        
        self.logger.info(f"Text Recognizer initialized with languages: {self.languages}")
        
        # Lazy initialization of EasyOCR reader
        # Will be loaded on first use to avoid long startup times
    
    def _initialize_reader(self):
        """Initialize EasyOCR reader if not already initialized"""
        if self.reader is None:
            if easyocr is None:
                self.logger.error("EasyOCR is not available. Please install it using: pip install easyocr")
                raise ImportError("EasyOCR is required for text recognition")
            
            try:
                self.logger.info(f"Initializing EasyOCR with languages: {self.languages}")
                self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
                self.logger.info("EasyOCR initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize EasyOCR: {e}")
                raise
    
    def _crop_regions(self, image, boxes):
        """
        Crop text regions from image based on bounding boxes
        
        Args:
            image: Original image
            boxes: List of bounding boxes (x, y, w, h)
            
        Returns:
            List of cropped regions (images)
        """
        regions = []
        
        for (x, y, w, h) in boxes:
            # Add padding around the region
            pad = 5
            x_start = max(0, x - pad)
            y_start = max(0, y - pad)
            x_end = min(image.shape[1], x + w + pad)
            y_end = min(image.shape[0], y + h + pad)
            
            # Crop region
            region = image[y_start:y_end, x_start:x_end]
            
            # Only add non-empty regions
            if region.size > 0:
                regions.append((region, (x, y, w, h)))
        
        return regions
    
    def recognize_text(self, image, boxes=None):
        """
        Recognize text in the image or in specified regions
        
        Args:
            image: Input image (original or preprocessed)
            boxes: Optional list of bounding boxes to focus on
                  If None, the entire image is processed
            
        Returns:
            List of dictionaries with recognized text and positions
            [{'text': str, 'position': (x, y, w, h), 'confidence': float}, ...]
        """
        self.logger.info("Recognizing text")
        
        # Initialize reader on first use
        self._initialize_reader()
        
        results = []
        
        try:
            if boxes:
                # Recognize text in specific regions
                regions = self._crop_regions(image, boxes)
                
                for region, (x, y, w, h) in regions:
                    # Skip too small regions
                    if region.shape[0] < 5 or region.shape[1] < 5:
                        continue
                    
                    # Recognize text in the cropped region
                    recognition_result = self.reader.readtext(region)
                    
                    # Process recognition results
                    for (pts, text, conf) in recognition_result:
                        # Adjust coordinates to original image
                        results.append({
                            'text': text,
                            'position': (x, y, w, h),
                            'confidence': conf
                        })
            else:
                # Recognize text in the entire image
                recognition_result = self.reader.readtext(image)
                
                # Process recognition results
                for (pts, text, conf) in recognition_result:
                    # Calculate bounding box from points
                    min_x = min(pt[0] for pt in pts)
                    min_y = min(pt[1] for pt in pts)
                    max_x = max(pt[0] for pt in pts)
                    max_y = max(pt[1] for pt in pts)
                    
                    x = int(min_x)
                    y = int(min_y)
                    w = int(max_x - min_x)
                    h = int(max_y - min_y)
                    
                    results.append({
                        'text': text,
                        'position': (x, y, w, h),
                        'confidence': conf
                    })
        
        except Exception as e:
            self.logger.error(f"Error during text recognition: {e}")
            # Return empty results in case of error
            return []
        
        self.logger.info(f"Recognized {len(results)} text regions")
        return results
    
    def visualize_results(self, image, recognition_results):
        """
        Create a visualization of recognition results
        
        Args:
            image: Original image
            recognition_results: Results from recognize_text()
            
        Returns:
            Image with visualized text recognition results
        """
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        for result in recognition_results:
            # Extract information
            text = result['text']
            (x, y, w, h) = result['position']
            conf = result.get('confidence', 0)
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Prepare text to display
            display_text = f"{text} ({conf:.2f})"
            
            # Determine text position
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            
            # Display text
            cv2.putText(vis_image, display_text, (x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return vis_image