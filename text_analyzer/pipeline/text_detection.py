import cv2
import numpy as np
import logging
import os
from pathlib import Path

class TextDetector:
    """Text region detector using OpenCV's EAST text detector"""
    
    def __init__(self, east_model_path=None, min_confidence=0.5):
        """
        Initialize text detector with EAST model
        
        Args:
            east_model_path: Path to pre-trained EAST text detector model
            min_confidence: Minimum confidence threshold for detections
        """
        self.logger = logging.getLogger(__name__)
        
        # Default EAST model path if not provided
        if east_model_path is None:
            # Default to looking in a models subdirectory
            root_dir = Path(__file__).parent.parent
            self.east_model_path = os.path.join(root_dir, "models", "frozen_east_text_detection.pb")
        else:
            self.east_model_path = east_model_path
            
        self.min_confidence = min_confidence
        self.logger.info(f"Text Detector initialized with confidence threshold: {min_confidence}")
    
    def _load_east_model(self):
        """Load EAST text detector model"""
        try:
            self.logger.info(f"Loading EAST model from {self.east_model_path}")
            net = cv2.dnn.readNet(self.east_model_path)
            return net
        except Exception as e:
            self.logger.error(f"Failed to load EAST model: {e}")
            self.logger.warning("Falling back to contour-based detection")
            return None
    
    def _detect_text_east(self, image):
        """
        Detect text regions using EAST text detector
        
        Args:
            image: Input image (preprocessed)
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        orig_h, orig_w = image.shape[:2]
        
        # EAST requires dimensions to be multiples of 32
        (new_w, new_h) = (320, 320)
        ratio_w = orig_w / float(new_w)
        ratio_h = orig_h / float(new_h)
        
        # Resize image to required dimensions
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(resized, 1.0, (new_w, new_h),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        
        # Load the pre-trained EAST model
        net = self._load_east_model()
        if net is None:
            return self._detect_text_contours(image)
        
        # Define output layer names for EAST
        layer_names = [
            "feature_fusion/Conv_7/Sigmoid",  # Confidence scores
            "feature_fusion/concat_3"         # Geometry (bounding boxes)
        ]
        
        # Forward pass
        net.setInput(blob)
        (scores, geometry) = net.forward(layer_names)
        
        # Decode predictions
        rects, confidences = self._decode_east_predictions(scores, geometry, self.min_confidence)
        
        # Apply non-maxima suppression
        boxes = []
        if len(rects) > 0:
            indices = cv2.dnn.NMSBoxes(rects, confidences, self.min_confidence, 0.3)
            
            for i in indices.flatten():
                # Get coordinates
                (x, y, w, h) = rects[i]
                
                # Scale back to original image dimensions
                x = int(x * ratio_w)
                y = int(y * ratio_h)
                w = int(w * ratio_w)
                h = int(h * ratio_h)
                
                boxes.append((x, y, w, h))
        
        return boxes
    
    def _decode_east_predictions(self, scores, geometry, min_confidence):
        """
        Decode EAST model predictions
        
        Args:
            scores: Confidence scores
            geometry: Geometry data for bounding boxes
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (rectangles, confidences)
        """
        (num_rows, num_cols) = scores.shape[2:4]
        rects = []
        confidences = []
        
        # Iterate through each prediction
        for y in range(0, num_rows):
            scores_data = scores[0, 0, y]
            x_data0 = geometry[0, 0, y]  # distances to top
            x_data1 = geometry[0, 1, y]  # distances to right
            x_data2 = geometry[0, 2, y]  # distances to bottom
            x_data3 = geometry[0, 3, y]  # distances to left
            angles_data = geometry[0, 4, y]  # rotation angles
            
            for x in range(0, num_cols):
                if scores_data[x] < min_confidence:
                    continue
                
                # Compute offset from prediction
                offset_x = x * 4.0
                offset_y = y * 4.0
                
                # Extract angle
                angle = angles_data[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                
                # Calculate dimensions and center of bounding box
                h = x_data0[x] + x_data2[x]
                w = x_data1[x] + x_data3[x]
                
                # Calculate bounding box coordinates
                end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
                end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
                start_x = int(end_x - w)
                start_y = int(end_y - h)
                
                rects.append((start_x, start_y, w, h))
                confidences.append(float(scores_data[x]))
        
        return (rects, confidences)
    
    def _detect_text_contours(self, image):
        """
        Alternative text detection using contour analysis
        Used as fallback when EAST model is not available
        
        Args:
            image: Input image (preprocessed)
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        self.logger.info("Using contour-based text detection")
        
        # If the image is not already binary, convert it
        if len(image.shape) > 2 or image.dtype != np.uint8:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            # Ensure black text on white background for contour detection
            if cv2.countNonZero(image) > (image.shape[0] * image.shape[1] / 2):
                binary = cv2.bitwise_not(image)
            else:
                binary = image
        
        # Apply morphological operations to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and extract bounding boxes
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio and size
            aspect_ratio = w / float(h)
            area = w * h
            
            # Typical text has aspect ratio > 1 and reasonable size
            if 0.2 < aspect_ratio < 15 and area > 100:
                boxes.append((x, y, w, h))
        
        return boxes
    
    def detect(self, image):
        """
        Detect text regions in an image
        
        Args:
            image: Input image (can be pre-processed or original)
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        self.logger.info("Detecting text regions")
        
        # Try using EAST detector first
        try:
            return self._detect_text_east(image)
        except Exception as e:
            self.logger.error(f"EAST detection failed: {e}")
            self.logger.info("Falling back to contour-based detection")
            return self._detect_text_contours(image)