# The Complete Guide to EasyOCR
## For Vision-Based Reading Applications

*Created: March 2, 2025*

---

## Table of Contents

1. [Introduction to EasyOCR](#introduction-to-easyocr)
2. [Installation and Setup](#installation-and-setup)
3. [Basic Usage](#basic-usage)
4. [Core Functions in Detail](#core-functions-in-detail)
5. [Parameter Explanations](#parameter-explanations)
6. [Advanced Usage](#advanced-usage)
7. [Optimizing Performance](#optimizing-performance)
8. [Integration in Vision Pipelines](#integration-in-vision-pipelines)
9. [Real-World Examples](#real-world-examples)
10. [Troubleshooting](#troubleshooting)
11. [Best Practices](#best-practices)

---

## Introduction to EasyOCR

EasyOCR is a Python package that allows you to extract text from images. It's built on PyTorch and provides a simple API while supporting 80+ languages.

### Key Features
- **User-Friendly**: Simple API with just a few lines of code to get started
- **Multi-Language Support**: Works with over 80 languages
- **Flexible Format Recognition**: Handles various text fonts, sizes, and orientations
- **GPU Acceleration**: Utilizes GPU for faster processing
- **Accuracy**: High accuracy rates compared to other open-source OCR solutions
- **Active Development**: Regular updates and improvements

### When to Use EasyOCR
- Document digitization
- Assistive technology (e.g., reading glasses for visually impaired)
- Automated data entry from images
- Real-time text recognition in videos
- Content extraction from screenshots
- Sign and label reading for autonomous systems

---

## Installation and Setup

### Basic Installation
```python
pip install easyocr
```

### With GPU Support (Recommended)
```python
# For CUDA support
pip install easyocr torch torchvision cudatoolkit
```

### Verify Installation
```python
import easyocr
print(easyocr.__version__)

# Check if GPU is available
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### Dependencies
EasyOCR depends on:
- PyTorch
- OpenCV
- Numpy
- Pillow
- scikit-image

These should be automatically installed during the EasyOCR installation.

---

## Basic Usage

### Initializing the Reader
```python
import easyocr

# Single language
reader = easyocr.Reader(['en'])  # English

# Multiple languages
reader = easyocr.Reader(['en', 'fr'])  # English and French

# With GPU acceleration
reader = easyocr.Reader(['en'], gpu=True)
```

### Reading Text from an Image
```python
# From a file path
results = reader.readtext('path/to/your/image.jpg')

# From a loaded image (OpenCV/Numpy array)
import cv2
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
results = reader.readtext(image)
```

### Processing the Results
```python
# Default output format (detail=1)
for (bbox, text, prob) in results:
    print(f"Text: {text}")
    print(f"Confidence: {prob:.2f}")
    print(f"Bounding box: {bbox}")
    
    # Draw bounding box on the image (for visualization)
    top_left = tuple(map(int, bbox[0]))
    bottom_right = tuple(map(int, bbox[2]))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# Display the image with bounding boxes
cv2.imshow('Detected Text', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
```

### Simplified Results
```python
# Get only the text (detail=0)
text_only = reader.readtext('image.jpg', detail=0)
print(text_only)  # List of detected text strings
```

---

## Core Functions in Detail

### `Reader` Class

The `Reader` class is the main interface for EasyOCR. It handles model loading and text detection/recognition.

```python
reader = easyocr.Reader(
    lang_list,                 # List of language codes
    gpu=True,                  # Use GPU if available
    model_storage_directory=None,  # Where to store model files
    download_enabled=True,     # Allow downloading models
    detector=True,             # Use text detector
    recognizer=True,           # Use text recognizer
    verbose=True,              # Show progress messages
    quantize=True,             # Use quantized models (faster, less memory)
    cudnn_benchmark=False      # Optimize for consistent input sizes
)
```

#### Key Parameters:

- **`lang_list`**: List of language codes (ISO 639-1)
  - Common options: 'en' (English), 'fr' (French), 'es' (Spanish), 'de' (German), 'zh' (Chinese)
  - Full list available at the [EasyOCR GitHub repository](https://github.com/JaidedAI/EasyOCR)

- **`gpu`**: Whether to use GPU acceleration
  - `True`: Use GPU if available (recommended)
  - `False`: Use CPU only

- **`verbose`**: Whether to show progress messages
  - Useful to see what's happening during model loading and inference

- **`quantize`**: Whether to use quantized models
  - Reduces model size and speeds up inference with slight accuracy reduction

### `readtext()` Method

The main method for text detection and recognition:

```python
results = reader.readtext(
    image,                      # Path or numpy array
    decoder='greedy',           # 'greedy' or 'beamsearch'
    beamWidth=5,                # Used with beamsearch
    batch_size=1,               # Larger batches for GPU efficiency
    workers=0,                  # CPU workers for preprocessing
    allowlist=None,             # Restrict to these characters
    blocklist=None,             # Exclude these characters
    detail=1,                   # 0: text only, 1: with position and confidence
    paragraph=False,            # Combine text into paragraphs
    min_size=20,                # Minimum text box size
    rotation_info=None,         # Text rotation handling
    text_threshold=0.7,         # Confidence threshold
    link_threshold=0.4,         # Text grouping threshold
    canvas_size=2560,           # Maximum image dimension
    mag_ratio=1,                # Image magnification ratio
    slope_ths=0.1,              # Maximum text line slope
    ycenter_ths=0.5,            # Vertical center threshold for text grouping
    height_ths=0.5,             # Height threshold for text grouping
    width_ths=0.5,              # Width threshold for text grouping
    add_margin=0.1              # Margin around text box
)
```

---

## Parameter Explanations

Let's break down each important parameter in simple terms:

### Essential Parameters

#### `image`
- **What it is**: The image containing text you want to read
- **Format**: File path (string) or numpy array (loaded image)
- **Example**: `reader.readtext("receipt.jpg")` or `reader.readtext(frame)`
- **Tips**: 
  - If passing a numpy array from OpenCV, convert BGR to RGB first
  - Images should be reasonably clear with good contrast

#### `detail`
- **What it is**: Controls how much information you get back
- **Options**:
  - `1` (default): Returns position, text, and confidence score
  - `0`: Returns just the text
- **Example**: `reader.readtext(image, detail=0)`
- **Use when**: You only need the text content and not its position or confidence

#### `text_threshold`
- **What it is**: How confident EasyOCR needs to be before accepting text
- **Range**: 0.0 to 1.0 (higher = more strict)
- **Default**: 0.7
- **Example**: `reader.readtext(image, text_threshold=0.8)`
- **Tips**:
  - Higher values give fewer but more accurate results
  - Lower values catch more text but include more errors
  - Start at 0.7 and adjust based on your needs

#### `paragraph`
- **What it is**: Whether to group nearby text into paragraphs
- **Options**: `True` or `False` (default)
- **Example**: `reader.readtext(image, paragraph=True)`
- **Use when**: 
  - Converting to speech (sounds more natural)
  - Preserving document structure
  - Processing articles or structured text

#### `allowlist` and `blocklist`
- **What they are**: Limit which characters to recognize or ignore
- **Format**: String of characters
- **Examples**: 
  - `reader.readtext(image, allowlist='0123456789')` (only numbers)
  - `reader.readtext(image, blocklist='!@#$%^&*')` (no special characters)
- **Use when**:
  - Reading specific formats (e.g., license plates, serial numbers)
  - Filtering out unwanted characters

### Advanced Parameters

#### `decoder`
- **What it is**: Method used to decode text
- **Options**: 
  - `'greedy'` (faster, default)
  - `'beamsearch'` (more accurate but slower)
- **Example**: `reader.readtext(image, decoder='beamsearch')`
- **Tips**:
  - Use 'greedy' for real-time applications
  - Use 'beamsearch' for document digitization where accuracy is critical

#### `batch_size`
- **What it is**: How many text regions to process at once
- **Default**: 1
- **Example**: `reader.readtext(image, batch_size=4)`
- **Tips**:
  - Larger values are faster on GPU but use more memory
  - Start with 1, then try 2, 4, 8 to find the sweet spot for your hardware
  - For real-time applications, balance between speed and memory usage

#### `min_size`
- **What it is**: Smallest text size to detect (in pixels)
- **Default**: 20 pixels
- **Example**: `reader.readtext(image, min_size=10)`
- **Tips**:
  - Lower values catch smaller text but may introduce more errors
  - Higher values focus on larger, typically more important text
  - Adjust based on your image resolution and text size

#### `rotation_info`
- **What it is**: Helps with rotated text
- **Format**: List of angles to check
- **Example**: `reader.readtext(image, rotation_info=[90, 180, 270])`
- **Use when**: 
  - Processing documents that might be scanned upside down
  - Reading text in various orientations (like signs in photos)
  - Note: Checking multiple rotations slows down processing significantly

#### `canvas_size`
- **What it is**: Maximum size for processing 
- **Default**: 2560 pixels
- **Example**: `reader.readtext(image, canvas_size=1280)`
- **Tips**:
  - Smaller values speed up processing but might miss details
  - For high-resolution images, reducing this can significantly speed up processing
  - For low-resolution images, keeping default is fine

#### `mag_ratio`
- **What it is**: How much to enlarge the image before processing
- **Default**: 1.0 (no change)
- **Example**: `reader.readtext(image, mag_ratio=1.5)`
- **Tips**:
  - Larger values help with small text but slow down processing
  - Useful for images with very small text
  - Values between 1.0 and 2.0 are typically most effective

### Grouping Parameters

These parameters control how EasyOCR groups detected characters into words and lines:

#### `link_threshold`
- **What it is**: Threshold for grouping text boxes
- **Default**: 0.4
- **Higher values**: Less grouping (more separate text boxes)
- **Lower values**: More aggressive grouping

#### `slope_ths`, `ycenter_ths`, `height_ths`, `width_ths`
- **What they are**: Thresholds for how text is grouped into lines and paragraphs
- **When to adjust**: When text layout is unusual (very slanted, varying sizes)
- **Default values** work well for most standard documents

---

## Advanced Usage

### Character Filtering

```python
# Only detect numbers
results = reader.readtext(image, allowlist='0123456789')

# Only detect alphanumeric characters (no symbols)
results = reader.readtext(image, allowlist='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

# Exclude certain characters
results = reader.readtext(image, blocklist='!@#$%^&*')
```

### Handling Rotated Text

```python
# Check for text in multiple orientations
results = reader.readtext(image, rotation_info=[0, 90, 180, 270])

# For pages that might be scanned upside down
results = reader.readtext(image, rotation_info=[0, 180])
```

### Paragraph Formation

```python
# Group text into paragraphs for more natural reading
results = reader.readtext(image, paragraph=True)

# Combine with confidence threshold
results = reader.readtext(image, paragraph=True, text_threshold=0.8)
```

### Beam Search Decoder

```python
# More accurate but slower decoding
results = reader.readtext(
    image, 
    decoder='beamsearch',
    beamWidth=5  # Higher = more accurate but slower
)
```

### Working with Low-Resolution Text

```python
# For small or low-quality text
results = reader.readtext(
    image,
    min_size=10,        # Detect smaller text
    mag_ratio=1.5,      # Enlarge image before processing
    text_threshold=0.5  # Lower confidence threshold
)
```

---

## Optimizing Performance

### Speed Optimization

For real-time applications (like your glasses project):

```python
# Fastest configuration
reader = easyocr.Reader(['en'], gpu=True, quantize=True)

results = reader.readtext(
    image,
    decoder='greedy',       # Faster decoding
    batch_size=4,           # Process multiple regions at once
    canvas_size=1280,       # Limit maximum image size
    paragraph=False,        # Skip paragraph formation
    min_size=20,            # Ignore very small text
    mag_ratio=1.0           # No magnification
)
```

### Accuracy Optimization

For document digitization or archival purposes:

```python
# Most accurate configuration
reader = easyocr.Reader(['en'], gpu=True, quantize=False)

results = reader.readtext(
    image,
    decoder='beamsearch',   # More accurate decoding
    beamWidth=5,            # Wider beam search
    batch_size=1,           # Process one region at a time
    text_threshold=0.5,     # Catch more potential text
    min_size=10,            # Detect smaller text
    mag_ratio=1.5,          # Enlarge image for better detection
    rotation_info=[0, 90, 180, 270]  # Check all orientations
)
```

### Memory Usage Optimization

For systems with limited GPU memory:

```python
# Memory-efficient configuration
reader = easyocr.Reader(['en'], gpu=True, quantize=True)

# Process image in smaller chunks
def process_large_image(image, chunk_size=1000):
    height, width = image.shape[:2]
    all_results = []
    
    # Process image in chunks
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            # Extract chunk
            x_end = min(x + chunk_size, width)
            y_end = min(y + chunk_size, height)
            chunk = image[y:y_end, x:x_end]
            
            # Process chunk
            chunk_results = reader.readtext(chunk)
            
            # Adjust coordinates to original image
            for i in range(len(chunk_results)):
                bbox, text, prob = chunk_results[i]
                adjusted_bbox = [
                    [p[0] + x, p[1] + y] for p in bbox
                ]
                all_results.append((adjusted_bbox, text, prob))
    
    return all_results
```

---

## Integration in Vision Pipelines

### Complete Pipeline for Glasses Project

```python
import cv2
import easyocr
import numpy as np
import pyttsx3
import time
from threading import Thread

class ReadingGlasses:
    def __init__(self):
        # Initialize OCR
        print("Loading OCR models...")
        self.reader = easyocr.Reader(['en'], gpu=True, quantize=True)
        
        # Initialize text-to-speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)  # Use 0 for webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Processing flags and data
        self.last_text = ""
        self.processing = False
        self.should_exit = False
        
        # Start processing thread
        self.process_thread = Thread(target=self.process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def detect_text_regions(self, frame):
        """
        Use a simple algorithm to detect potential text regions.
        This can be replaced with a more sophisticated object detector.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if w > 30 and h > 10 and w < frame.shape[1] * 0.9 and h < frame.shape[0] * 0.9:
                # Add margin
                x_start = max(0, x - 10)
                y_start = max(0, y - 10)
                x_end = min(frame.shape[1], x + w + 10)
                y_end = min(frame.shape[0], y + h + 10)
                
                regions.append((x_start, y_start, x_end, y_end))
        
        return regions
    
    def process_frame(self, frame):
        """Process a single frame to extract and read text."""
        # Convert from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Option 1: Process the entire frame
        results = self.reader.readtext(
            rgb_frame,
            text_threshold=0.7,
            paragraph=True,
            min_size=15,
            batch_size=4
        )
        
        # Option 2: First detect text regions, then process each
        # regions = self.detect_text_regions(frame)
        # results = []
        # for region in regions:
        #     x1, y1, x2, y2 = region
        #     cropped = rgb_frame[y1:y2, x1:x2]
        #     if cropped.size == 0:
        #         continue
        #     region_results = self.reader.readtext(cropped)
        #     # Adjust coordinates
        #     for bbox, text, prob in region_results:
        #         adjusted_bbox = [
        #             [p[0] + x1, p[1] + y1] for p in bbox
        #         ]
        #         results.append((adjusted_bbox, text, prob))
        
        # Process results
        detected_text = []
        for (bbox, text, prob) in results:
            if prob > 0.6:  # Confidence threshold
                detected_text.append(text)
                
                # Draw box around text (for visualization)
                points = np.array(bbox, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], True, (0, 255, 0), 2)
        
        # Combine all detected text
        full_text = " ".join(detected_text)
        
        return full_text, frame
    
    def speak_text(self, text):
        """Convert text to speech if it's new."""
        if text and text != self.last_text and len(text) > 3:
            self.last_text = text
            print(f"Reading: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
    
    def process_frames(self):
        """Background thread to process frames."""
        while not self.should_exit:
            if self.processing:
                # Skip if we're still processing the last frame
                time.sleep(0.1)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            self.processing = True
            
            # Process the frame
            try:
                start_time = time.time()
                text, processed_frame = self.process_frame(frame)
                elapsed = time.time() - start_time
                
                # Draw processing time
                cv2.putText(
                    processed_frame,
                    f"Process time: {elapsed:.2f}s", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), 
                    2
                )
                
                # Update display frame
                self.display_frame = processed_frame
                
                # Speak the text in a non-blocking way
                Thread(target=self.speak_text, args=(text,)).start()
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
            
            self.processing = False
    
    def run(self):
        """Main loop to display video feed."""
        self.display_frame = None
        
        while True:
            if self.display_frame is not None:
                cv2.imshow("Reading Glasses", self.display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
        
        # Cleanup
        self.should_exit = True
        self.cap.release()
        cv2.destroyAllWindows()

# Run the application
if __name__ == "__main__":
    app = ReadingGlasses()
    app.run()
```

### Integration with Other Object Detection Models

```python
# Example with YOLO for more accurate text region detection
def detect_text_regions_yolo(frame, yolo_model):
    # Preprocess frame for YOLO
    blob = cv2.dnn.blobFromImage(
        frame, 1/255.0, (416, 416), swapRB=True, crop=False
    )
    
    # Run YOLO detection
    yolo_model.setInput(blob)
    output_layers = yolo_model.forward(yolo_model.getUnconnectedOutLayersNames())
    
    # Process YOLO output
    regions = []
    for output in output_layers:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter for text class (assuming class 0 is 'text')
            if class_id == 0 and confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                
                # Calculate coordinates
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                
                regions.append((
                    max(0, x),
                    max(0, y),
                    min(frame.shape[1], x + width),
                    min(frame.shape[0], y + height)
                ))
    
    return regions
```

---

## Real-World Examples

### Example 1: Reading Product Labels

```python
def read_product_label(image_path):
    # Initialize reader with English
    reader = easyocr.Reader(['en'], gpu=True)
    
    # Read text with focus on product information
    results = reader.readtext(
        image_path,
        paragraph=False,  # We want individual elements
        min_size=10,      # Product details can be small
        text_threshold=0.7
    )
    
    # Categorize detected text
    product_info = {
        'name': '',
        'ingredients': [],
        'price': '',
        'weight': '',
        'nutrition': []
    }
    
    for _, text, _ in results:
        text = text.strip()
        
        # Basic classification based on content patterns
        if text.startswith('$') or text.endswith('$') or '€' in text:
            product_info['price'] = text
        elif 'g' in text and any(c.isdigit() for c in text):
            product_info['weight'] = text
        elif any(word in text.lower() for word in ['ingredients', 'contains']):
            product_info['ingredients'].append(text)
        elif any(word in text.lower() for word in ['calories', 'protein', 'fat', 'carb']):
            product_info['nutrition'].append(text)
        elif len(text) > 5 and len(product_info['name']) == 0:
            # Assume longer text at the top is the product name
            product_info['name'] = text
    
    return product_info
```

### Example 2: Document Scanner

```python
def scan_document(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get a binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours to detect document edges
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (assumed to be the document)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get approximate polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If we have a quadrilateral, assume it's the document
    if len(approx) == 4:
        # Sort points in order: top-left, top-right, bottom-right, bottom-left
        pts = np.array([pt[0] for pt in approx])
        rect = np.zeros((4, 2), dtype="float32")
        
        # Get top-left and bottom-right points
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        
        # Get top-right and bottom-left points
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        
        # Compute width and height of the destination image
        width_a = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        width_b = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
        height_b = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
        height = max(int(height_a), int(height_b))
        
        # Create destination points
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")
        
        # Compute perspective transform matrix
        transform_matrix = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply perspective transformation
        warped = cv2.warpPerspective(image, transform_matrix, (width, height))
        
        # Convert to RGB for EasyOCR
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        
        # Initialize reader
        reader = easyocr.Reader(['en'])
        
        # Read text with paragraph grouping
        results = reader.readtext(warped_rgb, paragraph=True)
        
        # Extract text
        document_text = []
        for _, text, _ in results:
            document_text.append(text)
        
        return "\n".join(document_text)
    else:
        return "Document edges not clearly detected"
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Low Detection Accuracy

**Symptoms**:
- Missing text
- Incorrect character recognition
- Text combined incorrectly

**Solutions**:
```python
# Improve image quality first
def enhance_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply morphological operations to remove noise
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Return the enhanced image
    return opening

# Then use enhanced image for OCR
enhanced = enhance_image(image)
results = reader.readtext(
    enhanced,
    text_threshold=0.5,  # Lower threshold
    min_size=10,         # Catch smaller text
    mag_ratio=1.5        # Magnify the image
)
```

#### 2. Slow Processing Speed

**Symptoms**:
- Long processing times
- Application lag
- High resource usage

**Solutions**:
```python
# Resize large images
def resize_for_ocr(image, max