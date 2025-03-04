## A Comprehensive Hands-on Bootcamp

---

## Module 1: Getting Started with EasyOCR
### Learning Objectives:
- Understand what OCR is and how EasyOCR works
- Install and configure EasyOCR for your projects
- Create your first OCR application to extract text from images

### Introduction to EasyOCR

EasyOCR is a Python library that implements Optical Character Recognition (OCR) using deep learning models. It allows developers to extract text from images with minimal effort and supports over 80 languages.

**Why EasyOCR?**
- Simple and consistent API
- Multi-language support (80+ languages)
- GPU acceleration support
- Built on PyTorch
- Active development and community
- No commercial restrictions

### Installation and Setup

```bash
# Basic installation
pip install easyocr

# For GPU support (CUDA)
pip install easyocr torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> **Note for Students:** If you're working on Windows, you might need to install Visual C++ Build Tools. On macOS, ensure you have Xcode Command Line Tools installed.

### Your First OCR Application

```python
import easyocr

# Initialize the reader with languages you need
reader = easyocr.Reader(['en', 'fr'])  # English and French

# Read text from an image
results = reader.readtext('image.jpg')

# Process results
for (bbox, text, prob) in results:
    print(f"Text: {text}, Confidence: {prob:.2f}")
```

**Understanding the Output:**
- `bbox`: Bounding box coordinates (top-left, top-right, bottom-right, bottom-left)
- `text`: Extracted text content
- `prob`: Confidence score (0-1)

### Exercise 1:
1. Install EasyOCR on your system
2. Download a sample image with text from the internet
3. Use the code above to extract and print the text
4. Experiment with different languages

---

## Module 2: Advanced EasyOCR Configuration
### Learning Objectives:
- Configure EasyOCR for optimal performance
- Understand and adjust parameters to improve results
- Handle different types of text recognition scenarios

### Advanced Configuration Options

```python
# Configure OCR with additional parameters
reader = easyocr.Reader(
    ['en'],
    gpu=True,                   # Use GPU acceleration
    model_storage_directory='models',  # Custom model directory
    download_enabled=True,      # Download models if needed
    detector=True,              # Text detection
    recognizer=True,            # Text recognition
    quantize=True,              # Model quantization for speed
    verbose=False               # Silence output
)

# Advanced readtext parameters
results = reader.readtext(
    'image.jpg',
    detail=1,                   # 0: text only, 1: with position and confidence
    paragraph=False,            # Group text into paragraphs
    min_size=10,                # Minimum text box size
    slope_ths=0.2,              # Slope threshold for merging text boxes
    ycenter_ths=0.5,            # Vertical center threshold for merging
    height_ths=0.5,             # Height threshold for merging
    width_ths=0.5,              # Width threshold for merging
    decoder='greedy',           # Decoding method ('greedy', 'beamsearch', or 'wordbeamsearch')
    beamWidth=5,                # Beam width for beam search decoder
    batch_size=1,               # Batch size for processing
    workers=0,                  # Number of worker processes
    allowlist='0123456789',     # Only allow these characters
    blocklist='!@#$%',          # Block these characters
    rotation_info=[90, 180, 270] # Try these rotations
)
```

### Key Parameters Explained:

| Parameter | Purpose | When to Adjust |
|-----------|---------|----------------|
| `gpu` | Enables GPU acceleration | When processing large images or batches |
| `detail` | Controls output format | Use 0 for just text, 1 for full details |
| `paragraph` | Groups text into paragraphs | For documents with structured content |
| `decoder` | Text recognition method | 'wordbeamsearch' for greater accuracy, 'greedy' for speed |
| `rotation_info` | Try different rotations | For images where text orientation varies |

### Exercise 2:
1. Take an image with difficult-to-read text
2. Use the default configuration and record the results
3. Experiment with different parameter combinations
4. Identify which parameters improve results the most

---

## Module 3: Image Preprocessing for OCR
### Learning Objectives:
- Apply image preprocessing techniques to improve OCR accuracy
- Understand how image quality affects text recognition
- Create a preprocessing pipeline for different image types

### Preprocessing Techniques

```python
import cv2
import numpy as np

# Load image
image = cv2.imread('document.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Noise removal
denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

# Deskew image (straighten text)
coords = np.column_stack(np.where(denoised > 0))
angle = cv2.minAreaRect(coords)[-1]
if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle
(h, w) = denoised.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
deskewed = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Save preprocessed image
cv2.imwrite('preprocessed.jpg', deskewed)

# Now perform OCR on the preprocessed image
results = reader.readtext('preprocessed.jpg')
```

### Common Preprocessing Steps Explained:

1. **Grayscale Conversion**: Reduces complexity and improves processing speed
2. **Thresholding**: Converts grayscale to binary image, separating text from background
3. **Noise Removal**: Removes speckles and artifacts that can confuse OCR
4. **Deskewing**: Straightens text for better recognition

### Exercise 3:
1. Find or create images with various issues (skewed text, noisy background, poor lighting)
2. Create a preprocessing function that applies appropriate techniques
3. Compare OCR results before and after preprocessing
4. Identify which preprocessing steps help most for different image problems

---

## Module 4: Post-processing OCR Results
### Learning Objectives:
- Implement post-processing techniques to improve OCR accuracy
- Use regular expressions to validate and correct extracted text
- Apply contextual corrections based on expected content

### Post-processing Techniques

```python
import re
from difflib import get_close_matches

# Dictionary for spellchecking
dictionary = ['apple', 'banana', 'orange', 'pineapple']

def post_process(results):
    processed_results = []
    
    for (bbox, text, prob) in results:
        # Convert to lowercase for processing
        text_lower = text.lower()
        
        # Remove non-alphanumeric characters
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Correct spacing
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Spellcheck if confidence is low
        if prob < 0.8:
            words = cleaned_text.split()
            corrected_words = []
            
            for word in words:
                matches = get_close_matches(word, dictionary, n=1, cutoff=0.8)
                if matches:
                    corrected_words.append(matches[0])
                else:
                    corrected_words.append(word)
            
            cleaned_text = ' '.join(corrected_words)
        
        processed_results.append((bbox, cleaned_text, prob))
    
    return processed_results

# Apply post-processing
final_results = post_process(results)
```

### Common Post-processing Techniques:

1. **Character Filtering**: Remove or replace unwanted characters
2. **Spell Checking**: Correct misspelled words using dictionaries
3. **Pattern Matching**: Use regular expressions to identify and correct patterns
4. **Confidence Filtering**: Discard or flag results with low confidence scores

### Exercise 4:
1. Take OCR results from a previous exercise
2. Identify common error patterns
3. Create a post-processing function to correct these errors
4. Test your function on new OCR results

---

## Module 5: Performance Optimization
### Learning Objectives:
- Optimize EasyOCR for processing speed and accuracy
- Implement batch processing for multiple images
- Extract text only from regions of interest (ROI)

### Performance Optimization Techniques

```python
# Use batch processing for multiple images
image_list = ['img1.jpg', 'img2.jpg', 'img3.jpg']
batch_results = reader.readtext_batched(
    image_list,
    batch_size=4,
    detail=1,
    paragraph=False
)

# Limit recognition to specific regions (ROI)
def extract_roi(image_path, roi_coordinates):
    image = cv2.imread(image_path)
    x, y, w, h = roi_coordinates
    roi = image[y:y+h, x:x+w]
    return roi

# Process only regions of interest
roi_coordinates = [100, 100, 400, 200]  # [x, y, width, height]
roi = extract_roi('document.jpg', roi_coordinates)
cv2.imwrite('roi.jpg', roi)
roi_results = reader.readtext('roi.jpg')
```

### Optimization Strategies:

1. **Batch Processing**: Process multiple images at once for efficiency
2. **Region of Interest (ROI)**: Only process relevant parts of images
3. **Model Quantization**: Use quantized models for faster inference
4. **GPU Acceleration**: Enable GPU for processing speed
5. **Parallel Processing**: Use multiple workers for concurrent processing

### Exercise 5:
1. Create a dataset of 10+ images with text
2. Implement batch processing to extract text from all images
3. Time the execution and compare with sequential processing
4. Identify a specific ROI in one image and extract only that text

---

## Module 6: Integrating with Other Libraries
### Learning Objectives:
- Combine EasyOCR with other popular libraries
- Create a web API for OCR services
- Compare results with alternative OCR tools

### Integration Examples

```python
# Integration with Tesseract OCR for comparison
import pytesseract
from PIL import Image

def compare_ocr_results(image_path):
    # EasyOCR
    reader = easyocr.Reader(['en'])
    easyocr_results = reader.readtext(image_path, detail=0)
    easyocr_text = ' '.join(easyocr_results)
    
    # Tesseract OCR
    tesseract_text = pytesseract.image_to_string(Image.open(image_path))
    
    return {
        'easyocr': easyocr_text,
        'tesseract': tesseract_text
    }

# Web API integration with Flask
from flask import Flask, request, jsonify
import base64
import io
from PIL import Image

app = Flask(__name__)
reader = easyocr.Reader(['en'])

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    # Get image from request
    image_data = base64.b64decode(request.json['image'])
    image = Image.open(io.BytesIO(image_data))
    image.save('temp.jpg')
    
    # Perform OCR
    results = reader.readtext('temp.jpg')
    
    # Format response
    response = []
    for (bbox, text, prob) in results:
        response.append({
            'text': text,
            'confidence': float(prob),
            'bbox': bbox
        })
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
```

### Common Integration Scenarios:

1. **Web APIs**: Create REST endpoints for OCR services
2. **Database Storage**: Store and query OCR results
3. **Document Processing Pipelines**: Combine with PDF processing tools
4. **Multi-tool Validation**: Compare results across different OCR engines

### Exercise 6:
1. Create a simple Flask application with an OCR endpoint
2. Add a web form that allows users to upload images
3. Return both the processed image and extracted text
4. Optional: Implement result caching for repeated requests

---

## Capstone Project 1: Document Text Extractor
### Project Requirements:
- Process different document types (PDF, images)
- Extract text with formatting preserved
- Export to structured formats (TXT, JSON)
- Implement a simple GUI

### Implementation Details

```python
# document_reader.py
import easyocr
import os
import json
from pdf2image import convert_from_path
import cv2
import numpy as np

class DocumentReader:
    def __init__(self, languages=['en']):
        self.reader = easyocr.Reader(languages)
        
    def preprocess_image(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Binarization with Otsu's method
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        return denoised
    
    def extract_from_image(self, image_path):
        # Read the image
        image = cv2.imread(image_path)
        
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Save processed image temporarily
        temp_path = "temp_processed.jpg"
        cv2.imwrite(temp_path, processed_image)
        
        # Perform OCR
        results = self.reader.readtext(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return results
    
    def extract_from_pdf(self, pdf_path, pages='all'):
        # Convert PDF to images
        if pages == 'all':
            images = convert_from_path(pdf_path)
        else:
            images = convert_from_path(pdf_path, first_page=pages[0], last_page=pages[-1])
        
        all_results = []
        
        # Process each page
        for i, image in enumerate(images):
            # Convert PIL Image to OpenCV format
            open_cv_image = np.array(image)
            open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR
            
            # Preprocess
            processed_image = self.preprocess_image(open_cv_image)
            
            # Save processed image temporarily
            temp_path = f"temp_page_{i}.jpg"
            cv2.imwrite(temp_path, processed_image)
            
            # Perform OCR
            results = self.reader.readtext(temp_path)
            
            # Add page information
            page_results = {
                'page': i + 1,
                'content': results
            }
            
            all_results.append(page_results)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return all_results
    
    def save_to_txt(self, results, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            if isinstance(results[0], dict):  # PDF results with pages
                for page in results:
                    f.write(f"===== PAGE {page['page']} =====\n\n")
                    for (_, text, _) in page['content']:
                        f.write(f"{text}\n")
                    f.write("\n\n")
            else:  # Image results
                for (_, text, _) in results:
                    f.write(f"{text}\n")
    
    def save_to_json(self, results, output_path):
        if isinstance(results[0], dict):  # PDF results with pages
            output = []
            for page in results:
                page_obj = {
                    'page': page['page'],
                    'content': []
                }
                for (bbox, text, prob) in page['content']:
                    page_obj['content'].append({
                        'text': text,
                        'confidence': float(prob),
                        'bbox': [[float(x), float(y)] for x, y in bbox]
                    })
                output.append(page_obj)
        else:  # Image results
            output = []
            for (bbox, text, prob) in results:
                output.append({
                    'text': text,
                    'confidence': float(prob),
                    'bbox': [[float(x), float(y)] for x, y in bbox]
                })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
```

### GUI Implementation

```python
# document_reader_gui.py
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
from document_reader import DocumentReader

class DocumentReaderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Text Extractor")
        self.root.geometry("600x400")
        
        self.document_reader = DocumentReader(['en'])
        
        self.create_widgets()
    
    def create_widgets(self):
        # Frame for input options
        input_frame = ttk.LabelFrame(self.root, text="Input Options")
        input_frame.pack(fill="x", padx=10, pady=10)
        
        # File selection
        ttk.Label(input_frame, text="Select Document:").grid(row=0, column=0, padx=5, pady=5)
        self.file_path_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.file_path_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse...", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)
        
        # Language selection
        ttk.Label(input_frame, text="Language:").grid(row=1, column=0, padx=5, pady=5)
        self.language_var = tk.StringVar(value="en")
        languages = ["en", "fr", "es", "de", "zh", "ja"]
        ttk.Combobox(input_frame, textvariable=self.language_var, values=languages).grid(row=1, column=1, padx=5, pady=5)
        
        # Output options
        output_frame = ttk.LabelFrame(self.root, text="Output Options")
        output_frame.pack(fill="x", padx=10, pady=10)
        
        # Output format
        ttk.Label(output_frame, text="Output Format:").grid(row=0, column=0, padx=5, pady=5)
        self.output_format_var = tk.StringVar(value="txt")
        formats = ["txt", "json"]
        ttk.Combobox(output_frame, textvariable=self.output_format_var, values=formats).grid(row=0, column=1, padx=5, pady=5)
        
        # Output path
        ttk.Label(output_frame, text="Output Path:").grid(row=1, column=0, padx=5, pady=5)
        self.output_path_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_path_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(output_frame, text="Browse...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)
        
        # Status frame
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=10, pady=10)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left")
        
        self.progress = ttk.Progressbar(status_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(side="right")
        
        # Action buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Process Document", command=self.process_document).pack(side="left", padx=10)
        ttk.Button(button_frame, text="Exit", command=self.root.quit).pack(side="left", padx=10)
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Documents", "*.pdf;*.png;*.jpg;*.jpeg;*.tiff")]
        )
        if file_path:
            self.file_path_var.set(file_path)
            
            # Suggest an output path
            dir_name = os.path.dirname(file_path)
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            output_format = self.output_format_var.get()
            suggested_output = os.path.join(dir_name, f"{file_name}.{output_format}")
            self.output_path_var.set(suggested_output)
    
    def browse_output(self):
        output_format = self.output_format_var.get()
        file_path = filedialog.asksaveasfilename(
            defaultextension=f".{output_format}",
            filetypes=[(f"{output_format.upper()} files", f"*.{output_format}")]
        )
        if file_path:
            self.output_path_var.set(file_path)
    
    def process_document(self):
        file_path = self.file_path_var.get()
        output_path = self.output_path_var.get()
        language = self.language_var.get()
        output_format = self.output_format_var.get()
        
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid document file.")
            return
        
        if not output_path:
            messagebox.showerror("Error", "Please specify an output path.")
            return
        
        # Update document reader with selected language
        self.document_reader = DocumentReader([language])
        
        self.status_var.set("Processing document...")
        self.progress["value"] = 20
        self.root.update_idletasks()
        
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.pdf':
                results = self.document_reader.extract_from_pdf(file_path)
            else:  # Image file
                results = self.document_reader.extract_from_image(file_path)
            
            self.progress["value"] = 80
            self.root.update_idletasks()
            
            # Save results
            if output_format == 'txt':
                self.document_reader.save_to_txt(results, output_path)
            else:  # json
                self.document_reader.save_to_json(results, output_path)
            
            self.progress["value"] = 100
            self.status_var.set("Document processed successfully!")
            messagebox.showinfo("Success", f"Document processed and saved to {output_path}")
            
        except Exception as e:
            self.status_var.set("Error processing document")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        
        self.progress["value"] = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentReaderGUI(root)
    root.mainloop()
```

### Project Testing

```python
# test_document_reader.py
import unittest
import os
import tempfile
import json
from document_reader import DocumentReader

class TestDocumentReader(unittest.TestCase):
    def setUp(self):
        self.reader = DocumentReader(['en'])
        self.test_image = "test_document.jpg"  # Create a test image for testing
        self.test_pdf = "test_document.pdf"    # Create a test PDF for testing
    
    def test_image_extraction(self):
        if not os.path.exists(self.test_image):
            self.skipTest("Test image not found")
        
        results = self.reader.extract_from_image(self.test_image)
        self.assertIsNotNone(results)
        self.assertTrue(len(results) > 0)
    
    def test_pdf_extraction(self):
        if not os.path.exists(self.test_pdf):
            self.skipTest("Test PDF not found")
        
        results = self.reader.extract_from_pdf(self.test_pdf)
        self.assertIsNotNone(results)
        self.assertTrue(len(results) > 0)
    
    def test_save_to_txt(self):
        if not os.path.exists(self.test_image):
            self.skipTest("Test image not found")
        
        results = self.reader.extract_from_image(self.test_image)
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            self.reader.save_to_txt(results, output_path)
            self.assertTrue(os.path.exists(output_path))
            
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertTrue(len(content) > 0)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
    
    def test_save_to_json(self):
        if not os.path.exists(self.test_image):
            self.skipTest("Test image not found")
        
        results = self.reader.extract_from_image(self.test_image)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            self.reader.save_to_json(results, output_path)
            self.assertTrue(os.path.exists(output_path))
            
            with open(output_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                self.assertTrue(len(content) > 0)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

if __name__ == '__main__':
    unittest.main()
```

### Project Challenges:
1. Add support for more document formats (DOCX, RTF)
2. Implement table detection and structured data extraction
3. Create a configuration panel for preprocessing options
4. Add a preview feature to show processed images

---

## Capstone Project 2: Receipt Parser
### Project Requirements:
- Extract structured information from receipts (date, merchant, items, total)
- Support multiple languages
- Use regular expressions and heuristics for field extraction
- Implement a REST API interface

### Receipt Parser Implementation

```python
# receipt_parser.py
import easyocr
import cv2
import numpy as np
import re
import dateparser
from typing import Dict, List, Any, Tuple, Optional
import os
import json
import pandas as pd

class ReceiptParser:
    def __init__(self, languages=['en']):
        self.reader = easyocr.Reader(languages)
        self.languages = languages
        
        # Regular expressions for different fields
        self.patterns = {
            'date': [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # 01/01/2023, 1-1-23
                r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',  # 1 January 2023
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}'  # January 1, 2023
            ],
            'total': [
                r'(?:total|sum|amount)(?:\s+\w+){0,2}\s*[\$€£]?\s*(\d+[.,]\d{2})',  # Total: $100.00
                r'[\$€£]?\s*(\d+[.,]\d{2})(?:\s+\w+){0,2}\s*(?:total|sum|amount)',  # $100.00 Total
                r'(?:total|sum|amount)(?:\s+\w+){0,2}\s*[\$€£]?\s*(\d+)',  # Total: $100
                r'[\$€£]?\s*(\d+)(?:\s+\w+){0,2}\s*(?:total|sum|amount)'   # $100 Total
            ],
            'merchant': [
                r'^([A-Z][A-Za-z\s]{2,30})$',  # Capitalized name at start of line
                r'((?:[A-Z][A-Za-z]+\s*){1,4})\s*(?:Ltd|LLC|Inc|GmbH|Co\.|Company)'  # Company name with suffix
            ],
            'tax': [
                r'(?:tax|vat|gst)(?:\s+\w+){0,2}\s*[\$€£]?\s*(\d+[.,]\d{2})',  # Tax: $10.00
                r'[\$€£]?\s*(\d+[.,]\d{2})(?:\s+\w+){0,2}\s*(?:tax|vat|gst)'   # $10.00 Tax
            ],
            'payment_method': [
                r'(?:paid|payment|method)(?:\s+\w+){0,2}\s*(?:by|via|with)?\s*(\w+\s*card|\w+\s*cash|credit|debit|visa|mastercard|amex|cash|check)',
                r'(?:visa|mastercard|amex|cash|credit\s*card|debit\s*card|check)(?:\s+\w+){0,2}\s*(?:payment|paid)'
            ]
        }
    
    def preprocess_image(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Denoise image
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Deskew image (straighten text)
        try:
            coords = np.column_stack(np.where(denoised > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = denoised.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            deskewed = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return deskewed
        except:
            # If deskewing fails, return denoised image
            return denoised
    
    def parse_receipt(self, image_path) -> Dict[str, Any]:
        """Parse receipt from image file."""
        # Read and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        processed_image = self.preprocess_image(image)
        
        # Save processed image temporarily
        temp_path = "_temp_receipt.jpg"
        cv2.imwrite(temp_path, processed_image)
        
        # Perform OCR
        results = self.reader.readtext(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Extract full text for analysis
        text_blocks = [text for _, text, _ in results]
        full_text = '\n'.join(text_blocks)
        
        # Parse structured data
        structured_data = self._extract_structured_data(full_text, text_blocks)
        
        # Extract line items (products, quantities, prices)
        line_items = self._extract_line_items(text_blocks)
        structured_data['items'] = line_items
        
        return structured_data
    
    def _extract_structured_data(self, full_text, text_blocks) -> Dict[str, Any]:
        """Extract structured data using regex patterns."""
        structured_data = {
            'date': None,
            'merchant': None,
            'total': None,
            'tax': None,
            'payment_method': None,
            'currency': None
        }
        
        # Currency detection
        currency_symbols = {
            '$': 'USD',
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            '₹': 'INR',
            'USD': 'USD',
            'EUR': 'EUR',
            'GBP': 'GBP'
        }
        
        for symbol, currency in currency_symbols.items():
            if symbol in full_text:
                structured_data['currency'] = currency
                break
        
        # Date extraction
        for pattern in self.patterns['date']:
            date_match = re.search(pattern, full_text, re.IGNORECASE)
            if date_match:
                date_str = date_match.group(0)
                try:
                    parsed_date = dateparser.parse(date_str, languages=self.languages)
                    if parsed_date:
                        structured_data['date'] = parsed_date.strftime('%Y-%m-%d')
                        break
                except:
                    continue
        
        # Total extraction
        for pattern in self.patterns['total']:
            total_match = re.search(pattern, full_text, re.IGNORECASE)
            if total_match:
                total_str = total_match.group(1)
                try:
                    # Replace comma with dot for decimal
                    total_str = total_str.replace(',', '.')
                    structured_data['total'] = float(total_str)
                    break
                except:
                    continue
        
        # If total wasn't found, try looking for a number after "total"
        if structured_data['total'] is None:
            for block in text_blocks:
                if re.search(r'total|sum|amount', block, re.IGNORECASE):
                    numbers = re.findall(r'[\$€£]?\s*(\d+[.,]\d{2})', block)
                    if numbers:
                        try:
                            total_str = numbers[-1].replace(',', '.')
                            structured_data['total'] = float(total_str)
                            break
                        except:
                            continue
        
        # Merchant name extraction
        # First, try the first few lines as they often contain the merchant name
        potential_merchants = []
        for i, block in enumerate(text_blocks[:5]):
            if len(block.strip()) > 3 and not re.search(r'receipt|invoice|order', block, re.IGNORECASE):
                potential_merchants.append(block.strip())
        
        if potential_merchants:
            structured_data['merchant'] = potential_merchants[0]
        
        # Otherwise use regex patterns
        if not structured_data['merchant']:
            for pattern in self.patterns['merchant']:
                for block in text_blocks:
                    merchant_match = re.search(pattern, block, re.IGNORECASE)
                    if merchant_match:
                        structured_data['merchant'] = merchant_match.group(1).strip()
                        break
                if structured_data['merchant']:
                    break
        
        # Tax extraction
        for pattern in self.patterns['tax']:
            tax_match = re.search(pattern, full_text, re.IGNORECASE)
            if tax_match:
                tax_str = tax_match.group(1)
                try:
                    # Replace comma with dot for
<think>The content appears to be cut off in the middle of the Receipt Parser implementation. I need to continue from that point and complete the tutorial in a bootcamp-style format.

Looking at what we have so far, the document is structured as follows:
1. An introduction to EasyOCR
2. Various modules covering different aspects of EasyOCR
3. Capstone projects to apply the knowledge

I should continue the receipt parser implementation and then add the REST API interface that was mentioned but not completed. I'll maintain the bootcamp-style formatting, with clear explanations, code samples, and learning goals.

The section was cut off in the middle of the `_extract_structured_data` method, specifically in the tax extraction part. I'll continue from there and complete the implementation.</think># EasyOCR: Optical Character Recognition with Python (Continued)

## Capstone Project 2: Receipt Parser (Continued)

```python
                    # Replace comma with dot for decimal
                    tax_str = tax_str.replace(',', '.')
                    structured_data['tax'] = float(tax_str)
                    break
                except:
                    continue
        
        # Payment method extraction
        for pattern in self.patterns['payment_method']:
            payment_match = re.search(pattern, full_text, re.IGNORECASE)
            if payment_match:
                structured_data['payment_method'] = payment_match.group(1).strip()
                break
        
        return structured_data
    
    def _extract_line_items(self, text_blocks) -> List[Dict[str, Any]]:
        """Extract line items (products, quantities, prices)."""
        line_items = []
        
        # Look for patterns like:
        # 1) Product name followed by price (Item $10.00)
        # 2) Product name, quantity, price (Item 2 $20.00)
        # 3) Tabular format (multiple spaces/tabs between name and price)
        
        for block in text_blocks:
            # Skip blocks that are likely headers or totals
            if re.search(r'total|subtotal|tax|date|receipt|invoice|merchant|payment', block, re.IGNORECASE):
                continue
            
            # Pattern 1: Product name followed by price
            match = re.search(r'(.+)\s+[\$€£]?(\d+[.,]\d{2})$', block)
            if match:
                name = match.group(1).strip()
                price_str = match.group(2).replace(',', '.')
                try:
                    price = float(price_str)
                    line_items.append({
                        'item': name,
                        'quantity': 1,
                        'price': price
                    })
                    continue
                except:
                    pass
            
            # Pattern 2: Product name, quantity, price
            match = re.search(r'(.+)\s+(\d+)\s+[\$€£]?(\d+[.,]\d{2})$', block)
            if match:
                name = match.group(1).strip()
                quantity = int(match.group(2))
                price_str = match.group(3).replace(',', '.')
                try:
                    price = float(price_str)
                    line_items.append({
                        'item': name,
                        'quantity': quantity,
                        'price': price
                    })
                    continue
                except:
                    pass
            
            # Pattern 3: Tabular format
            match = re.search(r'(.+?)\s{2,}[\$€£]?(\d+[.,]\d{2})$', block)
            if match:
                name = match.group(1).strip()
                price_str = match.group(2).replace(',', '.')
                try:
                    price = float(price_str)
                    line_items.append({
                        'item': name,
                        'quantity': 1,
                        'price': price
                    })
                except:
                    pass
        
        return line_items
    
    def save_to_json(self, parsed_data, output_path):
        """Save parsed receipt data as JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2, default=str)
    
    def save_to_csv(self, parsed_data, output_path):
        """Save parsed receipt data as CSV."""
        # Flatten the items into separate rows
        rows = []
        
        base_data = {key: val for key, val in parsed_data.items() if key != 'items'}
        
        if 'items' in parsed_data and parsed_data['items']:
            for item in parsed_data['items']:
                row = base_data.copy()
                row.update(item)
                rows.append(row)
        else:
            rows.append(base_data)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
```

### REST API Implementation

```python
# receipt_api.py
from flask import Flask, request, jsonify, send_file
import os
import tempfile
import uuid
from werkzeug.utils import secure_filename
from receipt_parser import ReceiptParser
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize receipt parser with English language
receipt_parser = ReceiptParser(['en'])

@app.route('/parse_receipt', methods=['POST'])
def parse_receipt():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        return jsonify({'error': 'Unsupported file format'}), 400
    
    # Get language preference
    language = request.form.get('language', 'en')
    languages = language.split(',')
    
    # Create parser with specified languages
    parser = ReceiptParser(languages)
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    try:
        # Parse the receipt
        parsed_data = parser.parse_receipt(filepath)
        
        # Determine output format
        output_format = request.form.get('format', 'json')
        
        if output_format == 'json':
            # Return JSON directly
            return jsonify(parsed_data)
        elif output_format == 'csv':
            # Save as CSV and return the file
            temp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
            temp_csv.close()
            parser.save_to_csv(parsed_data, temp_csv.name)
            
            return send_file(
                temp_csv.name,
                as_attachment=True,
                download_name=f"receipt_{unique_filename.split('_')[0]}.csv",
                mimetype='text/csv'
            )
        else:
            return jsonify({'error': 'Unsupported output format'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'service': 'receipt-parser-api'})

@app.route('/', methods=['GET'])
def home():
    return """
    <html>
        <head>
            <title>Receipt Parser API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                form { background: #f5f5f5; padding: 20px; border-radius: 5px; }
                label { display: block; margin: 10px 0 5px; }
                input, select { padding: 8px; width: 100%; }
                button { background: #4CAF50; color: white; padding: 10px 15px; border: none; border-radius: 4px; margin-top: 20px; cursor: pointer; }
                button:hover { background: #45a049; }
                pre { background: #f0f0f0; padding: 15px; overflow: auto; }
            </style>
        </head>
        <body>
            <h1>Receipt Parser API</h1>
            <p>Upload a receipt image to extract information.</p>
            
            <form action="/parse_receipt" method="post" enctype="multipart/form-data">
                <label for="file">Receipt Image:</label>
                <input type="file" id="file" name="file" accept="image/*" required>
                
                <label for="language">Language:</label>
                <select id="language" name="language">
                    <option value="en">English</option>
                    <option value="fr">French</option>
                    <option value="es">Spanish</option>
                    <option value="de">German</option>
                    <option value="zh">Chinese</option>
                </select>
                
                <label for="format">Output Format:</label>
                <select id="format" name="format">
                    <option value="json">JSON</option>
                    <option value="csv">CSV</option>
                </select>
                
                <button type="submit">Parse Receipt</button>
            </form>
            
            <h2>API Documentation</h2>
            <pre>
POST /parse_receipt
Content-Type: multipart/form-data

Parameters:
- file: Receipt image file (required)
- language: Language code(s) comma-separated (default: en)
- format: Output format (json or csv, default: json)

Response (JSON format):
{
    "date": "2023-01-01",
    "merchant": "GROCERY STORE",
    "total": 50.25,
    "tax": 4.50,
    "payment_method": "credit card",
    "currency": "USD",
    "items": [
        {
            "item": "Milk",
            "quantity": 1,
            "price": 3.99
        },
        {
            "item": "Bread",
            "quantity": 2,
            "price": 2.50
        }
    ]
}
            </pre>
        </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### Testing the Receipt Parser

```python
# test_receipt_parser.py
import unittest
import os
import tempfile
import json
from receipt_parser import ReceiptParser

class TestReceiptParser(unittest.TestCase):
    def setUp(self):
        self.parser = ReceiptParser(['en'])
        self.test_receipt = "test_receipt.jpg"  # Create a test receipt image
    
    def test_receipt_parsing(self):
        if not os.path.exists(self.test_receipt):
            self.skipTest("Test receipt image not found")
        
        parsed_data = self.parser.parse_receipt(self.test_receipt)
        
        # Check that we have basic structure
        self.assertIsInstance(parsed_data, dict)
        self.assertIn('items', parsed_data)
        
        # Check for essential fields
        essential_fields = ['date', 'merchant', 'total']
        for field in essential_fields:
            self.assertIn(field, parsed_data)
    
    def test_line_item_extraction(self):
        if not os.path.exists(self.test_receipt):
            self.skipTest("Test receipt image not found")
        
        parsed_data = self.parser.parse_receipt(self.test_receipt)
        
        # Check line items
        self.assertIsInstance(parsed_data['items'], list)
        
        # If receipt has items, validate their structure
        if parsed_data['items']:
            item = parsed_data['items'][0]
            self.assertIn('item', item)
            self.assertIn('price', item)
    
    def test_save_to_json(self):
        if not os.path.exists(self.test_receipt):
            self.skipTest("Test receipt image not found")
        
        parsed_data = self.parser.parse_receipt(self.test_receipt)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            self.parser.save_to_json(parsed_data, output_path)
            self.assertTrue(os.path.exists(output_path))
            
            with open(output_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                self.assertIn('items', content)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
    
    def test_save_to_csv(self):
        if not os.path.exists(self.test_receipt):
            self.skipTest("Test receipt image not found")
        
        parsed_data = self.parser.parse_receipt(self.test_receipt)
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            self.parser.save_to_csv(parsed_data, output_path)
            self.assertTrue(os.path.exists(output_path))
            
            # Check if file is not empty
            self.assertGreater(os.path.getsize(output_path), 0)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

if __name__ == '__main__':
    unittest.main()
```

### Deployment Script for Receipt Parser API

```python
# deploy_receipt_api.py
import argparse
import os
import subprocess
import sys

def setup_environment():
    """Set up the Python environment with required packages."""
    print("Setting up environment...")
    packages = [
        "easyocr",
        "flask",
        "opencv-python",
        "numpy",
        "pandas",
        "dateparser",
        "Werkzeug",
        "gunicorn"  # For production deployment
    ]
    
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
    print("Environment setup complete.")

def create_docker_file():
    """Create a Dockerfile for containerization."""
    print("Creating Dockerfile...")
    
    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY receipt_parser.py .
COPY receipt_api.py .

# Create upload directory
RUN mkdir -p uploads

# Expose port
EXPOSE 5000

# Run the API
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "receipt_api:app"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    requirements_content = """easyocr
flask
opencv-python
numpy
pandas
dateparser
Werkzeug
gunicorn
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    
    print("Dockerfile and requirements.txt created.")

def create_docker_compose():
    """Create docker-compose.yml for easy deployment."""
    print("Creating docker-compose.yml...")
    
    compose_content = """version: '3'

services:
  receipt-parser-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./uploads:/app/uploads
    restart: unless-stopped
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(compose_content)
    
    print("docker-compose.yml created.")

def main():
    parser = argparse.ArgumentParser(description="Deploy Receipt Parser API")
    parser.add_argument("--setup", action="store_true", help="Set up Python environment")
    parser.add_argument("--docker", action="store_true", help="Create Docker files")
    parser.add_argument("--run", action="store_true", help="Run the API")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_environment()
    
    if args.docker:
        create_docker_file()
        create_docker_compose()
    
    if args.run:
        print("Starting Receipt Parser API...")
        os.environ["FLASK_APP"] = "receipt_api.py"
        subprocess.call([sys.executable, "-m", "flask", "run", "--host=0.0.0.0"])

if __name__ == "__main__":
    main()
```

### Project Challenges:
1. Improve merchant name extraction using fuzzy matching
2. Add support for receipt validation (checking if totals match items)
3. Implement a reporting feature to track expenses by category
4. Create a mobile application that uses the API to scan receipts
5. Add user authentication to store receipts by user

---

## Module 7: Advanced OCR Techniques
### Learning Objectives:
- Implement OCR for complex documents and layouts
- Handle multilingual text recognition efficiently
- Extract structured data from documents with tables

### Handling Complex Document Layouts

```python
import easyocr
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

def extract_text_with_layout_analysis(image_path, languages=['en']):
    """Extract text while preserving layout structure."""
    # Initialize the reader
    reader = easyocr.Reader(languages)
    
    # Read text with bounding boxes
    results = reader.readtext(image_path, paragraph=False)
    
    # Extract coordinates for layout analysis
    text_blocks = []
    for bbox, text, conf in results:
        x_min = min(p[0] for p in bbox)
        y_min = min(p[1] for p in bbox)
        x_max = max(p[0] for p in bbox)
        y_max = max(p[1] for p in bbox)
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        text_blocks.append({
            'text': text,
            'confidence': conf,
            'bbox': [x_min, y_min, x_max, y_max],
            'center': [center_x, center_y]
        })
    
    # Extract centers for clustering
    centers = np.array([block['center'] for block in text_blocks])
    
    # Cluster by position (to identify paragraphs)
    clustering = DBSCAN(eps=50, min_samples=1).fit(centers)
    labels = clustering.labels_
    
    # Group text blocks by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(text_blocks[i])
    
    # Sort clusters by y-coordinate (top to bottom)
    sorted_clusters = []
    for label, blocks in clusters.items():
        # Calculate the average y-coordinate for the cluster
        avg_y = sum(block['center'][1] for block in blocks) / len(blocks)
        sorted_clusters.append((avg_y, label, blocks))
    
    sorted_clusters.sort()  # Sort by y-coordinate
    
    # For each cluster, sort blocks from left to right
    structured_text = []
    for _, label, blocks in sorted_clusters:
        # Sort blocks within cluster by x-coordinate (left to right)
        blocks.sort(key=lambda b: b['center'][0])
        
        # Add as paragraph
        paragraph = {
            'paragraph_id': label,
            'blocks': blocks,
            'text': ' '.join(block['text'] for block in blocks)
        }
        structured_text.append(paragraph)
    
    return structured_text

def visualize_layout_analysis(image_path, structured_text):
    """Visualize the layout analysis results."""
    image = cv2.imread(image_path)
    viz_image = image.copy()
    
    # Define colors for different paragraphs
    colors = [
        (255, 0, 0),   # Red
        (0, 255, 0),   # Green
        (0, 0, 255),   # Blue
        (255, 255, 0), # Yellow
        (255, 0, 255), # Magenta
        (0, 255, 255), # Cyan
        (128, 0, 0),   # Maroon
        (0, 128, 0)    # Green (dark)
    ]
    
    # Draw bounding boxes
    for i, paragraph in enumerate(structured_text):
        color = colors[i % len(colors)]
        
        for block in paragraph['blocks']:
            x_min, y_min, x_max, y_max = block['bbox']
            cv2.rectangle(viz_image, 
                         (int(x_min), int(y_min)), 
                         (int(x_max), int(y_max)), 
                         color, 2)
    
    # Resize for display if needed
    h, w = viz_image.shape[:2]
    max_height = 800
    if h > max_height:
        scale = max_height / h
        viz_image = cv2.resize(viz_image, (int(w * scale), max_height))
    
    # Save visualization
    cv2.imwrite('layout_analysis.jpg', viz_image)
    print("Layout analysis visualization saved as 'layout_analysis.jpg'")
```

### Multilingual Text Recognition

```python
def detect_and_recognize_languages(image_path):
    """Auto-detect languages in an image and perform OCR."""
    # First pass with English to detect potential other languages
    eng_reader = easyocr.Reader(['en'])
    eng_results = eng_reader.readtext(image_path)
    
    text = ' '.join([t for _, t, _ in eng_results])
    
    # Language detection based on character sets
    language_patterns = {
        'zh': r'[\u4e00-\u9fff]',  # Chinese
        'ja': r'[\u3040-\u309f\u30a0-\u30ff]',  # Japanese
        'ko': r'[\uac00-\ud7af]',  # Korean
        'ar': r'[\u0600-\u06ff]',  # Arabic
        'hi': r'[\u0900-\u097f]',  # Hindi
        'ru': r'[\u0400-\u04ff]',  # Russian/Cyrillic
        'th': r'[\u0e00-\u0e7f]',  # Thai
    }
    
    detected_languages = ['en']  # Start with English
    
    for lang, pattern in language_patterns.items():
        if re.search(pattern, text):
            detected_languages.append(lang)
    
    print(f"Detected languages: {', '.join(detected_languages)}")
    
    # Perform OCR with detected languages
    if len(detected_languages) > 1:
        multi_reader = easyocr.Reader(detected_languages)
        final_results = multi_reader.readtext(image_path)
    else:
        final_results = eng_results
    
    return {
        'detected_languages': detected_languages,
        'results': final_results
    }
```

### Table Detection and Extraction

```python
def detect_and_extract_tables(image_path, languages=['en']):
    """Detect and extract tables from document images."""
    # Read the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Find horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    # Combine lines
    table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    # Find contours of table regions
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize reader
    reader = easyocr.Reader(languages)
    
    tables = []
    
    for i, contour in enumerate(contours):
        # Filter out small contours
        if cv2.contourArea(contour) < 10000:  # Adjust threshold as needed
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract table region
        table_region = image[y:y+h, x:x+w]
        
        # Save temporarily
        temp_path = f"temp_table_{i}.jpg"
        cv2.imwrite(temp_path, table_region)
        
        # Process the table region
        results = reader.readtext(temp_path)
        
        # Organize text by row based on y-coordinate
        rows = {}
        for bbox, text, conf in results:
            # Adjust bbox coordinates relative to the table region
            bbox = [
                [p[0], p[1]] for p in bbox
            ]
            
            # Get center y-coordinate
            center_y = sum(p[1] for p in bbox) / 4
            
            # Group by row (using a 20-pixel threshold)
            row_idx = int(center_y / 20)
            if row_idx not in rows:
                rows[row_idx] = []
            
            # Add text with x-coordinate for sorting
            center_x = sum(p[0] for p in bbox) / 4
            rows[row_idx].append({
                'text': text,
                'confidence': conf,
                'bbox': bbox,
                'center_x': center_x
            })
        
        # Sort each row by x-coordinate
        table_data = []
        for row_idx in sorted(rows.keys()):
            # Sort cells in this row by x-coordinate
            sorted_row = sorted(rows[row_idx], key=lambda cell: cell['center_x'])
            
            # Extract just the text values
            row_texts = [cell['text'] for cell in sorted_row]
            table_data.append(row_texts)
        
        tables.append({
            'table_id': i,
            'position': (x, y, w, h),
            'data': table_data
        })
        
        # Clean up
        os.remove(temp_path)
    
    return tables

def tables_to_dataframes(tables):
    """Convert extracted tables to pandas DataFrames."""
    dataframes = []
    
    for table in tables:
        # Try to use the first row as headers
        if len(table['data']) > 1:
            headers = table['data'][0]
            data = table['data'][1:]
            
            # Ensure all rows have the same length as headers
            max_cols = len(headers)
            normalized_data = []
            for row in data:
                if len(row) < max_cols:
                    # Pad short rows
                    normalized_data.append(row + [''] * (max_cols - len(row)))
                elif len(row) > max_cols:
                    # Truncate long rows
                    normalized_data.append(row[:max_cols])
                else:
                    normalized_data.append(row)
            
            df = pd.DataFrame(normalized_data, columns=headers)
            dataframes.append({
                'table_id': table['table_id'],
                'dataframe': df
            })
        else:
            # Not enough rows for headers, use default column names
            if table['data']:
                df = pd.DataFrame(table['data'], columns=[f'Column_{i}' for i in range(len(table['data'][0]))])
                dataframes.append({
                    'table_id': table['table_id'],
                    'dataframe': df
                })
    
    return dataframes
```

### Exercise 7:
1. Find a document with a complex layout (e.g., newspaper, magazine)
2. Apply the layout analysis function to extract structured text
3. Visualize the detected layout regions
4. For a document with tables, extract and convert them to pandas DataFrames

---

## Final Assessment

### Knowledge Check:
1. What are the key components of the EasyOCR library?
2. Describe the typical OCR pipeline from image input to structured text output.
3. What preprocessing techniques would you use for a faded receipt with a noisy background?
4. Explain how to optimize EasyOCR for processing a large batch of documents.
5. What strategies would you use to extract structured information from semi-structured documents?

### Practical Challenge:
Develop a complete OCR system for one of the following use cases:
1. Business card information extraction
2. License plate recognition
3. Form data extraction
4. Menu item and price extraction
5. ID document information extraction

Your solution should include:
- Preprocessing pipeline
- OCR implementation
- Post-processing for accuracy
- Structured data extraction
- Simple user interface for input/output
- Unit tests for key components

### Project Presentation:
Prepare a 5-minute presentation that covers:
1. Problem statement and approach
2. Key technologies used
3. Challenges faced and solutions
4. Demonstration of your working solution
5. Future improvements

---

## Resources for Further Learning

### Recommended Reading:
1. "Digital Image Processing" by Gonzalez and Woods
2. "Deep Learning" by Goodfellow, Bengio, and Courville
3. "Python for Data Analysis" by Wes McKinney

### Online Resources:
1. [EasyOCR GitHub Repository](https://github.com/JaidedAI/EasyOCR)
2. [OpenCV Documentation](https://docs.opencv.org/)
3. [PyTorch Tutorials](https://pytorch.org/tutorials/)
4. [Regular Expressions Documentation](https://docs.python.org/3/library/re.html)
5. [Flask Documentation](https://flask.palletsprojects.com/)

### Tools and Libraries:
1. [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
2. [Pandas](https://pandas.pydata.org/)
3. [scikit-learn](https://scikit-learn.org/)
4. [Streamlit](https://streamlit.io/) (for building interactive applications)
5. [Docker](https://www.docker.com/) (for deployment)

---

## Conclusion

Congratulations on completing this comprehensive EasyOCR bootcamp! You've learned how to extract text from images, process and analyze that text, and build practical applications that solve real-world problems.

Remember that OCR is a continuously evolving field, and there's always room for improvement. Keep experimenting with different preprocessing techniques, model configurations, and post-processing methods to achieve the best results for your specific use cases.

We hope this course has provided you with the knowledge and confidence to implement OCR solutions in your own projects. Good luck with your future OCR endeavors!
