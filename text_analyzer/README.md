# Text Analyzer

Text Analyzer is a Digital Image Processing project for extracting and analyzing text from images. This project is designed to detect text regions, recognize the text content, and identify the language of the extracted text. It was developed as part of the ICS 2412 Digital Image Processing Term Project.

## Features

- **Image Preprocessing**: Apply various preprocessing techniques to enhance text visibility in images:
  - Grayscale conversion
  - Gaussian blur for noise reduction
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Adaptive thresholding and binarization
  - Skew correction for rotated text
  
- **Text Detection**: Detect text regions using advanced computer vision techniques:
  - EAST (Efficient and Accurate Scene Text Detector) model
  - Contour-based detection method as fallback
  - Non-maxima suppression for overlapping regions
  
- **Optical Character Recognition**: Convert detected text regions into machine-readable text:
  - EasyOCR integration for multi-language support
  - Confidence scores for recognized text
  - GPU acceleration for faster processing
  
- **Language Identification**: Identify the language of the extracted text:
  - langdetect library for primary detection
  - Character frequency analysis as fallback
  - Support for multiple languages
  
- **Visualization and Analysis**:
  - Real-time visualization of each processing step
  - Color histograms for image analysis
  - Detailed metrics on processing performance
  - Interactive web interface using Streamlit

## Project Structure

```
text_analyzer/
├── app.py                   # Streamlit web application
├── config/                  # Configuration files
│   ├── __init__.py
│   ├── model_config.py      # OCR model parameters
│   ├── paths.py             # Project paths
│   └── preprocessing_config.py  # Image preprocessing parameters
├── data/                    # Data directory
│   ├── input/               # Input images
│   └── output/              # Output results
├── models/                  # Model files
├── pipeline/                # Processing pipeline components
│   ├── __init__.py
│   ├── language_id.py       # Language identification
│   ├── preprocessing.py     # Image preprocessing
│   ├── recognition.py       # Text recognition
│   └── text_detection.py    # Text region detection
├── main.py                  # Main application script
└── requirements.txt         # Project dependencies
```

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- EasyOCR (for text recognition)
- Streamlit (for web interface)
- Matplotlib (for visualizations)
- langdetect (optional, for language detection)
- PyTorch (required by EasyOCR)

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the EAST text detection model (if using):
   ```
   mkdir -p models
   curl -o models/frozen_east_text_detection.pb https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/download_models.py
   python models/download_models.py
   ```

## Usage

### Web Application (Recommended)

The easiest way to use the Text Analyzer is through the Streamlit web interface:

```bash
# Run the Streamlit app
streamlit run app.py
```

This will open a web browser with the interactive application where you can:
- Upload images or use your camera to capture text
- Configure preprocessing parameters
- View the entire image processing pipeline step-by-step
- Analyze text detection and recognition results
- Examine image histograms and filtering techniques

### Command Line

```bash
# Process a single image
python main.py --input path/to/image.jpg

# Process all images in a directory
python main.py --input path/to/directory

# Specify output file
python main.py --input path/to/image.jpg --output results.txt

# Specify languages (comma-separated)
python main.py --input path/to/image.jpg --languages en,fr,es

# Use GPU acceleration
python main.py --input path/to/image.jpg --gpu

# Disable visualization output
python main.py --input path/to/image.jpg --no-vis
```

### Python API

```python
from text_analyzer.main import TextAnalyzer

# Create analyzer
analyzer = TextAnalyzer()

# Process image
results = analyzer.process_image('path/to/image.jpg')

# Save results
analyzer.save_results_to_txt(results, 'output.txt')
```

## Digital Image Processing Techniques

This project demonstrates several key Digital Image Processing concepts:

1. **Image Enhancement**:
   - Noise reduction using Gaussian filtering
   - Contrast enhancement using CLAHE
   - Binarization using adaptive thresholding

2. **Morphological Operations**:
   - Dilation to connect text components
   - Structural element design for text features

3. **Feature Extraction**:
   - Contour detection and analysis
   - Geometric property calculation (area, aspect ratio)

4. **Deep Learning Integration**:
   - CNN-based text detection (EAST)
   - OCR using CRNN architecture (via EasyOCR)

5. **Image Analysis**:
   - Histogram analysis for image characteristics
   - Edge detection using Sobel operators

## License

This project is licensed under the MIT License - see the LICENSE file for details.
