import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
import logging
import time
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to sys.path to allow imports from subdirectories
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import project components
from pipeline.preprocessing import ImagePreprocessor
from pipeline.text_detection import TextDetector
from pipeline.recognition import TextRecognizer
from pipeline.language_id import LanguageIdentifier
from config.preprocessing_config import PreprocessingConfig
from config.model_config import ModelConfig
from config.paths import PathConfig

# Set page configuration
st.set_page_config(
    page_title="Digital Image Processing - Text Analysis", 
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
<style>
    .main-header {text-align: center; color: #2c3e50; font-size: 42px; margin-bottom: 30px;}
    .subheader {color: #34495e; font-size: 26px; margin-top: 20px; margin-bottom: 20px;}
    .success-box {background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin-bottom: 20px; color: #1b5e20; border: 1px solid #2e7d32;}
    .info-box {background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin-bottom: 15px; color: #0d47a1; border: 1px solid #1565c0;}
    .section-divider {margin-top: 30px; margin-bottom: 30px; border-top: 2px solid #ecf0f1;}
    /* Added styles for better readability */
    h3 {color: #2c3e50; font-weight: 600;}
    .stApp {color: #333333;}
    strong {color: #0d47a1; font-weight: 600;}
    .success-box strong {color: #1b5e20; font-weight: 600;}
    .success-box p {color: #2e7d32; margin-bottom: 10px;}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>📷 Digital Image Processing:<br>Text Analysis System</h1>", unsafe_allow_html=True)

# Subtitle
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Extract, process, and analyze text from images with advanced computer vision techniques</p>", unsafe_allow_html=True)

# Initialize components
paths = PathConfig()
preproc_config = PreprocessingConfig()
model_config = ModelConfig()

# Create processing pipeline components
@st.cache_resource
def initialize_pipeline():
    preprocessor = ImagePreprocessor(preproc_config)
    text_detector = TextDetector(min_confidence=model_config.detection_confidence)
    text_recognizer = TextRecognizer(languages=model_config.languages, gpu=model_config.use_gpu)
    language_id = LanguageIdentifier(fallback_lang=model_config.fallback_language)
    
    logger.info("Text analysis pipeline initialized successfully")
    return preprocessor, text_detector, text_recognizer, language_id

preprocessor, text_detector, text_recognizer, language_id = initialize_pipeline()

# --- Sidebar for Options ---
st.sidebar.header("⚙️ Options")

# Input method selection
input_method = st.sidebar.radio("Choose input method:", ("Upload Image", "Use Camera"))

# Language selection (optional, based on analyzer capabilities)
# Assuming EasyOCR default is English, add more if needed
# supported_langs = analyzer.text_recognizer.reader.get_supported_language() # Might need adjustment based on easyocr version
default_langs = ['en']
selected_langs = st.sidebar.multiselect(
    "Select OCR Languages:",
    options=['en', 'fr', 'es', 'de', 'ja', 'ko', 'zh-cn', 'ru', 'ar', 'sw'], # Add more common languages
    default=default_langs
)
model_config.set_languages(selected_langs)
text_recognizer.languages = selected_langs # Update recognizer directly

# GPU option
use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=model_config.use_gpu)
model_config.enable_gpu(use_gpu)
text_recognizer.gpu = use_gpu # Update recognizer directly

# Confidence Thresholds
detection_conf = st.sidebar.slider("Detection Confidence", 0.1, 0.9, float(model_config.detection_confidence), 0.05)
recognition_conf = st.sidebar.slider("Recognition Confidence", 0.1, 0.9, float(model_config.recognition_min_confidence), 0.05)
model_config.set_detection_confidence(detection_conf)
model_config.set_recognition_confidence(recognition_conf)
# Update detector directly if needed (depends on implementation)
# analyzer.text_detector.min_confidence = detection_conf

# Preprocessing options
st.sidebar.subheader("Preprocessing Steps")
preproc_config.use_gaussian = st.sidebar.checkbox("Apply Gaussian Blur", value=preproc_config.use_gaussian)
preproc_config.use_clahe = st.sidebar.checkbox("Apply CLAHE", value=preproc_config.use_clahe)
preproc_config.use_binarization = st.sidebar.checkbox("Apply Binarization", value=preproc_config.use_binarization)
preproc_config.correct_skew = st.sidebar.checkbox("Correct Skew", value=preproc_config.correct_skew)

# --- Image Input Area ---
image_file = None
img_array = None

if input_method == "Upload Image":
    image_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    if image_file:
        try:
            # Read image using PIL and convert to OpenCV format
            pil_image = Image.open(image_file)
            img_array = np.array(pil_image)
            # Convert RGB (PIL) to BGR (OpenCV)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                 img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4: # Handle RGBA
                 img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

            st.image(pil_image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
            img_array = None


elif input_method == "Use Camera":
    camera_input = st.camera_input("Take a picture")
    if camera_input:
        try:
            # Read image from camera buffer using PIL and convert to OpenCV format
            pil_image = Image.open(camera_input)
            img_array = np.array(pil_image)
            # Convert RGB (PIL) to BGR (OpenCV)
            if len(img_array.shape) == 3:
                 img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            # No need to display again, camera_input widget shows preview
        except Exception as e:
            st.error(f"Error processing camera input: {e}")
            img_array = None


# --- Processing and Displaying Results ---
if img_array is not None:
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='subheader'>🔍 Image Analysis</h2>", unsafe_allow_html=True)
    
    # Process analysis in steps to show the Digital Image Processing pipeline
    with st.spinner("Processing image..."):
        try:
            # Track processing time for analysis
            start_time = time.time()
            
            # Use a temporary file for potential file-based operations
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                # Save the OpenCV image array to the temporary file
                cv2.imwrite(tmp_file.name, img_array)
                temp_image_path = tmp_file.name
            
            # Step 1: Preprocessing
            st.markdown("### Step 1: Image Preprocessing")
            col_orig, col_prep = st.columns(2)
            
            with col_orig:
                # Display original image
                st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), 
                         caption="Original Image", 
                         use_column_width=True)
            
            # Apply preprocessing
            preprocessed_image = preprocessor.process(img_array)
            
            with col_prep:
                # Display preprocessed image
                if len(preprocessed_image.shape) == 2:  # Grayscale
                    disp_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2RGB)
                else:  # Color
                    disp_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
                
                st.image(disp_image, 
                         caption="Preprocessed Image", 
                         use_column_width=True)
                
                preprocessing_time = time.time() - start_time
                st.text(f"Preprocessing completed in {preprocessing_time:.2f} seconds")
            
            # Step 2: Text Detection
            st.markdown("### Step 2: Text Region Detection")
            detection_start = time.time()
            
            # Detect text regions
            boxes = text_detector.detect(preprocessed_image)
            
            # Create a visualization with detected regions
            detection_vis = img_array.copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(detection_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display detection results
            st.image(cv2.cvtColor(detection_vis, cv2.COLOR_BGR2RGB), 
                     caption=f"Detected {len(boxes)} Text Regions", 
                     use_column_width=True)
            
            detection_time = time.time() - detection_start
            st.text(f"Text detection completed in {detection_time:.2f} seconds")
            
            # Step 3: Text Recognition
            st.markdown("### Step 3: Text Recognition")
            recognition_start = time.time()
            
            # Initialize results dict
            results = {}
            
            # Recognize text in detected regions
            recognition_results = text_recognizer.recognize_text(img_array, boxes)
            
            # Filter results by confidence
            filtered_results = [
                result for result in recognition_results 
                if result['confidence'] >= model_config.recognition_min_confidence
            ]
            
            # Create a visualization with recognized text
            ocr_vis = img_array.copy()
            for result in filtered_results:
                text = result['text']
                (x, y, w, h) = result['position']
                conf = result.get('confidence', 0)
                
                # Draw bounding box
                cv2.rectangle(ocr_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Prepare text to display
                display_text = f"{text} ({conf:.2f})"
                
                # Determine text position
                text_y = y - 10 if y - 10 > 10 else y + h + 20
                
                # Display text
                cv2.putText(ocr_vis, display_text, (x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Display recognition results
            st.image(cv2.cvtColor(ocr_vis, cv2.COLOR_BGR2RGB), 
                     caption="Text Recognition Results", 
                     use_column_width=True)
            
            recognition_time = time.time() - recognition_start
            st.text(f"Text recognition completed in {recognition_time:.2f} seconds")
            
            # Step 4: Language Identification
            st.markdown("### Step 4: Language Identification")
            langid_start = time.time()
            
            # Extract all recognized text
            all_text = " ".join([result['text'] for result in filtered_results])
            
            # Identify language
            language_info = language_id.identify_language(all_text)
            
            # Store results for potential later use
            results['text_content'] = filtered_results
            results['full_text'] = all_text
            results['language'] = language_info
            results['text_regions'] = len(boxes)
            results['recognized_items'] = len(filtered_results)
            
            # Display language identification results
            st.markdown(f"""
            <div class='success-box'>
                <h3>Analysis Results</h3>
                <p><strong>Detected Language:</strong> {language_info['lang_name']} (Confidence: {language_info['confidence']:.2f})</p>
                <p><strong>Text Regions:</strong> {len(boxes)}</p>
                <p><strong>Recognized Items:</strong> {len(filtered_results)}</p>
                <p><strong>Total Processing Time:</strong> {time.time() - start_time:.2f} seconds</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the extracted text
            st.subheader("📝 Extracted Text")
            if all_text:
                st.text_area("Full Text Content", all_text, height=150)
            else:
                st.info("No text was recognized in the image.")
            
            # Show detailed results in an expander
            with st.expander("View Detailed Recognition Results"):
                if filtered_results:
                    # Create a DataFrame for results
                    result_data = []
                    for i, res in enumerate(filtered_results):
                        result_data.append({
                            "Item": i+1,
                            "Text": res['text'],
                            "Confidence": f"{res['confidence']:.2f}",
                            "Position": f"({res['position'][0]}, {res['position'][1]})"
                        })
                    
                    st.table(result_data)
                else:
                    st.info("No detailed results available.")
            
            # Generate histograms for the original and processed images (common in DIP projects)
            with st.expander("Image Histograms Analysis"):
                hist_col1, hist_col2 = st.columns(2)
                
                with hist_col1:
                    st.markdown("**Original Image Histogram**")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    if len(img_array.shape) == 3:  # Color image
                        colors = ('b', 'g', 'r')
                        for i, color in enumerate(colors):
                            hist = cv2.calcHist([img_array], [i], None, [256], [0, 256])
                            ax.plot(hist, color=color)
                        ax.set_title('Color Histogram (BGR)')
                    else:  # Grayscale
                        hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
                        ax.plot(hist, color='gray')
                        ax.set_title('Grayscale Histogram')
                    
                    ax.set_xlabel('Pixel Value')
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                
                with hist_col2:
                    st.markdown("**Preprocessed Image Histogram**")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    if len(preprocessed_image.shape) == 3:  # Color image
                        colors = ('b', 'g', 'r')
                        for i, color in enumerate(colors):
                            hist = cv2.calcHist([preprocessed_image], [i], None, [256], [0, 256])
                            ax.plot(hist, color=color)
                        ax.set_title('Color Histogram (BGR)')
                    else:  # Grayscale
                        hist = cv2.calcHist([preprocessed_image], [0], None, [256], [0, 256])
                        ax.plot(hist, color='black')
                        ax.set_title('Grayscale Histogram')
                    
                    ax.set_xlabel('Pixel Value')
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
            
            # Image filtering visualizations (common in DIP courses)
            with st.expander("Image Filtering Techniques"):
                st.markdown("#### Various Digital Image Processing Filters")
                filter_col1, filter_col2, filter_col3 = st.columns(3)
                
                # Convert to grayscale for filters if needed
                gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
                
                with filter_col1:
                    # Apply Gaussian blur
                    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
                    st.image(blur, caption="Gaussian Blur", use_column_width=True)
                
                with filter_col2:
                    # Apply Sobel edge detector
                    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
                    sobel = cv2.magnitude(sobelx, sobely)
                    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    st.image(sobel, caption="Sobel Edge Detection", use_column_width=True)
                
                with filter_col3:
                    # Apply thresholding
                    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
                    st.image(thresh, caption="Binary Thresholding", use_column_width=True)
                
                st.markdown("These filtering techniques are common in Digital Image Processing to enhance features for better text detection and recognition.")
            
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                
        except Exception as e:
            st.error(f"An error occurred during image processing: {e}")
            logger.error(f"Processing error: {e}", exc_info=True)
            
            # Make sure temporary file is removed
            if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
else:
    st.info("Please upload an image or use the camera to start the analysis.")

# Footer
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d;'>
        <p>Digital Image Processing - Text Analysis System v1.0</p>
        <p>ICS 2412 Digital Image Processing - Term Project</p>
    </div>
    """, 
    unsafe_allow_html=True
)
