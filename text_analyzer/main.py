import argparse
import logging
import os
import sys
import cv2
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import project components
from config.preprocessing_config import PreprocessingConfig
from config.model_config import ModelConfig
from config.paths import PathConfig
from pipeline.preprocessing import ImagePreprocessor
from pipeline.text_detection import TextDetector
from pipeline.recognition import TextRecognizer
from pipeline.language_id import LanguageIdentifier

class TextAnalyzer:
    """Main text analyzer application"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load configurations
        self.path_config = PathConfig()
        self.preproc_config = PreprocessingConfig()
        self.model_config = ModelConfig()
        
        # Initialize pipeline components
        self.preprocessor = ImagePreprocessor(self.preproc_config)
        self.text_detector = TextDetector(
            east_model_path=self.path_config.east_model_path,
            min_confidence=self.model_config.detection_confidence
        )
        self.text_recognizer = TextRecognizer(
            languages=self.model_config.languages,
            gpu=self.model_config.use_gpu
        )
        self.language_id = LanguageIdentifier(
            fallback_lang=self.model_config.fallback_language
        )
        
        self.logger.info("Text Analyzer initialized")
    
    def process_image(self, image_path, save_visualization=True):
        """
        Process an image to extract and analyze text
        
        Args:
            image_path: Path to input image
            save_visualization: Whether to save visualization images
            
        Returns:
            Dictionary with analysis results
        """
        if not os.path.exists(image_path):
            self.logger.error(f"Image not found: {image_path}")
            return {"error": "Image not found"}
        
        try:
            # Load image
            self.logger.info(f"Processing image: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return {"error": "Failed to load image"}
            
            # Preprocess image
            preprocessed = self.preprocessor.process(image)
            
            # Save preprocessed image if requested
            if save_visualization:
                filename = os.path.basename(image_path)
                base_name, ext = os.path.splitext(filename)
                preproc_path = self.path_config.get_output_path(f"{base_name}_preprocessed{ext}")
                cv2.imwrite(preproc_path, preprocessed)
                self.logger.info(f"Saved preprocessed image to {preproc_path}")
            
            # Detect text regions
            self.logger.info("Detecting text regions")
            boxes = self.text_detector.detect(preprocessed)
            self.logger.info(f"Detected {len(boxes)} text regions")
            
            # Recognize text in detected regions
            self.logger.info("Recognizing text")
            recognition_results = self.text_recognizer.recognize_text(image, boxes)
            
            # Filter results by confidence
            filtered_results = [
                result for result in recognition_results 
                if result['confidence'] >= self.model_config.recognition_min_confidence
            ]
            
            # Identify language of recognized text
            all_text = " ".join([result['text'] for result in filtered_results])
            language_info = self.language_id.identify_language(all_text)
            
            # Save visualization if requested
            if save_visualization:
                vis_image = self.text_recognizer.visualize_results(image, filtered_results)
                vis_path = self.path_config.get_output_path(f"{base_name}_results{ext}")
                cv2.imwrite(vis_path, vis_image)
                self.logger.info(f"Saved visualization to {vis_path}")
            
            # Prepare results
            results = {
                "filename": os.path.basename(image_path),
                "text_regions": len(boxes),
                "recognized_items": len(filtered_results),
                "language": language_info,
                "text_content": filtered_results,
                "full_text": all_text
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return {"error": str(e)}
    
    def process_directory(self, directory_path, save_visualization=True):
        if not os.path.isdir(directory_path):
            self.logger.error(f"Directory not found: {directory_path}")
            return {"error": "Directory not found"}
        
        results = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Find all image files in directory
        image_files = []
        for ext in image_extensions:
            image_files.extend(
                [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                    if f.lower().endswith(ext)]
            )
        
        self.logger.info(f"Found {len(image_files)} images in {directory_path}")
        
        # Process each image
        for image_path in image_files:
            result = self.process_image(image_path, save_visualization)
            results.append(result)
        
        return results
    
    def save_results_to_txt(self, results, output_path=None):
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.path_config.get_output_path(f"results_{timestamp}.txt")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if isinstance(results, list):
                f.write(f"Text Analysis Results - {len(results)} images\n")
                f.write("=" * 50 + "\n\n")
                
                for i, result in enumerate(results):
                    f.write(f"Image {i+1}: {result.get('filename', 'Unknown')}\n")
                    f.write("-" * 30 + "\n")
                    
                    if "error" in result:
                        f.write(f"Error: {result['error']}\n\n")
                        continue
                    
                    f.write(f"Text regions: {result.get('text_regions', 0)}\n")
                    f.write(f"Recognized items: {result.get('recognized_items', 0)}\n")
                    
                    language = result.get('language', {})
                    f.write(f"Language: {language.get('lang_name', 'Unknown')} ")
                    f.write(f"(confidence: {language.get('confidence', 0):.2f})\n\n")
                    
                    f.write("Extracted text:\n")
                    f.write(result.get('full_text', 'No text extracted') + "\n\n")
                    
                    f.write("=" * 50 + "\n\n")
            else:
                # Single image result
                f.write("Text Analysis Results\n")
                f.write("=" * 50 + "\n\n")
                
                if "error" in results:
                    f.write(f"Error: {results['error']}\n")
                else:
                    f.write(f"Image: {results.get('filename', 'Unknown')}\n")
                    f.write(f"Text regions: {results.get('text_regions', 0)}\n")
                    f.write(f"Recognized items: {results.get('recognized_items', 0)}\n")
                    
                    language = results.get('language', {})
                    f.write(f"Language: {language.get('lang_name', 'Unknown')} ")
                    f.write(f"(confidence: {language.get('confidence', 0):.2f})\n\n")
                    
                    f.write("Extracted text:\n")
                    f.write(results.get('full_text', 'No text extracted') + "\n\n")
                    
                    f.write("\nDetailed results:\n")
                    f.write("-" * 30 + "\n")
                    
                    for i, item in enumerate(results.get('text_content', [])):
                        f.write(f"Item {i+1}:\n")
                        f.write(f"  Text: {item.get('text', '')}\n")
                        f.write(f"  Confidence: {item.get('confidence', 0):.2f}\n")
                        f.write(f"  Position: {item.get('position', '')}\n")
                        f.write("\n")
        
        self.logger.info(f"Results saved to {output_path}")
        return output_path


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Text Analyzer - Extract and analyze text from images")
    parser.add_argument("--input", "-i", required=True, help="Input image file or directory")
    parser.add_argument("--output", "-o", help="Output text file (optional)")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization output")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for text recognition")
    parser.add_argument("--languages", "-l", default="en", help="Languages for OCR (comma-separated)")
    
    args = parser.parse_args()
    
    # Create text analyzer
    analyzer = TextAnalyzer()
    
    # Configure analyzer based on command line arguments
    if args.languages:
        languages = args.languages.split(",")
        analyzer.model_config.set_languages(languages)
        analyzer.text_recognizer.languages = languages
    
    if args.gpu:
        analyzer.model_config.enable_gpu(True)
        analyzer.text_recognizer.gpu = True
    
    # Process input
    if os.path.isdir(args.input):
        results = analyzer.process_directory(args.input, not args.no_vis)
    else:
        results = analyzer.process_image(args.input, not args.no_vis)
    
    # Save results
    output_path = analyzer.save_results_to_txt(results, args.output)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
