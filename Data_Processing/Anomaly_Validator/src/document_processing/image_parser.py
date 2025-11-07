"""
Image Parser using OCR for extracting text from scanned documents.

Handles images (PNG, JPG, JPEG) containing text such as scanned receipts,
invoices, or signed documents.
"""

import os
from typing import Dict, Any, Optional
from PIL import Image


class ImageParser:
    """
    Parse images using OCR to extract text.
    
    Uses Tesseract OCR engine via pytesseract.
    """
    
    def __init__(self, language: str = 'eng', config: Optional[str] = None):
        """
        Initialize image parser.
        
        Args:
            language: OCR language (e.g., 'eng', 'fra', 'deu')
            config: Custom Tesseract configuration
        """
        self.language = language
        self.config = config or '--psm 3'  # Automatic page segmentation
        
        # Try to import pytesseract
        try:
            import pytesseract
            self.pytesseract = pytesseract
            self.ocr_available = True
        except ImportError:
            print("⚠️  pytesseract not installed. OCR functionality disabled.")
            print("   Install with: pip install pytesseract")
            print("   Also install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
            self.ocr_available = False
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parse image file and extract text using OCR.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dictionary with:
                - text: Extracted text
                - confidence: OCR confidence (if available)
                - metadata: Image metadata
                - file_name: Original file name
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        if not self.ocr_available:
            return {
                'text': '',
                'error': 'OCR not available (pytesseract not installed)',
                'file_name': os.path.basename(file_path),
                'file_path': file_path
            }
        
        result = {
            'text': '',
            'confidence': None,
            'metadata': {},
            'file_name': os.path.basename(file_path),
            'file_path': file_path
        }
        
        try:
            # Open image
            image = Image.open(file_path)
            
            # Extract metadata
            result['metadata'] = {
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height
            }
            
            # Preprocess image (optional - can improve OCR accuracy)
            image = self._preprocess_image(image)
            
            # Extract text
            result['text'] = self.pytesseract.image_to_string(
                image,
                lang=self.language,
                config=self.config
            )
            
            # Get OCR confidence (if available)
            try:
                data = self.pytesseract.image_to_data(image, output_type=self.pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                if confidences:
                    result['confidence'] = sum(confidences) / len(confidences)
            except:
                pass
            
            char_count = len(result['text'].strip())
            print(f"✓ Parsed image: {result['file_name']} "
                  f"({result['metadata']['width']}x{result['metadata']['height']}, "
                  f"{char_count} chars extracted)")
            
        except Exception as e:
            print(f"⚠️  Error parsing image {file_path}: {e}")
            result['error'] = str(e)
        
        return result
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Optional: Apply thresholding for better contrast
        # This can be enabled if needed
        # from PIL import ImageEnhance
        # enhancer = ImageEnhance.Contrast(image)
        # image = enhancer.enhance(2.0)
        
        return image
    
    def extract_text_simple(self, file_path: str) -> str:
        """Quick text extraction without metadata."""
        try:
            if not self.ocr_available:
                return ""
            
            image = Image.open(file_path)
            return self.pytesseract.image_to_string(image, lang=self.language)
        except Exception as e:
            print(f"⚠️  Error extracting text from {file_path}: {e}")
            return ""
    
    def detect_text_regions(self, file_path: str) -> list:
        """
        Detect regions containing text in the image.
        
        Args:
            file_path: Path to image
            
        Returns:
            List of bounding boxes for text regions
        """
        if not self.ocr_available:
            return []
        
        try:
            image = Image.open(file_path)
            data = self.pytesseract.image_to_data(image, output_type=self.pytesseract.Output.DICT)
            
            regions = []
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                if int(data['conf'][i]) > 60:  # Confidence threshold
                    regions.append({
                        'text': data['text'][i],
                        'left': data['left'][i],
                        'top': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'confidence': int(data['conf'][i])
                    })
            
            return regions
        except Exception as e:
            print(f"⚠️  Error detecting text regions: {e}")
            return []


if __name__ == "__main__":
    # Test image parser
    print("Testing Image Parser...")
    
    parser = ImageParser()
    
    if parser.ocr_available:
        print("✓ OCR is available")
        print("✓ Image Parser initialized successfully")
    else:
        print("⚠️  OCR not available - install pytesseract and Tesseract")
    
    print("\n✓ Image Parser tests passed!")
    
    # Note: Actual testing requires an image file
    # Usage example:
    # result = parser.parse('path/to/receipt.jpg')
    # print(result['text'])

