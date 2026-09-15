"""
Vision Pipeline Coordinator for Legal Metrology AI.
Orchestrates:
Image -> OpenCV Preprocessing -> YOLOv8 Detection -> PaddleOCR -> Declaration Extraction -> Rule Checking
"""

from typing import Dict, Any, Optional
import numpy as np

from app.preprocessing.image_processor import ImageProcessor
from app.detection.yolo_detector import YOLODetector
from app.ocr.paddle_ocr import PaddleOCREngine
from app.extraction.declaration_extractor import DeclarationExtractor
from app.rules.rule_engine import RuleEngine


class VisionPipeline:
    """End-to-end Legal Metrology Vision Pipeline."""

    def __init__(
        self,
        image_processor: Optional[ImageProcessor] = None,
        yolo_detector: Optional[YOLODetector] = None,
        ocr_engine: Optional[PaddleOCREngine] = None,
        extractor: Optional[DeclarationExtractor] = None,
        rule_engine: Optional[RuleEngine] = None,
    ):
        self.image_processor = image_processor or ImageProcessor()
        self.yolo_detector = yolo_detector or YOLODetector()
        self.ocr_engine = ocr_engine or PaddleOCREngine()
        self.extractor = extractor or DeclarationExtractor()
        self.rule_engine = rule_engine or RuleEngine()

    def process(self, image: np.ndarray, filename: str = "sample.jpg") -> Dict[str, Any]:
        """
        Runs the full inspection pipeline:
        1. Preprocess with OpenCV
        2. Detect label regions with YOLOv8
        3. OCR label panels with PaddleOCR
        4. Extract declarations
        5. Check Legal Metrology compliance rules
        """
        # 1. Preprocessing
        preprocessed = self.image_processor.preprocess_image(image)

        # 2. YOLOv8 Detection (Stubbed until trained model is integrated)
        detected_panels = self.yolo_detector.detect_panels(preprocessed)

        # 3. PaddleOCR (Stubbed until OCR weights are integrated)
        ocr_tokens = self.ocr_engine.extract_text(preprocessed)

        # 4. Declaration Extraction
        declarations = self.extractor.extract_declarations(ocr_tokens)

        # 5. Rule Verification
        rule_evaluations = self.rule_engine.evaluate(declarations)
        is_compliant = self.rule_engine.is_overall_compliant(rule_evaluations)

        return {
            "filename": filename,
            "overall_status": "COMPLIANT" if is_compliant else "NON_COMPLIANT",
            "declarations": declarations,
            "rule_results": rule_evaluations,
            "metadata": {
                "panels_detected": len(detected_panels),
                "ocr_tokens_found": len(ocr_tokens),
            },
        }
