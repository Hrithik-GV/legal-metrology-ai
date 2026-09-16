"""
PaddleOCR Module
"""

from .paddle_ocr import (
    PaddleOCREngine,
    FallbackOCREngine,
    extract_text,
    annotate_ocr_image,
)

__all__ = [
    "PaddleOCREngine",
    "FallbackOCREngine",
    "extract_text",
    "annotate_ocr_image",
]
