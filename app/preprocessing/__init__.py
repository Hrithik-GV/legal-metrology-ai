"""
OpenCV Image Preprocessing Module
"""

from .image_processor import (
    ImageProcessor,
    PreprocessResult,
    ImageProcessingError,
    preprocess_image,
)

__all__ = [
    "ImageProcessor",
    "PreprocessResult",
    "ImageProcessingError",
    "preprocess_image",
]
