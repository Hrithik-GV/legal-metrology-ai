"""
OpenCV Image Preprocessor for Legal Metrology AI.
Handles image normalization, contrast enhancement (CLAHE), and denoising.
"""

from typing import Optional, Tuple
import numpy as np


class ImageProcessor:
    """Provides OpenCV image processing utilities for packaging label analysis."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applies standard pre-processing:
        - Grayscale conversion (if 3-channel)
        - CLAHE (Contrast Limited Adaptive Histogram Equalization)
        - Bilateral filtering for text edge preservation
        """
        try:
            import cv2
        except ImportError:
            # Stubs gracefully if cv2 is not yet initialized
            return image

        if image is None or image.size == 0:
            raise ValueError("Empty image provided for preprocessing")

        # Convert to grayscale if color image
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = image.copy()

        # CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced = clahe.apply(gray)

        # Bilateral filter reduces noise while keeping edges sharp for OCR
        denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

        return denoised

    def crop_region(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """Crops a bounding box region from the image safely."""
        h, w = image.shape[:2]
        x1_clamped = max(0, min(x1, w))
        y1_clamped = max(0, min(y1, h))
        x2_clamped = max(0, min(x2, w))
        y2_clamped = max(0, min(y2, h))

        if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
            return image

        return image[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
