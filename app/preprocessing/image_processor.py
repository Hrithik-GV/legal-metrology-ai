"""
OpenCV Image Preprocessing Module for Legal Metrology AI.
Implements robust preprocessing for packaged commodity label inspection:
- Safe image loading (including Unicode and Windows path support)
- Aspect-ratio preserving resizing
- Bilateral denoising (preserves text edges for OCR)
- Adaptive contrast enhancement (CLAHE on luminance)
- Unsharp mask sharpening
- Optional grayscale conversion
- Non-destructive processed output saving
"""

import os
from pathlib import Path
from typing import NamedTuple, Tuple, Optional, Union
import cv2
import numpy as np


class PreprocessResult(NamedTuple):
    """Result of preprocessing an image."""
    processed_image: np.ndarray
    original_dimensions: Tuple[int, int]  # (height, width)
    processed_dimensions: Tuple[int, int]  # (height, width)
    output_path: str


class ImageProcessingError(Exception):
    """Custom exception raised when image preprocessing fails."""
    pass


class ImageProcessor:
    """
    OpenCV-based image preprocessor for packaging and label inspection.
    """

    def __init__(
        self,
        max_dimension: int = 1920,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        denoise_diameter: int = 7,
        denoise_sigma_color: float = 50.0,
        denoise_sigma_space: float = 50.0,
    ):
        self.max_dimension = max_dimension
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.denoise_diameter = denoise_diameter
        self.denoise_sigma_color = denoise_sigma_color
        self.denoise_sigma_space = denoise_sigma_space

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Safely loads an image from disk, handling Windows/Unicode paths.
        """
        path = Path(image_path).resolve()
        if not path.exists():
            raise ImageProcessingError(f"Image file does not exist: {path}")

        if not path.is_file():
            raise ImageProcessingError(f"Path is not a file: {path}")

        try:
            # np.fromfile + cv2.imdecode avoids Windows unicode filepath issues
            file_bytes = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ImageProcessingError(f"Failed to read image bytes from {path}: {str(e)}")

        if image is None or image.size == 0:
            raise ImageProcessingError(f"Corrupted or invalid image file: {path}")

        return image

    def resize_aspect_ratio(
        self, image: np.ndarray, max_dim: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Resizes image while strictly maintaining the original aspect ratio.
        Only scales down if dimensions exceed max_dim; does not upscale.
        """
        target_max = max_dim or self.max_dimension
        h, w = image.shape[:2]

        if max(h, w) <= target_max:
            return image.copy(), 1.0

        scale = target_max / float(max(h, w))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Bilateral filtering reduces sensor noise and packaging glare while
        keeping text edges and bar codes sharply defined.
        """
        return cv2.bilateralFilter(
            image,
            d=self.denoise_diameter,
            sigmaColor=self.denoise_sigma_color,
            sigmaSpace=self.denoise_sigma_space,
        )

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhances local contrast using CLAHE.
        For BGR color images, applies CLAHE exclusively to the L (luminance) channel in LAB color space.
        For grayscale, applies directly.
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )

        if len(image.shape) == 2:
            return clahe.apply(image)

        # Convert BGR to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Equalize only luminance
        enhanced_l = clahe.apply(l_channel)

        # Merge back and convert to BGR
        merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    def sharpen(self, image: np.ndarray, amount: float = 0.5) -> np.ndarray:
        """
        Applies subtle unsharp masking to enhance fine text readability.
        """
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return sharpened

    def save_image(self, image: np.ndarray, output_path: Union[str, Path]) -> str:
        """
        Saves the processed image safely to disk without altering the original.
        """
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        ext = out_p.suffix.lower() or ".jpg"
        success, encoded = cv2.imencode(ext, image)
        if not success:
            raise ImageProcessingError(f"Failed to encode processed image for {out_p}")

        try:
            with open(out_p, "wb") as f:
                f.write(encoded.tobytes())
        except Exception as e:
            raise ImageProcessingError(f"Failed to save image to {out_p}: {str(e)}")

        return str(out_p)

    def process(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        output_filename: Optional[str] = None,
        to_grayscale: bool = False,
        denoise: bool = True,
        enhance_contrast: bool = True,
        sharpen: bool = True,
    ) -> PreprocessResult:
        """
        Executes the full preprocessing pipeline:
        1. Load image safely
        2. Resize maintaining aspect ratio
        3. Denoise with bilateral filter
        4. Enhance contrast via CLAHE in LAB space
        5. Sharpen fine text contours
        6. Optional grayscale conversion
        7. Save processed output non-destructively
        """
        input_path = Path(image_path).resolve()
        original_img = self.load_image(input_path)
        orig_h, orig_w = original_img.shape[:2]

        # 2. Resize maintaining aspect ratio
        img, _ = self.resize_aspect_ratio(original_img)

        # 3. Denoise
        if denoise:
            img = self.denoise(img)

        # 4. Contrast enhancement
        if enhance_contrast:
            img = self.enhance_contrast(img)

        # 5. Sharpening
        if sharpen:
            img = self.sharpen(img)

        # 6. Optional grayscale
        if to_grayscale:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        proc_h, proc_w = img.shape[:2]

        # 7. Save to output directory
        if output_dir is None:
            # Default to outputs/preprocessed relative to project or input file
            output_dir = input_path.parent.parent / "outputs" / "preprocessed"
        else:
            output_dir = Path(output_dir)

        save_name = output_filename or f"{input_path.stem}_preprocessed{input_path.suffix or '.jpg'}"
        target_path = output_dir / save_name

        saved_path = self.save_image(img, target_path)

        return PreprocessResult(
            processed_image=img,
            original_dimensions=(orig_h, orig_w),
            processed_dimensions=(proc_h, proc_w),
            output_path=saved_path,
        )


def preprocess_image(
    image_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    output_filename: Optional[str] = None,
    to_grayscale: bool = False,
    denoise: bool = True,
    enhance_contrast: bool = True,
    sharpen: bool = True,
    max_dimension: int = 1920,
) -> PreprocessResult:
    """
    Standard top-level preprocessing function.

    Returns:
    - PreprocessResult: tuple with (processed_image, original_dimensions, processed_dimensions, output_path)
    """
    processor = ImageProcessor(max_dimension=max_dimension)
    return processor.process(
        image_path=image_path,
        output_dir=output_dir,
        output_filename=output_filename,
        to_grayscale=to_grayscale,
        denoise=denoise,
        enhance_contrast=enhance_contrast,
        sharpen=sharpen,
    )
