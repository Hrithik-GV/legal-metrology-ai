"""
YOLOv8 Detection Module for Legal Metrology AI.
Identifies important regions on packaged commodity images:
- Container / package object detection via Ultralytics YOLOv8 (loaded from models/)
- Packaging panel candidate extraction using contour/edge density heuristics when using pretrained COCO weights
- Clean pluggable architecture for custom trained Legal Metrology declaration models
- Transparent reporting of class names and detection methods (does NOT falsely attribute COCO weights to specific statutory declarations)
- Saves annotated visual outputs to outputs/detections/
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import cv2
import numpy as np

logger = logging.getLogger("LegalMetrologyAI.YOLO")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class BaseDetector:
    """Abstract interface for packaging region detectors."""

    def detect(self, image: np.ndarray, image_stem: str = "image") -> List[Dict[str, Any]]:
        raise NotImplementedError


class HeuristicPackagingPanelDetector(BaseDetector):
    """
    Fallback prototype panel detector using edge density, aspect ratio,
    and rectangular contour analysis.
    Used when the loaded YOLO model does not have custom declaration classes.
    """

    def detect(self, image: np.ndarray, image_stem: str = "image") -> List[Dict[str, Any]]:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Morphological operations to locate text/declaration panel blocks
        grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(grad_x, grad_y)
        gradient = cv2.convertScaleAbs(gradient)

        blurred = cv2.blur(gradient, (9, 9))
        _, thresh = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations=2)
        closed = cv2.dilate(closed, None, iterations=2)

        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        panel_candidates = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            # Filter panels by reasonable packaging proportions
            if area > (w * h * 0.015) and bw > 60 and bh > 30 and bw < (w * 0.95) and bh < (h * 0.95):
                aspect_ratio = bw / float(bh)
                # Label candidate type transparently based on geometry
                if aspect_ratio < 0.7:
                    class_name = "vertical_declaration_panel_candidate"
                elif 0.7 <= aspect_ratio <= 2.5:
                    class_name = "declaration_panel_candidate"
                else:
                    class_name = "horizontal_declaration_strip_candidate"

                panel_candidates.append({
                    "class_name": class_name,
                    "confidence": round(float(min(0.92, 0.70 + (area / (w * h)) * 0.5)), 2),
                    "bounding_box": [float(x), float(y), float(x + bw), float(y + bh)],
                    "image_coordinates": {
                        "x1": float(x),
                        "y1": float(y),
                        "x2": float(x + bw),
                        "y2": float(y + bh),
                        "width": float(bw),
                        "height": float(bh),
                    },
                    "detection_method": "heuristic_packaging_panel",
                })

        # Sort by area descending and cap to top candidates
        panel_candidates.sort(key=lambda r: r["image_coordinates"]["width"] * r["image_coordinates"]["height"], reverse=True)
        return panel_candidates[:6]


class YOLODetector:
    """
    Ultralytics YOLOv8 detector for packaging images.
    Loads models from models/ directory.
    Transparently distinguishes between COCO container classes and custom declaration classes.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.25,
        enable_panel_heuristics: bool = True,
    ):
        if model_path is None:
            # Default model path in models/
            project_root = Path(__file__).resolve().parent.parent.parent
            model_path = project_root / "models" / "yolov8n.pt"

        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.enable_panel_heuristics = enable_panel_heuristics
        self.model = None
        self.is_custom_trained = False
        self.classes: Dict[int, str] = {}
        self.heuristic_detector = HeuristicPackagingPanelDetector()

        self._load_model()

    def _load_model(self):
        """Loads YOLOv8 weights from models/ directory."""
        from ultralytics import YOLO

        if not self.model_path.exists():
            # If yolov8n.pt is not yet in models/, download directly to models/
            logger.info(f"[YOLO] Checkpoint not found at {self.model_path}. Initializing default in models/...")
            self.model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.model = YOLO(str(self.model_path))
            self.classes = self.model.names if hasattr(self.model, "names") else {}

            # Check if this model contains specific legal metrology declaration classes
            declaration_keywords = {"mrp", "declaration", "net_quantity", "label", "nutrition", "expiry", "panel"}
            model_class_set = {str(name).lower() for name in self.classes.values()}
            if any(any(kw in c for kw in declaration_keywords) for c in model_class_set):
                self.is_custom_trained = True
                logger.info(f"[YOLO] Custom trained declaration model detected: {self.model_path.name}")
            else:
                self.is_custom_trained = False
                logger.info(
                    f"[YOLO] Pretrained model loaded ({self.model_path.name}). Standard classes: {list(self.classes.values())[:5]}..."
                )
        except Exception as e:
            logger.error(f"[YOLO] Error loading model from {self.model_path}: {e}")
            raise

    def detect_regions(
        self,
        image_input: Union[str, Path, np.ndarray],
        conf: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs region detection on product packaging image.
        Returns list of detections with:
        - class_name
        - confidence
        - bounding_box [x1, y1, x2, y2]
        - image_coordinates {x1, y1, x2, y2, width, height}
        """
        # 1. Load image as numpy array
        if isinstance(image_input, (str, Path)):
            p = Path(image_input).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")
            file_bytes = np.fromfile(str(p), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            stem = p.stem
        elif isinstance(image_input, np.ndarray):
            image = image_input
            stem = "image"
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        if image is None or image.size == 0:
            raise ValueError("Invalid or empty image provided for detection")

        conf_threshold = conf if conf is not None else self.conf_threshold

        # 2. Run Ultralytics YOLOv8 inference
        yolo_results = self.model(image, conf=conf_threshold, verbose=False)
        detected_regions: List[Dict[str, Any]] = []

        if yolo_results and len(yolo_results) > 0:
            boxes = yolo_results[0].boxes
            for b in boxes:
                cls_id = int(b.cls[0])
                cls_name = self.classes.get(cls_id, f"class_{cls_id}")
                score = float(b.conf[0])
                xyxy = b.xyxy[0].tolist()  # [x1, y1, x2, y2]

                x1, y1, x2, y2 = xyxy
                detected_regions.append({
                    "class_name": cls_name,
                    "confidence": round(score, 4),
                    "bounding_box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "image_coordinates": {
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1),
                        "width": round(x2 - x1, 1),
                        "height": round(y2 - y1, 1),
                    },
                    "detection_method": "yolov8_custom" if self.is_custom_trained else "yolov8_pretrained_coco",
                })

        # 3. If using standard pretrained COCO weights and panel heuristics are enabled:
        # Extract prototype packaging declaration panels using heuristic detector
        if not self.is_custom_trained and self.enable_panel_heuristics:
            heuristic_panels = self.heuristic_detector.detect(image, image_stem=stem)
            detected_regions.extend(heuristic_panels)

        return detected_regions


def annotate_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> str:
    """
    Draws detected bounding boxes, classes, and confidence on the image.
    Saves the annotated image to output_path.
    """
    annotated = image.copy()
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Color palette for different detection types
    color_coco = (255, 120, 0)      # Blue-Orange for COCO
    color_panel = (0, 180, 255)     # Amber-Yellow for Panels

    for det in detections:
        box = det["bounding_box"]
        x1, y1, x2, y2 = [int(v) for v in box]
        cls_name = det["class_name"]
        conf = det["confidence"]
        method = det.get("detection_method", "")

        color = color_panel if "panel" in cls_name or "heuristic" in method else color_coco

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label tag
        label = f"{cls_name} ({conf:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = max(y1 - 5, text_h + 4)
        cv2.rectangle(
            annotated,
            (x1, label_y - text_h - 3),
            (x1 + text_w + 4, label_y + baseline - 2),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, label_y - 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )

    success, encoded = cv2.imencode(".jpg", annotated)
    if success:
        with open(out_p, "wb") as f:
            f.write(encoded.tobytes())

    return str(out_p)


def detect_regions(
    image_path: Union[str, Path, np.ndarray],
    model_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    conf_threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    Main detection entrypoint:
    1. Loads image
    2. Runs YOLOv8 and panel region detector
    3. Annotates detections and saves to outputs/detections/
    4. Returns structured JSON-compatible result
    """
    if isinstance(image_path, (str, Path)):
        img_p = Path(image_path).resolve()
        if not img_p.exists():
            raise FileNotFoundError(f"Image not found at: {img_p}")
        file_bytes = np.fromfile(str(img_p), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        stem = img_p.stem
        src_path_str = str(img_p)
    elif isinstance(image_path, np.ndarray):
        image = image_path
        stem = "image_input"
        src_path_str = "numpy_array"
    else:
        raise ValueError(f"Unsupported image input type: {type(image_path)}")

    detector = YOLODetector(model_path=model_path, conf_threshold=conf_threshold)
    regions = detector.detect_regions(image)

    # Output directory
    if output_dir is None:
        if isinstance(image_path, (str, Path)):
            target_dir = Path(image_path).resolve().parent.parent / "outputs" / "detections"
        else:
            target_dir = Path("outputs/detections").resolve()
    else:
        target_dir = Path(output_dir).resolve()

    annotated_file = target_dir / f"{stem}_detected.jpg"
    annotated_path_str = annotate_detections(image, regions, annotated_file)

    return {
        "status": "success",
        "image_path": src_path_str,
        "annotated_image_path": annotated_path_str,
        "model_loaded": str(detector.model_path),
        "is_custom_trained": detector.is_custom_trained,
        "total_regions": len(regions),
        "detections": regions,
    }
