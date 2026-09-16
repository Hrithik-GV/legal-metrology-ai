"""
Main Computer Vision Pipeline for Legal Metrology AI.

Pipeline Orchestration:
Input image
    ↓
OpenCV Preprocessing (contrast enhancement, bilateral denoising, aspect ratio scaling)
    ↓
YOLO Region Detection (container, label, and packaging candidate panels)
    ↓
PaddleOCR Text Detection (text recognition with polygonal bounding boxes and confidence)
    ↓
Spatial Association (assigns OCR tokens to detected YOLO regions)
    ↓
Unified Visual Annotation (composite YOLO boxes + OCR boxes + labels + confidences)
    ↓
Return unified JSON-compatible result

Can be run via CLI:
python -m app.pipeline.pipeline test_images/packaged_commodity_sample.jpg
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console UTF-8 output encoding
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from app.preprocessing.image_processor import preprocess_image, PreprocessResult
from app.detection.yolo_detector import YOLODetector, detect_regions
from app.ocr.paddle_ocr import extract_text
from app.extraction.declaration_extractor import extract_declarations


def get_box_center_and_bounds(box: List[Any]) -> Tuple[float, float, float, float, float, float]:
    """
    Computes (center_x, center_y, x_min, y_min, x_max, y_max)
    from a 4-point polygon or [x1, y1, x2, y2] format.
    """
    if len(box) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in box):
        xs = [float(pt[0]) for pt in box]
        ys = [float(pt[1]) for pt in box]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
    elif len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
        x_min, y_min, x_max, y_max = [float(v) for v in box]
    else:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return cx, cy, x_min, y_min, x_max, y_max


def calculate_overlap_ratio(ocr_box: List[Any], region_box: List[Any]) -> float:
    """
    Computes intersection area / ocr_box area to evaluate if an OCR snippet
    falls substantially inside a YOLO detected region.
    """
    _, _, ox1, oy1, ox2, oy2 = get_box_center_and_bounds(ocr_box)
    _, _, rx1, ry1, rx2, ry2 = get_box_center_and_bounds(region_box)

    ocr_area = max(1.0, (ox2 - ox1) * (oy2 - oy1))

    ix1 = max(ox1, rx1)
    iy1 = max(oy1, ry1)
    ix2 = min(ox2, rx2)
    iy2 = min(oy2, ry2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    return intersection / ocr_area


def associate_ocr_with_regions(
    detections: List[Dict[str, Any]], ocr_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Associates OCR text items with corresponding YOLO detected regions
    based on spatial inclusion or bounding box intersection.
    """
    associated_regions: List[Dict[str, Any]] = []
    assigned_ocr_indices = set()

    for idx, det in enumerate(detections, 1):
        r_box = det["bounding_box"]
        _, _, rx1, ry1, rx2, ry2 = get_box_center_and_bounds(r_box)

        matched_tokens = []
        for o_idx, ocr_item in enumerate(ocr_results):
            o_box = ocr_item.get("bounding_box", [])
            cx, cy, ox1, oy1, ox2, oy2 = get_box_center_and_bounds(o_box)

            # Check 1: Token center is inside the region
            center_inside = (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)

            # Check 2: Token overlaps at least 35% with the region
            overlap = calculate_overlap_ratio(o_box, r_box)

            if center_inside or overlap >= 0.35:
                matched_tokens.append(ocr_item)
                assigned_ocr_indices.add(o_idx)

        associated_regions.append({
            "region_id": idx,
            "class_name": det.get("class_name", "unknown_region"),
            "confidence": det.get("confidence", 0.0),
            "bounding_box": det.get("bounding_box", []),
            "image_coordinates": det.get("image_coordinates", {}),
            "detection_method": det.get("detection_method", "yolo"),
            "ocr_text_count": len(matched_tokens),
            "ocr_results": matched_tokens,
            "concatenated_text": " | ".join(t.get("text", "") for t in matched_tokens),
        })

    # Collect unassigned OCR items (e.g. text outside detected sub-panels)
    unassigned_tokens = [
        ocr_item for o_idx, ocr_item in enumerate(ocr_results)
        if o_idx not in assigned_ocr_indices
    ]

    if unassigned_tokens:
        associated_regions.append({
            "region_id": len(detections) + 1,
            "class_name": "general_packaging_text",
            "confidence": 1.0,
            "bounding_box": [0.0, 0.0, 0.0, 0.0],
            "image_coordinates": {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0, "width": 0.0, "height": 0.0},
            "detection_method": "unassigned_ocr_tokens",
            "ocr_text_count": len(unassigned_tokens),
            "ocr_results": unassigned_tokens,
            "concatenated_text": " | ".join(t.get("text", "") for t in unassigned_tokens),
        })

    return associated_regions


def create_composite_annotated_image(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    ocr_results: List[Dict[str, Any]],
    output_path: Union[str, Path],
) -> str:
    """
    Renders an annotated final image showing:
    - YOLO region boxes (Orange/Cyan with class & confidence tag)
    - OCR recognized text boxes (Vibrant Green polygons with text label)
    - Header banner identifying Legal Metrology Vision AI layers
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Draw YOLO Detection Regions (Orange / Amber)
    color_yolo = (0, 140, 255)  # BGR Orange
    for det in detections:
        box = det.get("bounding_box", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cls_name = det.get("class_name", "region")
        conf = det.get("confidence", 0.0)

        # Thick boundary for YOLO region
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_yolo, 3)

        label = f"YOLO: {cls_name} ({conf:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        label_y = max(y1 - 6, th + 6)
        cv2.rectangle(annotated, (x1, label_y - th - 3), (x1 + tw + 6, label_y + baseline - 1), color_yolo, -1)
        cv2.putText(annotated, label, (x1 + 3, label_y - 2), font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)

    # 2. Draw OCR Text Polygons & Labels (Vibrant Green)
    color_ocr = (0, 220, 0)
    for ocr in ocr_results:
        box = ocr.get("bounding_box", [])
        text = ocr.get("text", "")
        conf = ocr.get("confidence", 0.0)

        if not box:
            continue

        pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=color_ocr, thickness=2)

        x0, y0 = int(pts[0][0][0]), int(pts[0][0][1])
        ocr_label = f"{text[:22]} ({conf:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(ocr_label, font, font_scale, thickness)

        label_y = max(y0 - 5, th + 4)
        cv2.rectangle(annotated, (x0, label_y - th - 2), (x0 + tw + 4, label_y + baseline - 1), (0, 150, 0), -1)
        cv2.putText(annotated, ocr_label, (x0 + 2, label_y - 2), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    # 3. Top Status Legend Banner
    banner_h = 36
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

    legend_text = f"Legal Metrology AI  |  YOLO Regions: {len(detections)} (Orange)  |  OCR Tokens: {len(ocr_results)} (Green)"
    cv2.putText(annotated, legend_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    success, encoded = cv2.imencode(".jpg", annotated)
    if success:
        with open(out_p, "wb") as f:
            f.write(encoded.tobytes())

    return str(out_p)


class VisionPipeline:
    """
    Unified Computer Vision Engine coordinating:
    OpenCV Preprocessing -> YOLOv8 Detection -> PaddleOCR -> Region Association.
    """

    def __init__(
        self,
        yolo_model_path: Optional[Union[str, Path]] = None,
        output_base_dir: Optional[Union[str, Path]] = None,
    ):
        self.output_base_dir = Path(output_base_dir or PROJECT_ROOT / "outputs")
        self.yolo_detector = YOLODetector(model_path=yolo_model_path)

    def process(
        self,
        image_path: Union[str, Path],
        to_grayscale: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end vision pipeline:
        1. Preprocesses image
        2. Detects packaging regions with YOLOv8
        3. Extracts text and bounding boxes with PaddleOCR
        4. Associates OCR tokens with detected regions
        5. Saves final annotated visualization
        """
        input_path = Path(image_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Image not found at {input_path}")

        stem = input_path.stem

        # Step 1: OpenCV Preprocessing
        prep_dir = self.output_base_dir / "preprocessed"
        prep_result: PreprocessResult = preprocess_image(
            image_path=input_path,
            output_dir=prep_dir,
            to_grayscale=to_grayscale,
        )
        preprocessed_img = prep_result.processed_image

        # Step 2: YOLOv8 Region Detection
        det_dir = self.output_base_dir / "detections"
        detection_response = detect_regions(
            image_path=preprocessed_img,
            model_path=self.yolo_detector.model_path,
            output_dir=det_dir,
        )
        detections: List[Dict[str, Any]] = detection_response.get("detections", [])

        # Step 3: PaddleOCR Text Detection
        ocr_dir = self.output_base_dir / "ocr"
        ocr_response = extract_text(
            image_path=prep_result.output_path,
            output_dir=ocr_dir,
            image_stem=stem,
        )
        ocr_results: List[Dict[str, Any]] = ocr_response.get("results", [])

        # Step 4: Associate OCR text with detected YOLO regions
        associated_regions = associate_ocr_with_regions(detections, ocr_results)

        # Step 5: Statutory Declarations (Bonus extracted declarations)
        extracted_declarations = extract_declarations(ocr_results, image_id=stem)

        # Step 6: Create Composite Final Annotated Image
        final_dir = self.output_base_dir / "final"
        final_output_file = final_dir / f"{stem}_final.jpg"
        final_annotated_path = create_composite_annotated_image(
            image=preprocessed_img,
            detections=detections,
            ocr_results=ocr_results,
            output_path=final_output_file,
        )

        return {
            "image": str(input_path),
            "annotated_image": final_annotated_path,
            "preprocessed_image": prep_result.output_path,
            "dimensions": {
                "original": list(prep_result.original_dimensions),
                "processed": list(prep_result.processed_dimensions),
            },
            "detections": detections,
            "ocr_results": ocr_results,
            "regions": associated_regions,
            "declarations": extracted_declarations,
            "summary": {
                "total_detections": len(detections),
                "total_ocr_tokens": len(ocr_results),
                "total_associated_regions": len(associated_regions),
                "declarations_found": len(extracted_declarations),
            },
        }


def run_pipeline(
    image_path: Union[str, Path],
    output_base_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Top-level convenience entrypoint."""
    pipeline = VisionPipeline(output_base_dir=output_base_dir)
    return pipeline.process(image_path=image_path)


def main():
    parser = argparse.ArgumentParser(
        description="Legal Metrology AI: Unified Computer Vision Pipeline"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=str(PROJECT_ROOT / "test_images" / "packaged_commodity_sample.jpg"),
        help="Path to product package image (default: sample in test_images/)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(PROJECT_ROOT / "outputs"),
        help="Base output directory for artifacts (default: outputs/)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON output instead of formatted report",
    )

    args = parser.parse_args()
    img_path = Path(args.image)

    if not img_path.exists():
        print(f"[ERROR] Image file does not exist: {img_path}")
        sys.exit(1)

    print("=" * 70)
    print("LEGAL METROLOGY AI: VISION PIPELINE EXECUTION")
    print("=" * 70)
    print(f"Target Image: {img_path.resolve()}")

    try:
        result = run_pipeline(image_path=img_path, output_base_dir=args.output_dir)

        if args.json:
            print(json.dumps(result, indent=2))
            return

        print("\nPipeline Stages Completed Successfully:")
        print(f" 1. OpenCV Preprocessing : {result['preprocessed_image']}")
        print(f" 2. YOLOv8 Detections    : {len(result['detections'])} regions identified")
        print(f" 3. PaddleOCR Extraction : {len(result['ocr_results'])} text tokens detected")
        print(f" 4. Region Association   : {len(result['regions'])} mapped regions")
        print(f" 5. Declarations Parsed  : {len(result['declarations'])} statutory fields")
        print(f"\n[FINAL VISUALIZATION SAVED]:\n -> {result['annotated_image']}")

        print("\n--- Associated Regions Breakdown ---")
        for reg in result["regions"]:
            cls_name = reg["class_name"]
            tok_count = reg["ocr_text_count"]
            conf = reg["confidence"]
            print(f"  * Region {reg['region_id']}: {cls_name} (conf: {conf:.2f}) -> {tok_count} OCR texts")
            if reg["concatenated_text"]:
                print(f"      Text: {reg['concatenated_text'][:80]}...")

        print("\n--- Key Legal Metrology Declarations ---")
        for dec in result["declarations"]:
            print(f"  * {dec['field']:18}: {dec['value']} (conf: {dec['confidence']})")

        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION COMPLETED (STATUS: OK)")
        print("=" * 70)

    except Exception as e:
        print(f"\n[FATAL PIPELINE ERROR]: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
