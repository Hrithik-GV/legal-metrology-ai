"""
Independent Test Script for Legal Metrology AI YOLOv8 Detector.
Tests:
- Model checkpoint loading from models/
- Region detection interface returning class_name, confidence, bounding_box, and image_coordinates
- Annotated visualization generation in outputs/detections/
- Structured JSON-compatible output
- Multiple packaged commodity evaluations
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows console UTF-8 safety
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from app.detection.yolo_detector import detect_regions, YOLODetector


def run_detector_tests():
    test_images_dir = PROJECT_ROOT / "test_images"
    output_dir = PROJECT_ROOT / "outputs" / "detections"
    models_dir = PROJECT_ROOT / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LEGAL METROLOGY AI - YOLOV8 DETECTOR INDEPENDENT TEST")
    print("=" * 70)

    # 1. Verify model checkpoint in models/
    expected_model = models_dir / "yolov8n.pt"
    print(f"[CHECK] Model checkpoint path: {expected_model}")
    if not expected_model.exists():
        print(f" [ERROR] Model not found in {models_dir}")
        return 1
    print(f" [PASSED] Model checkpoint exists ({expected_model.stat().st_size} bytes)")

    # 2. Test instantiation and architecture inspection
    detector = YOLODetector(model_path=expected_model)
    print(f"[CHECK] Custom declaration model: {detector.is_custom_trained}")
    print(f"[CHECK] Pretrained COCO classes count: {len(detector.classes)}")

    # 3. Test detection on available packaged commodity images
    test_images = sorted([
        f for f in test_images_dir.glob("*.*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if not test_images:
        print(f"[ERROR] No test images found in {test_images_dir}")
        return 1

    all_passed = True

    for idx, img_p in enumerate(test_images, 1):
        print(f"\n[{idx}/{len(test_images)}] Testing Detection on: {img_p.name}")
        try:
            result = detect_regions(
                image_path=img_p,
                output_dir=output_dir
            )

            # Validate top-level response
            assert result["status"] == "success", "Status not success"
            assert "annotated_image_path" in result, "Missing annotated_image_path"
            assert "detections" in result, "Missing detections list"

            annotated_path = Path(result["annotated_image_path"])
            print(f"   - Annotated Output      : {annotated_path.name}")
            print(f"   - Output File Size      : {annotated_path.stat().st_size} bytes")
            print(f"   - Regions Detected      : {len(result['detections'])}")
            print(f"   - Custom Model Flag     : {result['is_custom_trained']}")

            # Validate each detection has required fields
            for i, det in enumerate(result["detections"][:5], 1):
                assert "class_name" in det, "Missing 'class_name'"
                assert "confidence" in det, "Missing 'confidence'"
                assert "bounding_box" in det, "Missing 'bounding_box'"
                assert "image_coordinates" in det, "Missing 'image_coordinates'"

                coords = det["image_coordinates"]
                assert all(k in coords for k in ["x1", "y1", "x2", "y2", "width", "height"]), "Incomplete image_coordinates"

                print(f"       [{i}] {det['class_name']} (conf: {det['confidence']:.2f}) -> {det['bounding_box']} [{det['detection_method']}]")

            # Verify JSON compatibility
            json_dump = json.dumps(result, indent=2)
            assert len(json_dump) > 0, "JSON serialization failed"

        except Exception as e:
            print(f"   - [ERROR] Detection failed for {img_p.name}: {e}")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print(" [PASSED] YOLOv8 detection tests completed successfully!")
    else:
        print(" [FAILED] One or more detection tests encountered errors.")
    print("=" * 70)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_detector_tests())
