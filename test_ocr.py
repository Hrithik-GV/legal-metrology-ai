"""
Test script for the Legal Metrology AI PaddleOCR Module.
Evaluates text extraction, confidence scoring, bounding box generation, and annotation on multiple packaged commodity test images.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ocr.paddle_ocr import extract_text


def run_ocr_tests():
    test_images_dir = PROJECT_ROOT / "test_images"
    output_dir = PROJECT_ROOT / "outputs" / "ocr"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_images = sorted([
        f for f in test_images_dir.glob("*.*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if not test_images:
        print(f"[ERROR] No test images found in {test_images_dir}")
        return 1

    print("=" * 70)
    print("LEGAL METROLOGY AI - OCR EXTRACTION TEST (MULTIPLE COMMODITIES)")
    print("=" * 70)
    print(f"Total test images detected: {len(test_images)}")

    all_passed = True

    for idx, img_path in enumerate(test_images, 1):
        print(f"\n[{idx}/{len(test_images)}] Testing Image: {img_path.name}")
        try:
            result = extract_text(
                image_path=img_path,
                output_dir=output_dir
            )

            # 1. Verify top-level structure
            engine_used = result.get("engine")
            annotated_path = result.get("annotated_image_path")
            regions = result.get("results", [])

            print(f"   - Engine Used          : {engine_used.upper()}")
            print(f"   - Regions Detected     : {len(regions)}")
            print(f"   - Annotated Image Path : {annotated_path}")

            # 2. Check annotated file exists on disk
            if Path(annotated_path).exists():
                file_size = Path(annotated_path).stat().st_size
                print(f"   - Annotated File OK    : {file_size} bytes")
            else:
                print(f"   - [FAIL] Annotated file not found at {annotated_path}")
                all_passed = False

            # 3. Validate every region matches required structure:
            # { "text": "...", "confidence": 0.95, "bounding_box": [...] }
            print("   - Sample Extracted Declarations:")
            for i, reg in enumerate(regions[:6], 1):
                assert "text" in reg, "Missing 'text' key"
                assert "confidence" in reg, "Missing 'confidence' key"
                assert "bounding_box" in reg, "Missing 'bounding_box' key"
                print(f"       [{i}] \"{reg['text']}\" (conf: {reg['confidence']:.2f}) -> box: {reg['bounding_box'][:2]}...")

            # 4. Confirm JSON-compatibility
            json_dump = json.dumps(result, indent=2)
            assert len(json_dump) > 0

        except Exception as e:
            print(f"   - [ERROR] OCR failed for {img_path.name}: {e}")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print(" [PASSED] All OCR tests completed successfully across all test images!")
    else:
        print(" [FAILED] Some OCR tests encountered errors.")
    print("=" * 70)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_ocr_tests())
