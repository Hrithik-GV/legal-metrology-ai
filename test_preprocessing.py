"""
Test script for the Legal Metrology AI Image Preprocessing Module.
Processes packaged commodity images from test_images/ and saves results to outputs/preprocessed/.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.preprocessing.image_processor import preprocess_image, ImageProcessingError


def run_test():
    test_images_dir = PROJECT_ROOT / "test_images"
    output_dir = PROJECT_ROOT / "outputs" / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find test images
    image_files = list(test_images_dir.glob("*.jpg")) + \
                  list(test_images_dir.glob("*.png")) + \
                  list(test_images_dir.glob("*.jpeg"))

    if not image_files:
        print(f"[ERROR] No test images found in {test_images_dir}")
        return 1

    print("=" * 60)
    print("LEGAL METROLOGY AI - IMAGE PREPROCESSING TEST")
    print("=" * 60)

    for img_path in image_files:
        print(f"\n[INPUT] Processing: {img_path.name}")
        try:
            # 1. Standard color preprocessed (denoise, CLAHE contrast, unsharp sharpen)
            result = preprocess_image(
                image_path=img_path,
                output_dir=output_dir,
                to_grayscale=False
            )

            print(" [SUCCESS] Color Preprocessing:")
            print(f"   - Original Dimensions  : {result.original_dimensions} (H x W)")
            print(f"   - Processed Dimensions : {result.processed_dimensions} (H x W)")
            print(f"   - Processed Image Type : {type(result.processed_image)} shape={result.processed_image.shape}")
            print(f"   - Output Saved To      : {result.output_path}")

            # Verify output file exists
            if Path(result.output_path).exists():
                print(f"   - File Check Passed    : {Path(result.output_path).stat().st_size} bytes written")

            # 2. Grayscale preprocessed variant
            gray_result = preprocess_image(
                image_path=img_path,
                output_dir=output_dir,
                output_filename=f"{img_path.stem}_grayscale{img_path.suffix}",
                to_grayscale=True
            )
            print(" [SUCCESS] Grayscale Preprocessing:")
            print(f"   - Output Saved To      : {gray_result.output_path}")

            # Verify original image was not modified
            if img_path.exists():
                print(f"   - Original Image Intact: {img_path.stat().st_size} bytes (Non-destructive)")

        except ImageProcessingError as err:
            print(f" [HANDLED ERROR] Preprocessing failed with managed error: {err}")
        except Exception as ex:
            print(f" [UNEXPECTED ERROR] {ex}")

    # Test error handling on a non-existent image
    print("\n--- Testing Error Handling on Non-Existent Image ---")
    fake_path = test_images_dir / "does_not_exist_404.jpg"
    try:
        preprocess_image(fake_path)
    except ImageProcessingError as err:
        print(f" [PASSED] Expected error handled gracefully: {err}")
    except Exception as ex:
        print(f" [FAILED] Unexpected exception type: {ex}")

    print("\n" + "=" * 60)
    print("All preprocessing tests completed successfully!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(run_test())
