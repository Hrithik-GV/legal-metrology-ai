"""
Unit Tests for Legal Metrology AI Declaration Extractor.
Tests:
- MRP variations (MRP, M.R.P., Maximum Retail Price, Rs, ₹, INR)
- Net Quantity variations (Net Qty, Net Quantity, Net Wt, Net Weight, kg, g, ml, L)
- Dates (PKD, Mfg Date, DD/MM/YYYY, MM/YYYY, Month YYYY)
- Consumer Care (Toll free numbers, emails)
- Country of origin
- Manufacturer / Packer / Importer / Address
- Spatial proximity multiline token resolution
- Modular LLM backend swapping
- End-to-end integration with OCR test images
"""

import sys
import unittest
from pathlib import Path

# Fix Windows console UTF-8 output encoding for symbols like ₹
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.extraction.declaration_extractor import (
    DeclarationExtractor,
    DeterministicDeclarationExtractor,
    BaseDeclarationExtractor,
    extract_declarations,
)
from app.ocr.paddle_ocr import extract_text


class TestDeclarationExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = DeclarationExtractor()

    def test_mrp_variations(self):
        """Test variations of MRP keyword and currency representation."""
        test_cases = [
            ("MRP ₹450", "₹450"),
            ("M.R.P.: Rs 120.00 (inclusive of all taxes)", "₹120.00"),
            ("Maximum Retail Price: INR 99.50/-", "₹99.50"),
            ("Retail Price Rs. 185.00", "₹185.00"),
        ]

        for text, expected_val in test_cases:
            tokens = [{"text": text, "confidence": 0.95, "bounding_box": [[10, 10], [100, 10], [100, 30], [10, 30]]}]
            results = self.extractor.extract_as_dict(tokens, image_id="test_mrp")
            self.assertIn("mrp", results, f"Failed to extract MRP from: '{text}'")
            self.assertEqual(results["mrp"]["value"], expected_val)
            self.assertEqual(results["mrp"]["field"], "mrp")
            self.assertIn("bounding_box", results["mrp"])
            self.assertEqual(results["mrp"]["image_id"], "test_mrp")

    def test_mrp_split_tokens_spatial_proximity(self):
        """Test MRP label in one box and price in adjacent box."""
        tokens = [
            {"text": "MRP (Incl. of all taxes):", "confidence": 0.96, "bounding_box": [[100, 200], [250, 200], [250, 230], [100, 230]]},
            {"text": "Rs. 350.00", "confidence": 0.94, "bounding_box": [[260, 200], [350, 200], [350, 230], [260, 230]]},
        ]
        results = self.extractor.extract_as_dict(tokens, image_id="spatial_mrp")
        self.assertIn("mrp", results)
        self.assertEqual(results["mrp"]["value"], "₹350.00")
        self.assertIn("MRP", results["mrp"]["source_text"])
        self.assertIn("350", results["mrp"]["source_text"])

    def test_net_quantity_variations(self):
        """Test Net Quantity variations and unit normalizations."""
        test_cases = [
            ("Net Qty: 500g", "500 g"),
            ("Net Quantity: 1 Litre", "1 L"),
            ("Net Wt. 2.5 kg", "2.5 kg"),
            ("Net Weight: 750 ml", "750 ml"),
            ("Net Content: 100 gm", "100 g"),
            ("Quantity: 1 L", "1 L"),
        ]

        for text, expected_val in test_cases:
            tokens = [{"text": text, "confidence": 0.96, "bounding_box": [[10, 50], [150, 50], [150, 75], [10, 75]]}]
            results = self.extractor.extract_as_dict(tokens, image_id="test_qty")
            self.assertIn("net_quantity", results, f"Failed to extract Net Quantity from: '{text}'")
            self.assertEqual(results["net_quantity"]["value"], expected_val)

    def test_dates_extraction(self):
        """Test PKD (packed date) and Mfg Date extraction."""
        tokens = [
            {"text": "PKD: 05/2026", "confidence": 0.98, "bounding_box": [[10, 100], [120, 100], [120, 125], [10, 125]]},
            {"text": "Mfg. Date: 15/08/2026", "confidence": 0.97, "bounding_box": [[10, 130], [180, 130], [180, 155], [10, 155]]},
        ]
        results = self.extractor.extract_as_dict(tokens, image_id="test_dates")
        self.assertIn("packed_date", results)
        self.assertEqual(results["packed_date"]["value"], "05/2026")

        self.assertIn("manufactured_date", results)
        self.assertEqual(results["manufactured_date"]["value"], "15/08/2026")

    def test_consumer_care_extraction(self):
        """Test extraction of toll-free numbers and consumer emails."""
        tokens = [
            {"text": "CONSUMER CARE: Toll-Free 1800-11-4000", "confidence": 0.97, "bounding_box": [[20, 200], [300, 200], [300, 225], [20, 225]]},
            {"text": "Email: care@harvestgrains.in", "confidence": 0.95, "bounding_box": [[20, 230], [250, 230], [250, 255], [20, 255]]},
        ]
        results = self.extractor.extract_as_dict(tokens, image_id="test_care")
        self.assertIn("consumer_care", results)
        self.assertIn("1800-11-4000", results["consumer_care"]["value"])

    def test_country_of_origin_extraction(self):
        """Test Country of Origin patterns."""
        tokens = [
            {"text": "Country of Origin: India", "confidence": 0.99, "bounding_box": [[50, 300], [200, 300], [200, 320], [50, 320]]}
        ]
        results = self.extractor.extract_as_dict(tokens, image_id="test_origin")
        self.assertIn("country_of_origin", results)
        self.assertEqual(results["country_of_origin"]["value"], "India")

    def test_manufacturer_and_address(self):
        """Test Manufacturer and Address with Pincode."""
        tokens = [
            {"text": "MANUFACTURED BY: SUNSHINE FOODS PVT. LTD.", "confidence": 0.98, "bounding_box": [[30, 400], [400, 400], [400, 425], [30, 425]]},
            {"text": "PLOT NO. 45, MIDC, ANDHERI (EAST), MUMBAI - 400093", "confidence": 0.97, "bounding_box": [[30, 430], [450, 430], [450, 455], [30, 455]]},
        ]
        results = self.extractor.extract_as_dict(tokens, image_id="test_mfg")
        self.assertIn("manufacturer", results)
        self.assertIn("SUNSHINE FOODS", results["manufacturer"]["value"])
        self.assertIn("address", results)
        self.assertIn("400093", results["address"]["value"])

    def test_modular_llm_pluggability(self):
        """Verify that the extractor allows hot-swapping with an LLM backend."""
        class MockLLMExtractor(BaseDeclarationExtractor):
            def extract(self, tokens, image_id="image_0"):
                return [{
                    "field": "mrp",
                    "value": "₹999",
                    "confidence": 0.99,
                    "source_text": "LLM extracted MRP ₹999",
                    "bounding_box": None,
                    "image_id": image_id
                }]

        custom_extractor = DeclarationExtractor(backend_extractor=MockLLMExtractor())
        res = custom_extractor.extract_as_dict([], image_id="llm_test")
        self.assertIn("mrp", res)
        self.assertEqual(res["mrp"]["value"], "₹999")

    def test_e2e_with_ocr_results(self):
        """Test declaration extraction directly on actual OCR output from test images."""
        test_images = list((PROJECT_ROOT / "test_images").glob("*.jpg"))
        self.assertGreaterEqual(len(test_images), 2, "Need at least 2 test images")

        for img_path in test_images:
            ocr_output = extract_text(img_path)
            declarations = self.extractor.extract_as_dict(ocr_output, image_id=img_path.name)

            print(f"\n[E2E EVALUATION] Image: {img_path.name}")
            for field, item in declarations.items():
                print(f"   -> {field:18}: {item['value']} (conf: {item['confidence']})")

            # Check that essential Legal Metrology Rule 6 declarations are extracted
            self.assertIn("mrp", declarations, f"MRP missing for {img_path.name}")
            self.assertIn("net_quantity", declarations, f"Net Quantity missing for {img_path.name}")
            self.assertTrue(
                "manufactured_date" in declarations or "packed_date" in declarations,
                f"Date missing for {img_path.name}"
            )
            self.assertIn("consumer_care", declarations, f"Consumer care missing for {img_path.name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
