"""
Unit Tests for Legal Metrology AI Modular Rule Engine.
Tests:
1. Fully populated product (All representative rules PASS)
2. Missing MRP (POTENTIAL_VIOLATION on LM005)
3. Missing consumer care (POTENTIAL_VIOLATION on LM007)
4. Missing multiple declarations (Multiple POTENTIAL_VIOLATIONs, degraded score)
5. Low-confidence OCR (WARNING on low-confidence tokens)
6. Non-verifiable ambiguous declaration handling (NOT_VERIFIABLE)
7. Statutory Phrasing Guard: Ensures 'AI-Assisted Preliminary Assessment' and
   'Manual verification required.' are present, and never claims legal compliance/illegality.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console UTF-8 output encoding
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from app.rules.rule_engine import (
    RuleEngine,
    evaluate_declarations,
    ASSESSMENT_LABEL,
    STATUTORY_DISCLAIMER,
)


class TestRuleEngine(unittest.TestCase):

    def setUp(self):
        self.engine = RuleEngine(confidence_threshold=0.75)

        # Baseline fully-populated declaration set
        self.fully_populated = [
            {"field": "product_name", "value": "Harvest Grains Ragi Flakes", "confidence": 0.98},
            {"field": "manufacturer", "value": "Harvest Foods Pvt. Ltd.", "confidence": 0.97},
            {"field": "address", "value": "Plot 45, MIDC, Andheri East, Mumbai 400093", "confidence": 0.95},
            {"field": "net_quantity", "value": "500 g", "confidence": 0.99},
            {"field": "mrp", "value": "₹120.00", "confidence": 0.98},
            {"field": "manufactured_date", "value": "15/08/2026", "confidence": 0.97},
            {"field": "consumer_care", "value": "1800-11-4000, care@harvestfoods.in", "confidence": 0.96},
        ]

    def test_1_fully_populated_product(self):
        """1. Fully populated product: all representative rules PASS, score is 100%."""
        result = self.engine.evaluate(self.fully_populated)

        self.assertEqual(result["assessment"], "AI-Assisted Preliminary Assessment")
        self.assertIn("Manual verification required", result["disclaimer"])
        self.assertEqual(result["compliance_score"], 100.0)
        self.assertEqual(result["summary"]["pass_count"], 7)
        self.assertEqual(result["summary"]["potential_violations"], 0)
        self.assertEqual(result["summary"]["warning_count"], 0)

        for r in result["rule_results"]:
            self.assertEqual(r["status"], "PASS")
            self.assertIn("rule_id", r)
            self.assertIn("field", r)
            self.assertIn("reason", r)
            self.assertIn("confidence", r)

    def test_2_missing_mrp(self):
        """2. Missing MRP: LM005 returns POTENTIAL_VIOLATION."""
        decs_no_mrp = [d for d in self.fully_populated if d["field"] != "mrp"]
        result = self.engine.evaluate(decs_no_mrp)

        mrp_rule = next(r for r in result["rule_results"] if r["rule_id"] == "LM005")
        self.assertEqual(mrp_rule["status"], "POTENTIAL_VIOLATION")
        self.assertEqual(mrp_rule["field"], "mrp")
        self.assertIn("Maximum Retail Price", mrp_rule["reason"])
        self.assertEqual(mrp_rule["confidence"], 0.0)

        # Score drops
        self.assertLess(result["compliance_score"], 100.0)
        self.assertEqual(result["summary"]["potential_violations"], 1)

    def test_3_missing_consumer_care(self):
        """3. Missing Consumer Care: LM007 returns POTENTIAL_VIOLATION."""
        decs_no_care = [d for d in self.fully_populated if d["field"] != "consumer_care"]
        result = self.engine.evaluate(decs_no_care)

        care_rule = next(r for r in result["rule_results"] if r["rule_id"] == "LM007")
        self.assertEqual(care_rule["status"], "POTENTIAL_VIOLATION")
        self.assertEqual(care_rule["field"], "consumer_care")
        self.assertIn("Consumer Care", care_rule["reason"])
        self.assertEqual(result["summary"]["potential_violations"], 1)

    def test_4_missing_multiple_declarations(self):
        """4. Missing multiple declarations: multiple POTENTIAL_VIOLATIONs, significantly lower score."""
        sparse_decs = [
            {"field": "product_name", "value": "Sample Biscuit", "confidence": 0.90},
            {"field": "net_quantity", "value": "100 g", "confidence": 0.95},
        ]
        result = self.engine.evaluate(sparse_decs)

        self.assertEqual(result["summary"]["pass_count"], 2)
        self.assertEqual(result["summary"]["potential_violations"], 5)
        # Score is (2 / 7) * 100 = 28.6%
        self.assertAlmostEqual(result["compliance_score"], 28.6, places=1)

    def test_5_low_confidence_ocr(self):
        """5. Low-confidence OCR: Returns WARNING rather than PASS or VIOLATION."""
        low_conf_decs = [
            {"field": "product_name", "value": "Harvest Grains", "confidence": 0.98},
            {"field": "manufacturer", "value": "Harvest Foods", "confidence": 0.95},
            {"field": "address", "value": "Mumbai 400093", "confidence": 0.95},
            {"field": "net_quantity", "value": "500 g", "confidence": 0.95},
            {"field": "mrp", "value": "₹120.00", "confidence": 0.45},  # Low confidence MRP (< 0.75)
            {"field": "manufactured_date", "value": "15/08/2026", "confidence": 0.95},
            {"field": "consumer_care", "value": "1800-11-4000", "confidence": 0.95},
        ]
        result = self.engine.evaluate(low_conf_decs)

        mrp_rule = next(r for r in result["rule_results"] if r["rule_id"] == "LM005")
        self.assertEqual(mrp_rule["status"], "WARNING")
        self.assertIn("low OCR confidence", mrp_rule["reason"])
        self.assertEqual(result["summary"]["warning_count"], 1)
        self.assertEqual(result["summary"]["potential_violations"], 0)

    def test_6_not_verifiable_status(self):
        """6. Ambiguous / corrupted field marked NOT_VERIFIABLE."""
        ambiguous_decs = [
            {"field": "product_name", "value": "Harvest Grains", "confidence": 0.98},
            {"field": "manufacturer", "value": "Harvest Foods", "confidence": 0.95},
            {"field": "address", "value": "Mumbai 400093", "confidence": 0.95},
            {"field": "net_quantity", "value": "unverifiable", "confidence": 0.85},  # Explicitly ambiguous
            {"field": "mrp", "value": "₹120.00", "confidence": 0.95},
            {"field": "manufactured_date", "value": "15/08/2026", "confidence": 0.95},
            {"field": "consumer_care", "value": "1800-11-4000", "confidence": 0.95},
        ]
        result = self.engine.evaluate(ambiguous_decs)

        qty_rule = next(r for r in result["rule_results"] if r["rule_id"] == "LM004")
        self.assertEqual(qty_rule["status"], "NOT_VERIFIABLE")
        self.assertEqual(result["summary"]["not_verifiable"], 1)

    def test_7_statutory_phrasing_guard(self):
        """Ensure assessment is preliminary and disclaimers are strictly enforced."""
        result = self.engine.evaluate(self.fully_populated)
        dump = str(result)

        # Mandatory positive phrasing
        self.assertIn("AI-Assisted Preliminary Assessment", dump)
        self.assertIn("Manual verification required", dump)

        # Prohibited absolutes
        self.assertNotIn("the product is legally compliant", dump.lower())
        self.assertNotIn("the product is illegal", dump.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
