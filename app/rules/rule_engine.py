"""
Modular Rule Engine for Legal Metrology AI.
Evaluates extracted declarations against representative Legal Metrology rules.

Important Disclaimer:
These checks represent prototype checks for preliminary assessment.
They do NOT represent the complete Legal Metrology (Packaged Commodities) Rules, 2011.
Never states 'the product is legally compliant' or 'illegal'.
Uses:
- "AI-Assisted Preliminary Assessment"
- "Manual verification required."
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


# Standard Statutory Phrasings
ASSESSMENT_LABEL = "AI-Assisted Preliminary Assessment"
STATUTORY_DISCLAIMER = (
    "Manual verification required. This prototype assesses representative declarations "
    "and does not constitute a legal certification under the Legal Metrology Act or Rules."
)


class RuleEngine:
    """
    Modular Rule Engine evaluating package declarations against statutory rules.
    """

    def __init__(
        self,
        rules_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.75,
    ):
        if rules_path is None:
            rules_path = Path(__file__).parent / "rules.json"
        self.rules_path = Path(rules_path)
        self.confidence_threshold = confidence_threshold
        self.rules_data = self._load_rules()

    def _load_rules(self) -> Dict[str, Any]:
        """Loads representative rules from JSON file."""
        if not self.rules_path.exists():
            return {"rules": []}
        with open(self.rules_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _normalize_input_declarations(
        self, declarations: Union[List[Dict[str, Any]], Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Normalizes input declarations into a uniform dictionary keyed by field name:
        { "mrp": {"value": "...", "confidence": 0.95, ...} }
        """
        normalized = {}
        if isinstance(declarations, list):
            for item in declarations:
                if isinstance(item, dict) and "field" in item:
                    normalized[item["field"]] = item
        elif isinstance(declarations, dict):
            # Check if dict of field->dict or simple field->value
            for k, v in declarations.items():
                if isinstance(v, dict):
                    normalized[k] = v
                elif isinstance(v, str):
                    normalized[k] = {
                        "field": k,
                        "value": v,
                        "confidence": 0.95,
                        "source_text": v,
                    }
        return normalized

    def evaluate_rule(
        self, rule: Dict[str, Any], declarations: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates a single rule against normalized declarations.

        Returns:
        {
          "rule_id": "LM001",
          "field": "product_name",
          "status": "PASS" | "WARNING" | "POTENTIAL_VIOLATION" | "NOT_VERIFIABLE",
          "reason": "...",
          "confidence": 0.98
        }
        """
        rule_id = rule.get("rule_id", "LM000")
        target_field = rule.get("field", "unknown")
        rule_name = rule.get("name", target_field)
        is_required = rule.get("required", True)

        # Handle alias / fallback / primary fields (e.g. date_declaration -> mfg or pkd)
        candidate_fields = [target_field]
        if "primary_fields" in rule:
            candidate_fields.extend(rule["primary_fields"])
        if "fallback_fields" in rule:
            candidate_fields.extend(rule["fallback_fields"])

        found_item = None
        for f in candidate_fields:
            if f in declarations and declarations[f].get("value"):
                found_item = declarations[f]
                break

        # Case 1: Field is absent or empty
        if not found_item or not found_item.get("value"):
            if is_required:
                status = "POTENTIAL_VIOLATION"
                reason = f"Required declaration '{rule_name}' was not detected."
            else:
                status = "WARNING"
                reason = f"Optional declaration '{rule_name}' was not detected."
            return {
                "rule_id": rule_id,
                "field": target_field,
                "status": status,
                "reason": reason,
                "confidence": 0.0,
            }

        value = str(found_item.get("value", "")).strip()
        confidence = float(found_item.get("confidence", 1.0))

        # Case 2: Field present but explicitly marked unverified or ambiguous
        if found_item.get("is_ambiguous", False) or value.lower() in ["unverifiable", "corrupted", "n/a"]:
            return {
                "rule_id": rule_id,
                "field": target_field,
                "status": "NOT_VERIFIABLE",
                "reason": f"Declaration '{rule_name}' detected ('{value}'), but validity cannot be determined automatically.",
                "confidence": round(confidence, 2),
            }

        # Case 3: Low OCR Confidence
        if confidence < self.confidence_threshold:
            return {
                "rule_id": rule_id,
                "field": target_field,
                "status": "WARNING",
                "reason": (
                    f"Declaration '{rule_name}' detected ('{value}') with low OCR confidence "
                    f"({confidence:.2f} < {self.confidence_threshold:.2f}). Manual verification required."
                ),
                "confidence": round(confidence, 2),
            }

        # Case 4: Field is present and verified with good confidence
        return {
            "rule_id": rule_id,
            "field": target_field,
            "status": "PASS",
            "reason": f"Declaration '{rule_name}' found: '{value}'.",
            "confidence": round(confidence, 2),
        }

    def evaluate(
        self, declarations: Union[List[Dict[str, Any]], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates all representative rules and computes the overall prototype compliance score.
        """
        norm_decs = self._normalize_input_declarations(declarations)
        rules = self.rules_data.get("rules", [])

        rule_results: List[Dict[str, Any]] = []
        pass_count = 0
        warning_count = 0
        violation_count = 0
        unverifiable_count = 0

        total_rules = len(rules)

        for rule in rules:
            res = self.evaluate_rule(rule, norm_decs)
            rule_results.append(res)

            status = res["status"]
            if status == "PASS":
                pass_count += 1
            elif status == "WARNING":
                warning_count += 1
            elif status == "POTENTIAL_VIOLATION":
                violation_count += 1
            elif status == "NOT_VERIFIABLE":
                unverifiable_count += 1

        # Prototype Compliance Score Calculation (0.0 to 100.0)
        # PASS: 1.0 point, WARNING: 0.5 point, NOT_VERIFIABLE: 0.25 point, VIOLATION: 0.0 points
        if total_rules > 0:
            raw_score = (
                (pass_count * 1.0)
                + (warning_count * 0.5)
                + (unverifiable_count * 0.25)
            ) / total_rules
            compliance_score = round(raw_score * 100.0, 1)
        else:
            compliance_score = 0.0

        return {
            "assessment": ASSESSMENT_LABEL,
            "disclaimer": STATUTORY_DISCLAIMER,
            "compliance_score": compliance_score,
            "summary": {
                "total_rules": total_rules,
                "pass_count": pass_count,
                "warning_count": warning_count,
                "potential_violations": violation_count,
                "not_verifiable": unverifiable_count,
            },
            "rule_results": rule_results,
        }


# Module level helper
_default_engine = RuleEngine()


def evaluate_declarations(
    declarations: Union[List[Dict[str, Any]], Dict[str, Any]],
    confidence_threshold: float = 0.75,
) -> Dict[str, Any]:
    """Convenience functional interface."""
    engine = RuleEngine(confidence_threshold=confidence_threshold)
    return engine.evaluate(declarations)
