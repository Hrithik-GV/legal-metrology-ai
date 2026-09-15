"""
Legal Metrology Rule Engine.
Evaluates extracted packaging declarations against Legal Metrology (Packaged Commodities) Rules, 2011.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class RuleEngine:
    """Validates extracted declarations against statutory legal metrology rules."""

    def __init__(self, rules_file: Optional[Path] = None):
        if rules_file is None:
            rules_file = Path(__file__).parent / "rules.json"
        self.rules_file = rules_file
        self.rules_data = self._load_rules()

    def _load_rules(self) -> Dict[str, Any]:
        """Loads rules definitions from JSON."""
        if not self.rules_file.exists():
            return {"rules": []}
        with open(self.rules_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate(self, declarations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluates extracted declaration dictionary against rules.
        Returns a list of compliance results per rule.
        """
        results = []
        rules = self.rules_data.get("rules", [])

        for rule in rules:
            field_key = rule.get("field_key")
            rule_id = rule.get("rule_id", "UNKNOWN")
            rule_name = rule.get("rule_name", "Unknown Rule")
            mandatory = rule.get("mandatory", True)

            value = declarations.get(field_key)

            if value:
                status = "PASS"
                details = f"Found: {value}"
            else:
                if mandatory:
                    status = "FAIL"
                    details = f"Mandatory declaration '{rule_name}' is missing or undetectable."
                else:
                    status = "WARNING"
                    details = f"Optional/Conditional declaration '{rule_name}' was not detected."

            results.append({
                "rule_id": rule_id,
                "rule_name": rule_name,
                "status": status,
                "details": details,
                "mandatory": mandatory
            })

        return results

    def is_overall_compliant(self, evaluation_results: List[Dict[str, Any]]) -> bool:
        """Determines if the product passes all mandatory Legal Metrology rules."""
        return not any(r["status"] == "FAIL" and r.get("mandatory", True) for r in evaluation_results)
