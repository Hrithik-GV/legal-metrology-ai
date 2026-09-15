"""
Declaration Extractor for Legal Metrology AI.
Extracts mandatory packaging declarations from OCR text tokens using pattern matching and heuristics.
"""

from typing import List, Dict, Any, Optional
import re


class DeclarationExtractor:
    """Parses extracted text tokens into structured Legal Metrology declarations."""

    def __init__(self):
        # Baseline regex patterns for key packaged commodity attributes
        self.mrp_pattern = re.compile(r"(?:MRP|M\.R\.P\.?|Rs\.?|₹)\s*[:.]?\s*([0-9]+(?:\.[0-9]{1,2})?)", re.IGNORECASE)
        self.net_qty_pattern = re.compile(r"(?:Net\s*(?:Qty|Quantity|Wt|Weight)|Quantity)\s*[:.]?\s*([0-9]+(?:\.[0-9]+)?\s*(?:g|kg|ml|l|ltr|pieces|units|N))\b", re.IGNORECASE)
        self.mfg_date_pattern = re.compile(r"(?:Mfg|Pkg|Packed|PKD|Date of Pkg)\s*[:.]?\s*([0-9]{1,2}[/\.-][0-9]{1,2}[/\.-][0-9]{2,4}|[A-Za-z]{3,}\s*[0-9]{2,4})", re.IGNORECASE)
        self.expiry_pattern = re.compile(r"(?:Exp|Expiry|Use\s*by|Best\s*before)\s*[:.]?\s*([0-9]{1,2}[/\.-][0-9]{1,2}[/\.-][0-9]{2,4}|[0-9]+\s*months?)", re.IGNORECASE)
        self.consumer_care_pattern = re.compile(r"(?:Consumer\s*Care|Customer\s*Care|Helpline|Toll\s*Free)\s*[:.]?\s*([0-9]{10,12}|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})", re.IGNORECASE)
        self.country_origin_pattern = re.compile(r"(?:Country\s*of\s*Origin|Made\s*in)\s*[:.]?\s*([A-Za-z]+)", re.IGNORECASE)

    def extract_declarations(self, ocr_tokens: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
        """
        Parses list of OCR tokens into structured declaration dictionary.
        """
        combined_text = " \n ".join(token.get("text", "") for token in ocr_tokens)

        declarations = {
            "mrp": None,
            "net_quantity": None,
            "manufacturing_date": None,
            "expiry_date": None,
            "manufacturer_name": None,
            "consumer_care": None,
            "country_of_origin": None,
            "raw_text": combined_text,
        }

        mrp_match = self.mrp_pattern.search(combined_text)
        if mrp_match:
            declarations["mrp"] = mrp_match.group(0).strip()

        qty_match = self.net_qty_pattern.search(combined_text)
        if qty_match:
            declarations["net_quantity"] = qty_match.group(0).strip()

        mfg_match = self.mfg_date_pattern.search(combined_text)
        if mfg_match:
            declarations["manufacturing_date"] = mfg_match.group(0).strip()

        exp_match = self.expiry_pattern.search(combined_text)
        if exp_match:
            declarations["expiry_date"] = exp_match.group(0).strip()

        care_match = self.consumer_care_pattern.search(combined_text)
        if care_match:
            declarations["consumer_care"] = care_match.group(0).strip()

        origin_match = self.country_origin_pattern.search(combined_text)
        if origin_match:
            declarations["country_of_origin"] = origin_match.group(0).strip()

        return declarations
