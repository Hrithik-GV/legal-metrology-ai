"""
Declaration Extraction Module for Legal Metrology AI.
Extracts statutory Legal Metrology declarations from OCR tokens:
- product_name
- manufacturer
- packer
- importer
- address
- net_quantity
- mrp
- packed_date
- manufactured_date
- consumer_care
- country_of_origin

Implements deterministic techniques:
- Regex and keyword matching
- Case-insensitive pattern normalization
- Spatial bounding-box proximity for multiline/adjacent token stitching
- Confidence aggregation
- Modular architecture allowing future LLM extraction plugins
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple


def normalize_text(text: str) -> str:
    """Cleans up OCR noise, non-breaking spaces, and duplicate whitespace."""
    if not text:
        return ""
    # Replace common currency symbol variants
    cleaned = text.replace("â‚¹", "₹").replace("\xa0", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def parse_box_bounds(box: List[Any]) -> Optional[Tuple[float, float, float, float]]:
    """
    Computes (x_min, y_min, x_max, y_max) from 4-point polygon or 4-element box.
    """
    if not box:
        return None
    try:
        if len(box) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in box):
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            return (min(xs), min(ys), max(xs), max(ys))
        elif len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            return (box[0], box[1], box[2], box[3])
    except Exception:
        return None
    return None


def union_boxes(box1: Optional[List[Any]], box2: Optional[List[Any]]) -> Optional[List[List[float]]]:
    """Computes a bounding box polygon covering both boxes."""
    if not box1:
        return box2
    if not box2:
        return box1

    b1 = parse_box_bounds(box1)
    b2 = parse_box_bounds(box2)
    if not b1 or not b2:
        return box1

    x_min = min(b1[0], b2[0])
    y_min = min(b1[1], b2[1])
    x_max = max(b1[2], b2[2])
    y_max = max(b1[3], b2[3])

    return [
        [round(x_min, 1), round(y_min, 1)],
        [round(x_max, 1), round(y_min, 1)],
        [round(x_max, 1), round(y_max, 1)],
        [round(x_min, 1), round(y_max, 1)],
    ]


class BaseDeclarationExtractor(ABC):
    """Abstract interface for Legal Metrology declaration extractors."""

    @abstractmethod
    def extract(
        self, tokens: List[Dict[str, Any]], image_id: str = "image_0"
    ) -> List[Dict[str, Any]]:
        """
        Extracts structured declarations from OCR tokens.
        Each item returns:
        {
            "field": "...",
            "value": "...",
            "confidence": 0.95,
            "source_text": "...",
            "bounding_box": [...],
            "image_id": "..."
        }
        """
        pass


class DeterministicDeclarationExtractor(BaseDeclarationExtractor):
    """
    Deterministic rule & spatial-based extractor using regular expressions,
    keyword triggers, and bounding-box proximity.
    """

    def __init__(self):
        # 1. MRP Patterns
        self.mrp_triggers = re.compile(
            r"\b(?:MRP|M\.R\.P\.?|Maximum\s+Retail\s+Price|Max\s+Retail\s+Price|Retail\s+Price)\b",
            re.IGNORECASE,
        )
        self.mrp_value_regex = re.compile(
            r"(?:₹|Rs\.?|INR)?\s*([0-9]+(?:[,\.][0-9]{1,2})?)\s*(?:/-)?",
            re.IGNORECASE,
        )

        # 2. Net Quantity Patterns
        self.net_qty_triggers = re.compile(
            r"\b(?:Net\s*(?:Quantity|Qty|Wt|Weight|Content)|Quantity|Weight|Vol|Volume)\b",
            re.IGNORECASE,
        )
        self.qty_value_regex = re.compile(
            r"([0-9]+(?:\.[0-9]+)?)\s*(kg|g|gm|gms|gram|grams|ml|l|ltr|litre|litres|n|units|pieces|pc|pcs)\b",
            re.IGNORECASE,
        )

        # 3. Manufactured Date Patterns
        self.mfg_triggers = re.compile(
            r"\b(?:Mfg(?:\s*Date|\s*Dt)?|Mfd(?:\s*Date|\s*Dt)?|Date\s*of\s*Mfg|Date\s*of\s*Manufacture|Manufactured\s*(?:on|Date)?)\b",
            re.IGNORECASE,
        )

        # 4. Packed Date Patterns
        self.pkd_triggers = re.compile(
            r"\b(?:PKD|Pkg|Packed(?:\s*on|\s*date)?|Date\s*of\s*Pkg|Date\s*of\s*Packing)\b",
            re.IGNORECASE,
        )

        self.date_regex = re.compile(
            r"((?:[0-3]?[0-9][/-])?(?:0[1-9]|1[0-2]|[1-9])[/-](?:19|20)?[0-9]{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*(?:19|20)?[0-9]{2,4})",
            re.IGNORECASE,
        )

        # 5. Consumer Care Patterns
        self.care_triggers = re.compile(
            r"\b(?:Consumer\s*Care|Customer\s*Care|Customer\s*Helpline|Helpline|Toll\s*Free|Feedback|Complaints?)\b",
            re.IGNORECASE,
        )
        self.phone_regex = re.compile(r"(?:1800[-\s]?[0-9]{2,4}[-\s]?[0-9]{3,4}|[0-9]{10,12})")
        self.email_regex = re.compile(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")

        # 6. Country of Origin Patterns
        self.origin_triggers = re.compile(
            r"\b(?:Country\s*of\s*Origin|Made\s*in|Product\s*of)\b",
            re.IGNORECASE,
        )

        # 7. Manufacturer / Packer / Importer Patterns
        self.mfg_by_trigger = re.compile(
            r"\b(?:Manufactured\s*by|Mfd\s*by|Mfg\s*by|Manufacturer)\b",
            re.IGNORECASE,
        )
        self.packer_by_trigger = re.compile(
            r"\b(?:Packed\s*by|Pkd\s*by|Packer)\b",
            re.IGNORECASE,
        )
        self.importer_by_trigger = re.compile(
            r"\b(?:Imported\s*by|Imp\s*by|Importer)\b",
            re.IGNORECASE,
        )

        # 8. Address / Pincode
        self.pincode_regex = re.compile(r"\b([1-9][0-9]{5})\b")
        self.address_keywords = re.compile(
            r"\b(?:Plot\s*No\.?|MIDC|Industrial\s*Area|Road|Street|Lane|Sector|Phase|Pincode|Pin|Dist\.?|State|Mumbai|Delhi|Bengaluru|Bangalore|Chennai|Kolkata|Hyderabad|Andheri|Pune)\b",
            re.IGNORECASE,
        )

    def _find_spatially_adjacent(
        self, target_idx: int, tokens: List[Dict[str, Any]], max_x_dist: float = 450.0, max_y_dist: float = 60.0
    ) -> List[Tuple[int, Dict[str, Any]]]:
        """
        Finds tokens immediately adjacent to the right or below the target token.
        """
        target_token = tokens[target_idx]
        t_box = parse_box_bounds(target_token.get("bounding_box", []))
        if not t_box:
            # If no spatial info, return the immediate next token
            if target_idx + 1 < len(tokens):
                return [(target_idx + 1, tokens[target_idx + 1])]
            return []

        tx_min, ty_min, tx_max, ty_max = t_box
        adjacent = []

        for i, token in enumerate(tokens):
            if i == target_idx:
                continue
            box = parse_box_bounds(token.get("bounding_box", []))
            if not box:
                continue
            x_min, y_min, x_max, y_max = box

            # Check horizontally adjacent to the right
            horizontal_aligned = abs(y_min - ty_min) <= max_y_dist
            to_the_right = 0 <= (x_min - tx_max) <= max_x_dist
            if horizontal_aligned and to_the_right:
                adjacent.append((i, token))
                continue

            # Check immediately below (vertical label stacked pattern)
            vertical_aligned = abs(x_min - tx_min) <= max_x_dist
            below = 0 <= (y_min - ty_max) <= max_y_dist
            if vertical_aligned and below:
                adjacent.append((i, token))

        return adjacent

    def extract(
        self, tokens: List[Dict[str, Any]], image_id: str = "image_0"
    ) -> List[Dict[str, Any]]:
        """
        Processes token list and produces canonical declarations.
        """
        declarations: Dict[str, Dict[str, Any]] = {}

        # Pre-normalize token texts
        clean_tokens = []
        for t in tokens:
            raw_t = t.get("text", "")
            clean_tokens.append({
                "text": normalize_text(raw_t),
                "raw_text": raw_t,
                "confidence": float(t.get("confidence", 0.90)),
                "bounding_box": t.get("bounding_box", []),
            })

        all_text_combined = " \n ".join(t["text"] for t in clean_tokens)

        # 1. EXTRACT MRP
        for i, tok in enumerate(clean_tokens):
            txt = tok["text"]
            if self.mrp_triggers.search(txt):
                # Check within same token
                mrp_match = self.mrp_value_regex.search(txt)
                val = None
                src_tok = tok
                box = tok["bounding_box"]

                if mrp_match:
                    num = mrp_match.group(1).replace(",", ".")
                    val = f"₹{num}"
                else:
                    # Look at adjacent tokens
                    adj = self._find_spatially_adjacent(i, clean_tokens)
                    for _, adj_tok in adj:
                        sub_match = self.mrp_value_regex.search(adj_tok["text"])
                        if sub_match:
                            num = sub_match.group(1).replace(",", ".")
                            val = f"₹{num}"
                            box = union_boxes(tok["bounding_box"], adj_tok["bounding_box"])
                            src_tok = {"text": f"{tok['text']} {adj_tok['text']}", "confidence": (tok["confidence"] + adj_tok["confidence"]) / 2}
                            break

                if val and "mrp" not in declarations:
                    declarations["mrp"] = {
                        "field": "mrp",
                        "value": val,
                        "confidence": round(src_tok["confidence"], 2),
                        "source_text": src_tok["text"],
                        "bounding_box": box,
                        "image_id": image_id,
                    }
                    break

        # 2. EXTRACT NET QUANTITY
        # Pass 1: Prioritize explicit "Net Quantity / Net Weight / Net Wt" triggers
        for i, tok in enumerate(clean_tokens):
            txt = tok["text"]
            if self.net_qty_triggers.search(txt):
                qty_match = self.qty_value_regex.search(txt)
                val = None
                box = tok["bounding_box"]
                src_tok = tok

                if qty_match:
                    amount, unit = qty_match.groups()
                    u_norm = unit.lower()
                    if u_norm in ["g", "gm", "gms", "gram", "grams"]:
                        u_norm = "g"
                    elif u_norm in ["kg"]:
                        u_norm = "kg"
                    elif u_norm in ["ml"]:
                        u_norm = "ml"
                    elif u_norm in ["l", "ltr", "litre", "litres"]:
                        u_norm = "L"
                    val = f"{amount} {u_norm}"
                else:
                    # Look at adjacent tokens
                    adj = self._find_spatially_adjacent(i, clean_tokens)
                    for _, adj_tok in adj:
                        sub_match = self.qty_value_regex.search(adj_tok["text"])
                        if sub_match:
                            amount, unit = sub_match.groups()
                            u_norm = unit.lower()
                            if u_norm in ["g", "gm", "gms", "gram", "grams"]:
                                u_norm = "g"
                            elif u_norm in ["l", "ltr", "litre", "litres"]:
                                u_norm = "L"
                            val = f"{amount} {u_norm}"
                            box = union_boxes(tok["bounding_box"], adj_tok["bounding_box"])
                            src_tok = {"text": f"{tok['text']} {adj_tok['text']}", "confidence": (tok["confidence"] + adj_tok["confidence"]) / 2}
                            break

                if val and "net_quantity" not in declarations:
                    declarations["net_quantity"] = {
                        "field": "net_quantity",
                        "value": val,
                        "confidence": round(src_tok["confidence"], 2),
                        "source_text": src_tok["text"],
                        "bounding_box": box,
                        "image_id": image_id,
                    }
                    break

        # Pass 2: Fallback to standalone quantity pattern if not in a nutrition table
        if "net_quantity" not in declarations:
            for i, tok in enumerate(clean_tokens):
                txt = tok["text"]
                if "NUTRITION" in txt.upper() or "PER 100" in txt.upper():
                    continue
                qty_match = self.qty_value_regex.search(txt)
                if qty_match:
                    amount, unit = qty_match.groups()
                    u_norm = unit.lower()
                    if u_norm in ["g", "gm", "gms", "gram", "grams"]:
                        u_norm = "g"
                    elif u_norm in ["l", "ltr", "litre", "litres"]:
                        u_norm = "L"
                    declarations["net_quantity"] = {
                        "field": "net_quantity",
                        "value": f"{amount} {u_norm}",
                        "confidence": round(tok["confidence"], 2),
                        "source_text": tok["text"],
                        "bounding_box": tok["bounding_box"],
                        "image_id": image_id,
                    }
                    break

        # 3. EXTRACT PACKED DATE (PKD)
        for i, tok in enumerate(clean_tokens):
            txt = tok["text"]
            if self.pkd_triggers.search(txt):
                val = None
                box = tok["bounding_box"]
                src_tok = tok

                d_match = self.date_regex.search(txt)
                if d_match:
                    val = d_match.group(1).strip()
                else:
                    adj = self._find_spatially_adjacent(i, clean_tokens)
                    for _, adj_tok in adj:
                        sub_d = self.date_regex.search(adj_tok["text"])
                        if sub_d:
                            val = sub_d.group(1).strip()
                            box = union_boxes(tok["bounding_box"], adj_tok["bounding_box"])
                            src_tok = {"text": f"{tok['text']} {adj_tok['text']}", "confidence": (tok["confidence"] + adj_tok["confidence"]) / 2}
                            break

                if val and "packed_date" not in declarations:
                    declarations["packed_date"] = {
                        "field": "packed_date",
                        "value": val,
                        "confidence": round(src_tok["confidence"], 2),
                        "source_text": src_tok["text"],
                        "bounding_box": box,
                        "image_id": image_id,
                    }
                    break

        # 4. EXTRACT MANUFACTURED DATE
        for i, tok in enumerate(clean_tokens):
            txt = tok["text"]
            if self.mfg_triggers.search(txt):
                val = None
                box = tok["bounding_box"]
                src_tok = tok

                d_match = self.date_regex.search(txt)
                if d_match:
                    val = d_match.group(1).strip()
                else:
                    adj = self._find_spatially_adjacent(i, clean_tokens)
                    for _, adj_tok in adj:
                        sub_d = self.date_regex.search(adj_tok["text"])
                        if sub_d:
                            val = sub_d.group(1).strip()
                            box = union_boxes(tok["bounding_box"], adj_tok["bounding_box"])
                            src_tok = {"text": f"{tok['text']} {adj_tok['text']}", "confidence": (tok["confidence"] + adj_tok["confidence"]) / 2}
                            break

                if val and "manufactured_date" not in declarations:
                    declarations["manufactured_date"] = {
                        "field": "manufactured_date",
                        "value": val,
                        "confidence": round(src_tok["confidence"], 2),
                        "source_text": src_tok["text"],
                        "bounding_box": box,
                        "image_id": image_id,
                    }
                    break

        # 5. EXTRACT CONSUMER CARE
        for i, tok in enumerate(clean_tokens):
            txt = tok["text"]
            is_care = self.care_triggers.search(txt)
            phone_m = self.phone_regex.search(txt)
            email_m = self.email_regex.search(txt)

            if is_care or phone_m or email_m:
                val_parts = []
                box = tok["bounding_box"]
                src_tok = tok

                if phone_m:
                    val_parts.append(phone_m.group(0))
                if email_m:
                    val_parts.append(email_m.group(0))

                if not val_parts and is_care:
                    adj = self._find_spatially_adjacent(i, clean_tokens, max_y_dist=80.0)
                    for _, adj_tok in adj:
                        p_sub = self.phone_regex.search(adj_tok["text"])
                        e_sub = self.email_regex.search(adj_tok["text"])
                        if p_sub:
                            val_parts.append(p_sub.group(0))
                        if e_sub:
                            val_parts.append(e_sub.group(0))
                        if p_sub or e_sub:
                            box = union_boxes(box, adj_tok["bounding_box"])

                if val_parts and "consumer_care" not in declarations:
                    declarations["consumer_care"] = {
                        "field": "consumer_care",
                        "value": ", ".join(val_parts),
                        "confidence": round(src_tok["confidence"], 2),
                        "source_text": src_tok["text"],
                        "bounding_box": box,
                        "image_id": image_id,
                    }
                    break

        # 6. EXTRACT COUNTRY OF ORIGIN
        for i, tok in enumerate(clean_tokens):
            txt = tok["text"]
            if self.origin_triggers.search(txt):
                val = None
                box = tok["bounding_box"]
                src_tok = tok

                # Strip trigger prefix
                after_trigger = self.origin_triggers.split(txt)[-1].strip(" :.-")
                if after_trigger and len(after_trigger) > 2:
                    val = after_trigger
                else:
                    adj = self._find_spatially_adjacent(i, clean_tokens)
                    for _, adj_tok in adj:
                        if len(adj_tok["text"]) > 2:
                            val = adj_tok["text"].strip(" :.-")
                            box = union_boxes(box, adj_tok["bounding_box"])
                            src_tok = {"text": f"{tok['text']} {adj_tok['text']}", "confidence": (tok["confidence"] + adj_tok["confidence"]) / 2}
                            break

                if val and "country_of_origin" not in declarations:
                    declarations["country_of_origin"] = {
                        "field": "country_of_origin",
                        "value": val,
                        "confidence": round(src_tok["confidence"], 2),
                        "source_text": src_tok["text"],
                        "bounding_box": box,
                        "image_id": image_id,
                    }
                    break

        # 7. EXTRACT MANUFACTURER / PACKER / IMPORTER / ADDRESS
        for i, tok in enumerate(clean_tokens):
            txt = tok["text"]

            # Manufacturer
            if self.mfg_by_trigger.search(txt) and "manufacturer" not in declarations:
                after_mfg = self.mfg_by_trigger.split(txt)[-1].strip(" :.-")
                mfg_val = after_mfg if len(after_mfg) > 3 else None
                box = tok["bounding_box"]
                src_tok = tok

                if not mfg_val:
                    adj = self._find_spatially_adjacent(i, clean_tokens)
                    if adj:
                        _, adj_tok = adj[0]
                        mfg_val = adj_tok["text"]
                        box = union_boxes(box, adj_tok["bounding_box"])
                        src_tok = {"text": f"{tok['text']} {adj_tok['text']}", "confidence": (tok["confidence"] + adj_tok["confidence"]) / 2}

                if mfg_val:
                    # Clean company name
                    declarations["manufacturer"] = {
                        "field": "manufacturer",
                        "value": mfg_val,
                        "confidence": round(src_tok["confidence"], 2),
                        "source_text": src_tok["text"],
                        "bounding_box": box,
                        "image_id": image_id,
                    }

            # Packer
            if self.packer_by_trigger.search(txt) and "packer" not in declarations:
                after_pkr = self.packer_by_trigger.split(txt)[-1].strip(" :.-")
                pkr_val = after_pkr if len(after_pkr) > 3 else None
                box = tok["bounding_box"]
                src_tok = tok

                if not pkr_val:
                    adj = self._find_spatially_adjacent(i, clean_tokens)
                    if adj:
                        _, adj_tok = adj[0]
                        pkr_val = adj_tok["text"]
                        box = union_boxes(box, adj_tok["bounding_box"])
                        src_tok = {"text": f"{tok['text']} {adj_tok['text']}", "confidence": (tok["confidence"] + adj_tok["confidence"]) / 2}

                if pkr_val:
                    declarations["packer"] = {
                        "field": "packer",
                        "value": pkr_val,
                        "confidence": round(src_tok["confidence"], 2),
                        "source_text": src_tok["text"],
                        "bounding_box": box,
                        "image_id": image_id,
                    }

            # Importer
            if self.importer_by_trigger.search(txt) and "importer" not in declarations:
                after_imp = self.importer_by_trigger.split(txt)[-1].strip(" :.-")
                imp_val = after_imp if len(after_imp) > 3 else None
                box = tok["bounding_box"]
                src_tok = tok

                if not imp_val:
                    adj = self._find_spatially_adjacent(i, clean_tokens)
                    if adj:
                        _, adj_tok = adj[0]
                        imp_val = adj_tok["text"]
                        box = union_boxes(box, adj_tok["bounding_box"])
                        src_tok = {"text": f"{tok['text']} {adj_tok['text']}", "confidence": (tok["confidence"] + adj_tok["confidence"]) / 2}

                if imp_val:
                    declarations["importer"] = {
                        "field": "importer",
                        "value": imp_val,
                        "confidence": round(src_tok["confidence"], 2),
                        "source_text": src_tok["text"],
                        "bounding_box": box,
                        "image_id": image_id,
                    }

            # Address extraction
            if (self.pincode_regex.search(txt) or self.address_keywords.search(txt)) and "address" not in declarations:
                declarations["address"] = {
                    "field": "address",
                    "value": txt,
                    "confidence": round(tok["confidence"], 2),
                    "source_text": tok["text"],
                    "bounding_box": tok["bounding_box"],
                    "image_id": image_id,
                }

        # 8. PRODUCT NAME
        # Locate brand / main headline token (typically larger text or prominent first headers)
        # 8. PRODUCT NAME
        # Locate brand / main headline token (typically prominent brand/product text)
        if "product_name" not in declarations:
            candidate_names = []
            exclude_keywords = [
                "NUTRITION", "BATCH", "FSSAI", "BEST BEFORE", "USE BY", "EXPIRY",
                "PRODUCT INFORMATION", "INGREDIENTS", "VEGETABLE OIL", "STORAGE",
                "KEEP IN", "EDIBLE", "VEG"
            ]
            for tok in clean_tokens:
                txt = tok["text"]
                txt_upper = txt.upper()
                # Exclude standard rule label lines
                if (
                    not self.mrp_triggers.search(txt)
                    and not self.net_qty_triggers.search(txt)
                    and not self.mfg_triggers.search(txt)
                    and not self.pkd_triggers.search(txt)
                    and not self.care_triggers.search(txt)
                    and not self.origin_triggers.search(txt)
                    and not self.mfg_by_trigger.search(txt)
                    and not self.packer_by_trigger.search(txt)
                    and not any(ex in txt_upper for ex in exclude_keywords)
                    and len(txt) > 3
                ):
                    candidate_names.append(tok)

            if candidate_names:
                # Prefer candidates with title/brand characteristics
                best_name_tok = max(candidate_names, key=lambda t: len(t["text"]))
                declarations["product_name"] = {
                    "field": "product_name",
                    "value": best_name_tok["text"],
                    "confidence": round(best_name_tok["confidence"], 2),
                    "source_text": best_name_tok["text"],
                    "bounding_box": best_name_tok["bounding_box"],
                    "image_id": image_id,
                }

        return list(declarations.values())


class DeclarationExtractor:
    """
    Main Declaration Extractor facade.
    Default uses DeterministicDeclarationExtractor.
    Modularly accepts any BaseDeclarationExtractor (e.g. future LLM extractor).
    """

    def __init__(self, backend_extractor: Optional[BaseDeclarationExtractor] = None):
        self.backend = backend_extractor or DeterministicDeclarationExtractor()

    def set_backend(self, extractor: BaseDeclarationExtractor):
        """Allows hot-swapping or injecting an LLM extractor in the future."""
        self.backend = extractor

    def extract_declarations(
        self, tokens: Union[List[Dict[str, Any]], Dict[str, Any]], image_id: str = "image_0"
    ) -> List[Dict[str, Any]]:
        """
        Accepts raw token list or the output dict from PaddleOCR extract_text().
        Returns list of structured declaration items:
        [
            {
                "field": "mrp",
                "value": "₹450",
                "confidence": 0.94,
                "source_text": "MRP ₹450",
                "bounding_box": [...],
                "image_id": "..."
            }, ...
        ]
        """
        token_list = tokens
        if isinstance(tokens, dict):
            token_list = tokens.get("results") or tokens.get("detected_regions") or []
            if not token_list and "regions" in tokens:
                token_list = tokens["regions"]
            image_id = tokens.get("image_path", image_id)

        return self.backend.extract(token_list, image_id=str(image_id))

    def extract_as_dict(
        self, tokens: Union[List[Dict[str, Any]], Dict[str, Any]], image_id: str = "image_0"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convenience accessor returning a dictionary keyed by field name:
        { "mrp": { ... }, "net_quantity": { ... } }
        """
        items = self.extract_declarations(tokens, image_id=image_id)
        return {item["field"]: item for item in items}


# Module level convenience function
_default_extractor = DeclarationExtractor()


def extract_declarations(
    tokens: Union[List[Dict[str, Any]], Dict[str, Any]], image_id: str = "image_0"
) -> List[Dict[str, Any]]:
    """Top-level convenience function."""
    return _default_extractor.extract_declarations(tokens, image_id=image_id)
