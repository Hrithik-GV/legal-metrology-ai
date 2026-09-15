from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="System operational status")
    project: str = Field(..., description="Project name")


class BoundingBox(BaseModel):
    """Bounding box coordinates [x1, y1, x2, y2]."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: Optional[float] = None
    label: Optional[str] = None


class DetectionResult(BaseModel):
    """YOLOv8 detected label / panel regions."""
    boxes: List[BoundingBox] = Field(default_factory=list)
    image_shape: Optional[List[int]] = None


class OCRToken(BaseModel):
    """PaddleOCR recognized text snippet with location & confidence."""
    text: str
    confidence: float
    box: Optional[List[List[float]]] = None


class ExtractedDeclaration(BaseModel):
    """Extracted Legal Metrology Package Declarations (Rule 6)."""
    mrp: Optional[str] = None
    net_quantity: Optional[str] = None
    manufacturing_date: Optional[str] = None
    expiry_date: Optional[str] = None
    manufacturer_name: Optional[str] = None
    consumer_care: Optional[str] = None
    country_of_origin: Optional[str] = None
    raw_text: Optional[str] = None


class ComplianceRuleResult(BaseModel):
    """Result of an individual Legal Metrology rule check."""
    rule_id: str
    rule_name: str
    status: str  # "PASS", "FAIL", "WARNING"
    details: str
    mandatory: bool = True


class InspectionResponse(BaseModel):
    """End-to-end Legal Metrology inspection output."""
    filename: str
    overall_status: str  # "COMPLIANT", "NON_COMPLIANT"
    declarations: ExtractedDeclaration
    rule_results: List[ComplianceRuleResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
