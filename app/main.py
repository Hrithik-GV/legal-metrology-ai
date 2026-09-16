"""
FastAPI Application for Legal Metrology AI Engine.
Provides RESTful endpoints for packaged commodity inspection:
- POST /analyze: Multipart file upload running full vision & rule pipeline
- GET /results/{inspection_id}: Retrieves persisted inspection JSON artifact
- GET /health: Operational readiness check
- Static file serving for visual outputs under /outputs
- Cross-Origin Resource Sharing (CORS) enabled for future frontend integration
"""

import os
import sys
import json
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from app.schemas.models import HealthResponse
from app.pipeline.pipeline import VisionPipeline, run_pipeline

# Initialize FastAPI app
app = FastAPI(
    title="Legal Metrology AI",
    description=(
        "Core AI/Vision Engine for Legal Metrology Packaged Commodity Compliance. "
        "Performs OpenCV preprocessing, YOLOv8 region detection, PaddleOCR, declaration extraction, "
        "and modular rule checking."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for future frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directory structure
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"
FINAL_DIR = OUTPUTS_DIR / "final"

for d in [OUTPUTS_DIR, RESULTS_DIR, UPLOADS_DIR, FINAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Mount outputs directory for visual inspections
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Singleton pipeline instance
pipeline = VisionPipeline(output_base_dir=OUTPUTS_DIR)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check():
    """Health check endpoint returning system status."""
    return {
        "status": "ok",
        "project": "Legal Metrology AI",
    }


@app.post("/analyze", tags=["Inspection"])
async def analyze_product_package(file: UploadFile = File(...)):
    """
    Accepts a packaged commodity image (multipart/form-data) and executes:
    1. OpenCV Preprocessing
    2. YOLOv8 Region Detection
    3. PaddleOCR Text Recognition
    4. Legal Metrology Declaration Extraction
    5. Modular Rule Engine Evaluation

    Returns complete JSON results and stores a permanent record in outputs/results/.
    """
    # 1. Validate file format
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    filename = file.filename or "upload.jpg"
    ext = Path(filename).suffix.lower()
    if ext not in valid_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed extensions: {list(valid_extensions)}",
        )

    # 2. Assign unique inspection identifier and save uploaded file
    inspection_id = f"insp_{uuid.uuid4().hex[:12]}"
    saved_filename = f"{inspection_id}_{filename}"
    saved_path = UPLOADS_DIR / saved_filename

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )
        with open(saved_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}",
        )

    # 3. Execute Complete Vision and Rule Pipeline
    try:
        pipeline_output = pipeline.process(image_path=saved_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vision pipeline processing error: {str(e)}",
        )

    # 4. Extract compliance metrics
    rule_assessment = pipeline_output.get("rule_assessment", {})
    compliance_checks = rule_assessment.get("rule_results", [])
    compliance_score = rule_assessment.get("compliance_score", 0.0)

    # Identify potential violations and warnings
    violations = [
        check for check in compliance_checks
        if check.get("status") == "POTENTIAL_VIOLATION"
    ]
    warnings = [
        check for check in compliance_checks
        if check.get("status") == "WARNING"
    ]

    # Derive overall status adhering to statutory preliminary assessment wording
    if len(violations) > 0:
        overall_status = "POTENTIAL_VIOLATIONS_FOUND"
    elif len(warnings) > 0:
        overall_status = "MANUAL_REVIEW_RECOMMENDED"
    else:
        overall_status = "PRELIMINARY_ASSESSMENT_PASSED"

    # Image info
    img_dims = pipeline_output.get("dimensions", {}).get("original", [0, 0])
    image_information = {
        "original_filename": filename,
        "saved_path": str(saved_path),
        "content_type": file.content_type or "image/jpeg",
        "file_size_bytes": len(contents),
        "dimensions": {
            "height": img_dims[0] if len(img_dims) > 0 else 0,
            "width": img_dims[1] if len(img_dims) > 1 else 0,
        },
    }

    # Format unified response
    response_payload: Dict[str, Any] = {
        "inspection_id": inspection_id,
        "assessment": rule_assessment.get("assessment", "AI-Assisted Preliminary Assessment"),
        "disclaimer": rule_assessment.get("disclaimer", "Manual verification required."),
        "overall_status": overall_status,
        "compliance_score": compliance_score,
        "image_information": image_information,
        "extracted_declarations": pipeline_output.get("declarations", []),
        "ocr_results": pipeline_output.get("ocr_results", []),
        "detections": pipeline_output.get("detections", []),
        "regions": pipeline_output.get("regions", []),
        "compliance_checks": compliance_checks,
        "violations": violations,
        "final_annotated_image_path": pipeline_output.get("annotated_image", ""),
        "annotated_image_url": f"/outputs/final/{Path(pipeline_output.get('annotated_image', '')).name}",
    }

    # 5. Persist inspection result to outputs/results/{inspection_id}.json
    result_json_path = RESULTS_DIR / f"{inspection_id}.json"
    try:
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(response_payload, f, indent=2)
    except Exception as err:
        # Non-fatal to response, log warning
        print(f"[WARN] Failed to write result cache for {inspection_id}: {err}")

    return response_payload


@app.get("/results/{inspection_id}", tags=["Inspection"])
def get_inspection_result(inspection_id: str):
    """
    Retrieves stored prototype inspection result from outputs/results/.
    """
    result_file = RESULTS_DIR / f"{inspection_id}.json"
    if not result_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inspection result '{inspection_id}' not found in records.",
        )

    try:
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read inspection record: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
