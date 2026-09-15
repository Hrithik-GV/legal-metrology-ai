# Legal Metrology AI (Core AI/Vision Engine)

Prototype vision engine for the **SIH Internal Hackathon**, designed to automate compliance checks for packaged commodities under the **Legal Metrology (Packaged Commodities) Rules, 2011**.

---

## 🚀 Vision & Compliance Pipeline Flow

```
Product Image
      │
      ▼
OpenCV Preprocessing (CLAHE, Grayscale, Denoising)
      │
      ▼
YOLOv8 Detection (Declaration Panel Bounding Boxes)
      │
      ▼
PaddleOCR (Text Recognition & Polygons)
      │
      ▼
Declaration Extraction (MRP, Net Qty, Dates, Mfg, Consumer Care)
      │
      ▼
Legal Metrology Rule Checking (PCR 2011 Rule 6 & 9)
      │
      ▼
Compliance Result (Pass / Fail / Warnings)
```

---

## 📁 Project Structure

```
legal-metrology-ai/
│
├── .venv/                      # Python virtual environment
├── app/
│   ├── main.py                 # FastAPI service entrypoint (GET /health)
│   ├── preprocessing/
│   │   └── image_processor.py  # OpenCV preprocessing operations
│   ├── detection/
│   │   └── yolo_detector.py    # YOLOv8 panel detection interface
│   ├── ocr/
│   │   └── paddle_ocr.py       # PaddleOCR text recognition interface
│   ├── extraction/
│   │   └── declaration_extractor.py # Regex/Rule-based declaration extractor
│   ├── rules/
│   │   ├── rule_engine.py      # Statutory rule evaluator
│   │   └── rules.json          # Legal Metrology Rule 6 specifications
│   ├── pipeline/
│   │   └── pipeline.py         # End-to-end vision pipeline coordinator
│   └── schemas/
│       └── models.py           # Pydantic schemas
│
├── models/                     # YOLO checkpoints & model weights (.pt, .onnx)
├── test_images/                # Sample packaging images
├── outputs/                    # Processed images and inspection artifacts
├── requirements.txt            # Dependency manifest
├── README.md                   # Project documentation
└── .gitignore                  # Git exclusions
```

---

## ⚙️ Setup & Installation

### 1. Create and Activate Virtual Environment
```bash
# In the legal-metrology-ai directory:
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Windows (CMD):
.\.venv\Scripts\activate.bat

# Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🌐 Running the FastAPI Application

Start the development server using Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔍 Verification

Verify the service health:

```bash
curl http://127.0.0.1:8000/health
```

Expected output:
```json
{
  "status": "ok",
  "project": "Legal Metrology AI"
}
```

Interactive Swagger documentation is available at:
`http://127.0.0.1:8000/docs`
