"""
Main FastAPI entrypoint for Legal Metrology AI.
"""

from fastapi import FastAPI
from app.schemas.models import HealthResponse

app = FastAPI(
    title="Legal Metrology AI",
    description="Core AI/Vision Engine for Legal Metrology Packaged Commodity Compliance",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint returning system status."""
    return {
        "status": "ok",
        "project": "Legal Metrology AI",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
