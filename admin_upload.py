"""
Dementia App - Admin PDF Upload & Knowledge Update
===================================================
FR-204: System Administrator uploads PDF documents.
        The system then:
          1. Ingests the PDF into the multimodal RAG index.
          2. Auto-regenerates assessment questions for affected domains.
          3. Updates Vapi voice agent response context.

This is a Python FastAPI server intended to run alongside the Android backend.
The Android app (Kotlin) calls these endpoints from the admin dashboard.
"""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag_pipeline import get_store
from assessment_generator import generate_full_assessment, ASSESSMENT_DOMAINS
from voice_agent_context import build_vapi_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dementia App – RAG Admin API",
    description="FR-204: PDF upload endpoint for System Administrator knowledge updates.",
    version="1.0.0",
)

UPLOAD_DIR = Path("uploaded_pdfs")
UPLOAD_DIR.mkdir(exist_ok=True)


# ─── Models ───────────────────────────────────────────────────────────────────

class IngestStatus(BaseModel):
    status: str
    filename: str
    docs_ingested: int
    images_ingested: int
    message: str


class RegenerateRequest(BaseModel):
    domains: list[str] = ASSESSMENT_DOMAINS
    questions_per_domain: int = 20


# ─── Background Tasks ─────────────────────────────────────────────────────────

def _run_full_update(pdf_path: str, domains: list[str], questions_per_domain: int):
    """
    Background job: ingest PDF → regenerate questions → refresh Vapi context.
    Runs after the upload response is already returned to the admin.
    """
    try:
        store = get_store()
        store.ingest_pdf(pdf_path)
        logger.info(f"PDF ingested: {pdf_path}")

        generate_full_assessment(
            domains=domains,
            questions_per_domain=questions_per_domain,
            save=True,
        )
        logger.info("Assessment questions regenerated.")

        build_vapi_context(save=True)
        logger.info("Vapi voice agent context refreshed.")

    except Exception as e:
        logger.error(f"Background update failed: {e}")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/admin/upload-pdf", response_model=IngestStatus)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    domains: str = ",".join(ASSESSMENT_DOMAINS),
    questions_per_domain: int = 20,
):
    """
    FR-204: Upload a clinical PDF (dementia guidelines, assessment scales, etc.).
    Triggers background ingestion, question regeneration, and Vapi context update.

    Args:
        file:                  PDF file to upload.
        domains:               Comma-separated list of domains to regenerate (default: all).
        questions_per_domain:  Number of MCQ questions to generate per domain (default: 20).
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save upload to disk
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"Received PDF: {dest}")

    # Quick inline ingest for immediate feedback
    store = get_store()
    prev_count = len(store.image_store)
    store.ingest_pdf(str(dest))
    images_added = len(store.image_store) - prev_count

    # Count docs ingested (approximation via store size diff not exposed by FAISS;
    # use fitz for a quick page count as proxy)
    import fitz
    doc_count = sum(1 for _ in fitz.open(str(dest)))

    # Schedule full regeneration in background (non-blocking)
    domain_list = [d.strip() for d in domains.split(",") if d.strip()]
    background_tasks.add_task(
        _run_full_update, str(dest), domain_list, questions_per_domain
    )

    return IngestStatus(
        status="accepted",
        filename=file.filename,
        docs_ingested=doc_count,
        images_ingested=images_added,
        message=(
            "PDF ingested into RAG index. "
            "Assessment questions and Vapi context are being regenerated in the background."
        ),
    )


@app.post("/admin/regenerate-questions")
async def regenerate_questions(request: RegenerateRequest):
    """
    Manually trigger question regeneration without uploading a new PDF.
    Useful after fine-tuning prompts or adjusting domain configuration.
    """
    try:
        assessment = generate_full_assessment(
            domains=request.domains,
            questions_per_domain=request.questions_per_domain,
            save=True,
        )
        total = sum(len(v) for v in assessment.values())
        return JSONResponse(
            content={
                "status": "success",
                "total_questions": total,
                "domains": list(assessment.keys()),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/index-status")
def index_status():
    """Return current RAG index statistics."""
    store = get_store()
    return {
        "vector_store_ready": store.vector_store is not None,
        "images_indexed": len(store.image_store),
        "index_path": str(Path("output/dementia_rag_index").resolve()),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
