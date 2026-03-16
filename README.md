# Dementia App — Multimodal RAG System

AI-powered Retrieval-Augmented Generation backend for the Android dementia monitoring app (Team 10, SRS v1.0).

---

## What This Covers (SRS Requirements)

| SRS Req | Implemented In | Description |
|---------|---------------|-------------|
| FR-201  | `assessment_generator.py` | Generates disease-progression MCQ assessments |
| FR-202  | `voice_agent_context.py`  | AI conversational support during assessments |
| FR-203  | `voice_agent_context.py`  | Time-per-question tracking injected into Vapi prompt |
| FR-204  | `admin_upload.py`, `rag_pipeline.py` | Admin PDF upload → auto-updates questionnaire + voice context |
| FR-205  | `voice_agent_context.py`  | AI invigilator reads questions/options via Vapi TTS |
| SP-301  | `assessment_generator.py` | 20 questions × 6 domains (Stress, Activity, Sleep, Food, Disease, Memory) |
| SP-303  | `voice_agent_context.py`  | Audio test via Vapi TTS/STT |
| SP-304  | `assessment_generator.py:clarify_question()` | Patient clarification via RAG + LLM |

---

## Architecture

```
Admin uploads PDF
       │
       ▼
┌─────────────────────┐
│   admin_upload.py   │  FastAPI endpoint (FR-204)
│   POST /admin/upload│
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   rag_pipeline.py   │  PDF → text chunks + images
│                     │  CLIP embeddings → FAISS index
└────────┬────────────┘
         │
    ┌────┴────────────────────┐
    ▼                         ▼
┌──────────────────┐  ┌───────────────────────┐
│assessment_        │  │ voice_agent_context.py│
│generator.py       │  │                       │
│ - MCQ questions   │  │ - Vapi system prompt  │
│ - Clarifications  │  │ - Per-question context│
│   (SP-304)        │  │ - Vapi call payload   │
└──────────────────┘  └───────────────────────┘
         │
         ▼
   evaluation.py
   (retrieval + question + clarification metrics)
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama (local LLM)
```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2          # or any vision-capable model
```
To use OpenAI instead, set `LLM_MODEL=gpt-4o` and swap `ChatOllama` for `ChatOpenAI` in the source files.

### 3. Configure Vapi
Set your Vapi API key in the environment before calling `build_vapi_call_payload()`:
```bash
export VAPI_API_KEY=your_key_here
```

---

## Usage

### Ingest a PDF (Admin — FR-204)
```bash
python main.py ingest --pdf path/to/clinical_guidelines.pdf
```

### Generate Assessment Questions (FR-201, SP-301)
```bash
# All 6 domains, 20 questions each
python main.py generate

# Specific domains
python main.py generate --domains Memory,Stress --questions 20
```

### Build Vapi Voice Agent Prompt (FR-202, FR-205)
```bash
python main.py vapi
# Output: output/vapi_system_prompt.txt
```

### Run Evaluation
```bash
python main.py evaluate --k 5
# Output: output/evaluation/
```

### Interactive Query (dev/testing)
```bash
python main.py query --q "What are early signs of dementia?"
```

### Start Admin API Server (FR-204)
```bash
python main.py serve --port 8000
# Swagger UI: http://localhost:8000/docs
```

The Android backend (Kotlin) POSTs to `http://<server>:8000/admin/upload-pdf` from the System Administrator dashboard.

---

## Output Files

```
output/
├── dementia_rag_index/        # FAISS vector index (text + images)
├── image_store.npy            # Base64-encoded images from PDFs
├── vapi_system_prompt.txt     # Vapi AI invigilator system prompt
├── questions/
│   ├── full_assessment.json   # All domains combined
│   ├── memory.json
│   ├── stress.json
│   └── ...
└── evaluation/
    ├── retrieval_eval.csv
    ├── question_quality_eval.csv
    ├── clarification_eval.csv
    └── summary.json
```

---

## File Overview

| File | Purpose |
|------|---------|
| `rag_pipeline.py` | Core RAG: CLIP embeddings, PDF processing, FAISS vector store |
| `assessment_generator.py` | MCQ generation per domain, clarification handler |
| `voice_agent_context.py` | Vapi system prompt builder, per-question context, call payload |
| `admin_upload.py` | FastAPI server for admin PDF uploads (FR-204) |
| `evaluation.py` | Retrieval, question quality, and clarification evaluation |
| `main.py` | CLI entrypoint for all commands |
| `requirements.txt` | Python dependencies |
