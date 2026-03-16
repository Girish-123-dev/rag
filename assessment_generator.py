"""
Dementia App - Assessment Question Generator
=============================================
FR-201: Disease progression assessments
FR-204: AI-Knowledge updates — questions auto-generated from admin-uploaded PDFs

This module generates:
- MCQ questions for cognitive assessment across 6 domains
  (Stress, Activity, Sleep, Food, Disease, Memory)
- Voice-friendly question text for Vapi TTS integration (FR-205)
- Difficulty-tagged questions suitable for dementia patients
"""

import json
import logging
import os
import re
from typing import Optional

from langchain_core.messages import SystemMessage

from rag_pipeline import build_multimodal_message, get_store
from llm_factory import get_llm

logger = logging.getLogger(__name__)

# ── LLM (controlled by LLM_BACKEND env var — see llm_factory.py) ──────────────
llm = get_llm()

# SP-301: 6 assessment domains, 20 questions each
ASSESSMENT_DOMAINS = [
    "Stress",
    "Activity",
    "Sleep",
    "Food",
    "Disease",
    "Memory",
]

QUESTION_OUTPUT_DIR = "output/questions"


# ── JSON Cleaning ─────────────────────────────────────────────────────────────

def clean_and_parse_json(raw: str) -> list:
    """
    Robustly extract a JSON array from LLM output.
    Handles markdown fences, trailing commas, truncated output, and extra text.
    """
    # Step 1: Strip markdown code fences (```json ... ``` or ``` ... ```)
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()

    # Step 2: Find the JSON array boundaries — from first [ to last ]
    start = raw.find("[")
    end   = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in LLM output.")
    raw = raw[start:end + 1]

    # Step 3: Remove trailing commas before ] or } (common LLM mistake)
    raw = re.sub(r",\s*(\})", r"\1", raw)
    raw = re.sub(r",\s*(\])", r"\1", raw)

    # Step 4: Try parsing directly
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Step 5: If still broken, extract individual objects one by one
    # Find all {...} blocks and parse each separately
    objects = []
    depth = 0
    current = ""
    for char in raw:
        if char == "{":
            depth += 1
        if depth > 0:
            current += char
        if char == "}":
            depth -= 1
            if depth == 0 and current.strip():
                try:
                    obj = json.loads(current.strip())
                    objects.append(obj)
                except json.JSONDecodeError:
                    pass  # skip malformed individual object
                current = ""

    if objects:
        return objects

    raise ValueError("Could not extract any valid JSON objects from LLM output.")


# ── MCQ Generation ────────────────────────────────────────────────────────────

def generate_mcq_for_domain(domain: str, num_questions: int = 20) -> list:
    """
    Generate MCQ assessment questions for a given domain using RAG context.

    Each question format (SP-301, SP-302):
    {
        "id":         "stress_001",
        "domain":     "Stress",
        "question":   "How often do you feel overwhelmed?",
        "options":    ["Never", "Sometimes", "Often", "Always"],
        "correct":    null,
        "voice_text": "...",       # TTS-friendly for Vapi (FR-205)
        "difficulty": "easy"       # easy | medium | hard
    }
    """
    store = get_store()

    retrieved = store.retrieve(
        query=f"dementia patient assessment questions about {domain.lower()}",
        k=8,
    )

    generation_prompt = (
        f"Generate exactly {num_questions} multiple-choice survey questions "
        f"for dementia patients about the domain: '{domain}'.\n\n"
        "STRICT RULES:\n"
        "- Use simple, clear language suitable for elderly patients.\n"
        "- Each question must have exactly 4 options.\n"
        "- Include a 'voice_text' field: natural spoken version for text-to-speech.\n"
        "- Include a 'difficulty' field: easy, medium, or hard.\n"
        "- Return ONLY a valid JSON array. No explanation. No markdown. No extra text.\n"
        "- Do NOT add a trailing comma after the last item.\n"
        "- Make sure every string value uses double quotes.\n\n"
        "Use this exact JSON format for every item:\n"
        '{"id":"stress_001","domain":"Stress","question":"Question text here?",'
        '"options":["Option A","Option B","Option C","Option D"],"correct":null,'
        '"voice_text":"Spoken version here.","difficulty":"easy"}\n\n'
        "Use the clinical context below to make questions relevant to dementia assessment."
    )

    message = build_multimodal_message(
        query=generation_prompt,
        retrieved_docs=retrieved,
        image_store=store.image_store,
        domain="dementia cognitive assessment",
    )

    system = SystemMessage(
        content=(
            "You are a clinical neuropsychologist specialising in dementia assessment. "
            "You create compassionate, unambiguous survey questions for patients with "
            "moderate cognitive impairment. You always return pure valid JSON arrays only."
        )
    )

    logger.info(f"Generating {num_questions} MCQ questions for domain: {domain}")

    # Retry up to 3 times — LLMs occasionally produce malformed JSON
    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            logger.info(f"  Retrying {domain} (attempt {attempt}/{MAX_RETRIES})...")

        response = llm.invoke([system, message])

        try:
            questions = clean_and_parse_json(response.content)
            logger.info(f"  Generated {len(questions)} questions for {domain}")
            return questions
        except Exception as e:
            logger.warning(f"  Attempt {attempt} failed for {domain}: {e}")
            if attempt == MAX_RETRIES:
                logger.error(f"All {MAX_RETRIES} attempts failed for {domain}. Skipping.")
                logger.debug(f"Last raw output:\n{response.content[:800]}")
                return []


# ── Full Assessment ────────────────────────────────────────────────────────────

def generate_full_assessment(
    domains: Optional[list] = None,
    questions_per_domain: int = 20,
    save: bool = True,
) -> dict:
    """
    Generate the complete SP-301 questionnaire across all (or selected) domains.
    Saves to output/questions/<domain>.json and output/questions/full_assessment.json.
    """
    if domains is None:
        domains = ASSESSMENT_DOMAINS

    os.makedirs(QUESTION_OUTPUT_DIR, exist_ok=True)
    full_assessment = {}

    for domain in domains:
        questions = generate_mcq_for_domain(domain, questions_per_domain)
        full_assessment[domain] = questions

        if save and questions:
            path = os.path.join(QUESTION_OUTPUT_DIR, f"{domain.lower()}.json")
            with open(path, "w") as f:
                json.dump(questions, f, indent=2)
            logger.info(f"  Saved {len(questions)} questions → {path}")
        elif not questions:
            logger.warning(f"  No questions generated for {domain} — skipping save.")

    if save:
        full_path = os.path.join(QUESTION_OUTPUT_DIR, "full_assessment.json")
        with open(full_path, "w") as f:
            json.dump(full_assessment, f, indent=2)
        logger.info(f"Full assessment saved → {full_path}")

    return full_assessment


def load_assessment(domain: Optional[str] = None):
    """Load previously generated assessment questions from disk."""
    if domain:
        path = os.path.join(QUESTION_OUTPUT_DIR, f"{domain.lower()}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No questions found for domain '{domain}'.")
        with open(path) as f:
            return json.load(f)

    full_path = os.path.join(QUESTION_OUTPUT_DIR, "full_assessment.json")
    if not os.path.exists(full_path):
        raise FileNotFoundError("No full assessment found. Run generate_full_assessment() first.")
    with open(full_path) as f:
        return json.load(f)


# ── Clarification (SP-304) ────────────────────────────────────────────────────

def clarify_question(question_text: str, patient_follow_up: str) -> str:
    """
    SP-304 / FR-202: Patient asks for clarification about a survey question.
    RAG retrieves context, LLM explains in simple terms.
    """
    store = get_store()
    retrieved = store.retrieve(
        query=f"clarification: {question_text} — {patient_follow_up}", k=5
    )

    clarification_prompt = (
        f"The patient was asked this survey question:\n\"{question_text}\"\n\n"
        f"The patient said: \"{patient_follow_up}\"\n\n"
        "Re-explain the question in very simple, kind, short terms (2-3 sentences max). "
        "No medical jargon."
    )

    message = build_multimodal_message(
        query=clarification_prompt,
        retrieved_docs=retrieved,
        image_store=store.image_store,
        domain="dementia patient support",
    )

    system = SystemMessage(
        content=(
            "You are a compassionate AI assistant helping dementia patients "
            "understand survey questions. Use simple words and a calm, reassuring tone."
        )
    )

    response = llm.invoke([system, message])
    return response.content.strip()


if __name__ == "__main__":
    print("Generating Memory domain questions (5 only for quick test)...")
    qs = generate_mcq_for_domain("Memory", num_questions=5)
    for q in qs:
        print(json.dumps(q, indent=2))