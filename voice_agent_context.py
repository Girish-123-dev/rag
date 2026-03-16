"""
Dementia App - Vapi Voice Agent Context Builder
================================================
FR-202: AI-Based Conversation Support — conversational AI assists patients
        during assessments and routine checks.
FR-205: AI-Based Invigilator — AI reads questions and options aloud, handles
        verbal responses, guides the session.

This module:
- Retrieves relevant clinical context from the RAG store.
- Builds a Vapi-compatible system prompt that is injected into the
  voice agent at session start.
- Provides per-question context enrichment (called during live sessions).

Vapi integration: https://docs.vapi.ai
The Android app sends a /call/create request to Vapi with the system prompt
returned by build_vapi_context(). Vapi handles TTS/STT via the PSTN layer.
"""

import json
import logging
import os
from typing import Optional

from langchain_core.messages import SystemMessage

from rag_pipeline import build_multimodal_message, get_store
from llm_factory import get_llm

logger = logging.getLogger(__name__)

# ── LLM (controlled by LLM_BACKEND env var — see llm_factory.py) ──────────────
llm = get_llm()

VAPI_CONTEXT_PATH = "output/vapi_system_prompt.txt"


# ─── Vapi System Prompt ───────────────────────────────────────────────────────

def build_vapi_context(save: bool = True) -> str:
    """
    FR-202 / FR-205: Build the Vapi voice agent system prompt using RAG context.

    The prompt instructs the AI invigilator to:
    - Read each question and its options clearly.
    - Accept verbal yes/no and option-letter answers.
    - Offer gentle clarification when the patient is confused (SP-304).
    - Keep a calm, reassuring tone appropriate for dementia patients.
    - Log answer and time-on-question (FR-203).
    """
    store = get_store()

    # Pull broad clinical context about dementia communication
    retrieved = store.retrieve(
        query="how to communicate clearly with dementia patients during cognitive assessments",
        k=10,
    )

    synthesis_prompt = (
        "Synthesise the clinical context below into a concise set of instructions "
        "for an AI voice assistant that will conduct cognitive assessments with "
        "dementia patients over a phone call.\n\n"
        "The instructions should cover:\n"
        "1. Tone and pacing (calm, slow, patient).\n"
        "2. How to read a multiple-choice question and its options clearly.\n"
        "3. How to handle a confused or non-responsive patient.\n"
        "4. How to confirm the patient's answer before moving on.\n"
        "5. How to offer a simple clarification without biasing the answer.\n"
        "6. When to escalate to a caretaker.\n\n"
        "Output plain text only — no headers, no markdown. "
        "This text will be used directly as the Vapi assistant system prompt."
    )

    message = build_multimodal_message(
        query=synthesis_prompt,
        retrieved_docs=retrieved,
        image_store=store.image_store,
        domain="dementia patient voice assessment",
    )

    system = SystemMessage(
        content=(
            "You are an expert clinical AI designer specialising in voice-based "
            "cognitive assessments for dementia patients."
        )
    )

    response = llm.invoke([system, message])
    vapi_prompt = response.content.strip()

    # Prepend hard role definition (not overridable by retrieved content)
    full_prompt = _BASE_VAPI_PROMPT + "\n\n" + vapi_prompt

    if save:
        os.makedirs("output", exist_ok=True)
        with open(VAPI_CONTEXT_PATH, "w") as f:
            f.write(full_prompt)
        logger.info(f"Vapi system prompt saved → {VAPI_CONTEXT_PATH}")

    return full_prompt


_BASE_VAPI_PROMPT = """\
You are ARIA — an AI Assessment and Routine Invigilator Agent for a dementia \
monitoring application. Your role is to conduct cognitive surveys with patients \
who may have mild to moderate dementia.

CORE RULES (always follow these):
- Speak slowly and clearly. Pause after each sentence.
- Never rush the patient. Allow up to 30 seconds for a response.
- Address the patient by their first name when known.
- Never reveal assessment scores or progression data to the patient (FR-104).
- After each question, repeat the options once before accepting an answer.
- If the patient says "I don't know" twice for the same question, skip it and \
  note it as unanswered — do not press further.
- If the patient seems distressed, pause the session and offer to connect them \
  to their caretaker.
- Record the time taken to answer each question (FR-203).\
"""


# ─── Per-Question Context ─────────────────────────────────────────────────────

def get_question_context(question_text: str, domain: str) -> str:
    """
    FR-205: Retrieve relevant background for the AI invigilator before
    it reads a specific question. Injected as a mid-session Vapi context update.

    Returns a short paragraph (2-4 sentences) the voice agent uses internally
    to understand what the question is testing — not read aloud to the patient.
    """
    store = get_store()
    retrieved = store.retrieve(
        query=f"{domain} dementia assessment: {question_text}", k=4
    )

    if not retrieved:
        return (
            f"This question assesses the patient's {domain.lower()} domain. "
            "Listen for any signs of confusion or distress."
        )

    message = build_multimodal_message(
        query=(
            f"In 2-4 sentences, explain what the following question is testing in a dementia assessment, "
            f"and any watch-out signs for the invigilator:\n\"{question_text}\""
        ),
        retrieved_docs=retrieved,
        image_store=store.image_store,
        domain="dementia assessment invigilator guidance",
    )

    system = SystemMessage(content="You are a clinical neuropsychologist. Be brief and practical.")
    response = llm.invoke([system, message])
    return response.content.strip()


# ─── Vapi Payload Builder ─────────────────────────────────────────────────────

def build_vapi_call_payload(
    patient_name: str,
    assessment_questions: list[dict],
    vapi_phone_number_id: str,
    patient_phone: str,
    assistant_id: Optional[str] = None,
) -> dict:
    """
    Build the full Vapi /call/create payload for an outbound assessment call.
    The Android backend POSTs this to https://api.vapi.ai/call/outbound.

    Args:
        patient_name:           Patient's first name.
        assessment_questions:   List of MCQ dicts (from assessment_generator).
        vapi_phone_number_id:   Vapi phone number ID (from Vapi dashboard).
        patient_phone:          Patient's phone number (E.164 format).
        assistant_id:           Optional Vapi assistant ID if pre-configured.
    """
    system_prompt = build_vapi_context(save=False)
    system_prompt = system_prompt.replace("their first name", patient_name)

    # Serialize questions into a structured string for the agent
    q_block = "\n\n".join(
        f"Q{i+1} [{q['domain']}]: {q['question']}\n"
        + "\n".join(f"  {chr(65+j)}) {opt}" for j, opt in enumerate(q["options"]))
        + f"\n[Voice]: {q.get('voice_text', q['question'])}"
        for i, q in enumerate(assessment_questions)
    )

    full_prompt = f"{system_prompt}\n\nASSESSMENT QUESTIONS:\n{q_block}"

    payload = {
        "phoneNumberId": vapi_phone_number_id,
        "customer": {"number": patient_phone, "name": patient_name},
        "assistant": {
            "model": {
                "provider": "openai",
                "model": "gpt-4o",
                "systemPrompt": full_prompt,
            },
            "voice": {
                "provider": "11labs",
                "voiceId": "rachel",   # Calm, clear female voice
            },
            "firstMessage": (
                f"Hello {patient_name}, this is ARIA, your health assistant. "
                "I'd like to ask you a few questions today. "
                "Please take your time and answer as best you can. Ready to start?"
            ),
        },
    }

    if assistant_id:
        payload["assistantId"] = assistant_id

    return payload


if __name__ == "__main__":
    prompt = build_vapi_context(save=True)
    print("─── Vapi System Prompt ───")
    print(prompt[:1000])