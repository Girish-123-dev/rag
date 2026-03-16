"""
Dementia App - RAG Evaluation Script
======================================
Evaluates:
1. Retrieval quality — does the RAG return relevant clinical context?
2. Question quality — are generated MCQ questions clinically appropriate?
3. Clarification quality — are clarifications simple and accurate?

Adapted from the original VQA evaluation pipeline (evaluationscript.py).
"""

import json
import logging
import os
from difflib import SequenceMatcher
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from rag_pipeline import embed_text, get_store
from assessment_generator import clarify_question

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OUTPUT_DIR = "output/evaluation"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def soft_match(a: str, b: str) -> float:
    """Fuzzy string similarity (0–1)."""
    a = ("" if a is None else a).strip().lower()
    b = ("" if b is None else b).strip().lower()
    return SequenceMatcher(None, a, b).ratio()


def semantic_similarity(text_a: str, text_b: str) -> float:
    """CLIP-based cosine similarity between two text strings."""
    emb_a = embed_text(text_a).reshape(1, -1)
    emb_b = embed_text(text_b).reshape(1, -1)
    return float(cosine_similarity(emb_a, emb_b)[0][0])


# ─── Retrieval Evaluation ─────────────────────────────────────────────────────

def evaluate_retrieval(test_queries: list[dict], k: int = 5) -> pd.DataFrame:
    """
    Evaluate retrieval quality using a set of query–expected_content pairs.

    test_queries format:
    [
        {
            "query": "what are the early signs of dementia",
            "expected_keyword": "memory loss"   # must appear in retrieved content
        },
        ...
    ]

    Metrics per query:
    - keyword_hit     : does any retrieved doc contain the expected keyword?
    - avg_semantic_sim: average CLIP cosine similarity between query and retrieved docs
    - recall_at_k     : binary (1 if any relevant doc in top-k)
    """
    store = get_store()
    records = []

    for item in test_queries:
        query = item["query"]
        expected = item.get("expected_keyword", "").lower()

        retrieved = store.retrieve(query, k=k)
        all_text = " ".join(
            d.page_content for d in retrieved if d.metadata.get("type") == "text"
        ).lower()

        keyword_hit = int(expected in all_text) if expected else 0

        # Semantic similarity of query to each retrieved text doc
        sims = []
        for d in retrieved:
            if d.metadata.get("type") == "text":
                sims.append(semantic_similarity(query, d.page_content))
        avg_sim = float(np.mean(sims)) if sims else 0.0

        records.append(
            {
                "query": query,
                "expected_keyword": expected,
                "keyword_hit": keyword_hit,
                "avg_semantic_similarity": round(avg_sim, 4),
                "recall_at_k": keyword_hit,
                "num_retrieved": len(retrieved),
            }
        )

    df = pd.DataFrame(records)
    logger.info("\n=== Retrieval Evaluation ===")
    logger.info(f"Recall@{k}:              {df['recall_at_k'].mean():.2%}")
    logger.info(f"Avg Semantic Similarity: {df['avg_semantic_similarity'].mean():.4f}")
    logger.info(f"Keyword Hit Rate:        {df['keyword_hit'].mean():.2%}")
    return df


# ─── Question Quality Evaluation ──────────────────────────────────────────────

def evaluate_questions(questions_path: str = "output/questions/full_assessment.json") -> pd.DataFrame:
    """
    Automatically evaluate generated MCQ questions for:
    - completeness  : question + 4 options + voice_text present
    - readability   : average word length as proxy (lower = simpler)
    - domain_match  : semantic similarity of question to its domain label
    """
    with open(questions_path) as f:
        assessment = json.load(f)

    records = []
    for domain, questions in assessment.items():
        for q in questions:
            text = q.get("question", "")
            options = q.get("options", [])
            voice = q.get("voice_text", "")

            complete = int(bool(text) and len(options) == 4 and bool(voice))
            words = text.split()
            avg_word_len = round(np.mean([len(w) for w in words]) if words else 0, 2)
            domain_sim = semantic_similarity(text, f"{domain} dementia assessment")

            records.append(
                {
                    "domain": domain,
                    "question_id": q.get("id", ""),
                    "question": text[:80],
                    "complete": complete,
                    "avg_word_length": avg_word_len,
                    "domain_semantic_sim": round(domain_sim, 4),
                    "difficulty": q.get("difficulty", ""),
                }
            )

    df = pd.DataFrame(records)
    logger.info("\n=== Question Quality Evaluation ===")
    logger.info(f"Completeness:         {df['complete'].mean():.2%}")
    logger.info(f"Avg Word Length:      {df['avg_word_length'].mean():.2f} chars")
    logger.info(f"Domain Alignment:     {df['domain_semantic_sim'].mean():.4f}")
    return df


# ─── Clarification Evaluation ─────────────────────────────────────────────────

def evaluate_clarifications(test_cases: list[dict]) -> pd.DataFrame:
    """
    Evaluate SP-304 clarification responses.

    test_cases format:
    [
        {
            "question": "How often do you feel sad?",
            "patient_query": "What do you mean by sad?",
            "expected_theme": "emotions"
        },
        ...
    ]
    """
    records = []
    for case in test_cases:
        clarification = clarify_question(case["question"], case["patient_query"])
        word_count = len(clarification.split())
        theme_sim = semantic_similarity(clarification, case.get("expected_theme", case["question"]))

        records.append(
            {
                "question": case["question"][:60],
                "clarification": clarification[:120],
                "word_count": word_count,
                "is_concise": int(word_count <= 60),
                "theme_similarity": round(theme_sim, 4),
            }
        )

    df = pd.DataFrame(records)
    logger.info("\n=== Clarification Evaluation ===")
    logger.info(f"Conciseness (<60 words): {df['is_concise'].mean():.2%}")
    logger.info(f"Theme Similarity:        {df['theme_similarity'].mean():.4f}")
    return df


# ─── Full Evaluation Suite ────────────────────────────────────────────────────

def run_full_evaluation(
    retrieval_test_file: Optional[str] = None,
    clarification_test_file: Optional[str] = None,
    k: int = 5,
):
    """Run all evaluation suites and save results to output/evaluation/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Default test data if no files provided
    retrieval_tests = [
        {"query": "early signs of dementia", "expected_keyword": "memory"},
        {"query": "sleep problems in elderly patients", "expected_keyword": "sleep"},
        {"query": "daily activity tracking for dementia", "expected_keyword": "activity"},
        {"query": "stress assessment for cognitive decline", "expected_keyword": "stress"},
    ]
    if retrieval_test_file and os.path.exists(retrieval_test_file):
        with open(retrieval_test_file) as f:
            retrieval_tests = json.load(f)

    clarification_tests = [
        {
            "question": "How often do you feel overwhelmed by daily tasks?",
            "patient_query": "What does overwhelmed mean?",
            "expected_theme": "stress difficulty",
        },
        {
            "question": "How many hours of sleep did you get last night?",
            "patient_query": "I don't understand",
            "expected_theme": "sleep hours",
        },
    ]
    if clarification_test_file and os.path.exists(clarification_test_file):
        with open(clarification_test_file) as f:
            clarification_tests = json.load(f)

    # Run evaluations
    retrieval_df = evaluate_retrieval(retrieval_tests, k=k)
    retrieval_df.to_csv(f"{OUTPUT_DIR}/retrieval_eval.csv", index=False)

    question_df = None
    if os.path.exists("output/questions/full_assessment.json"):
        question_df = evaluate_questions()
        question_df.to_csv(f"{OUTPUT_DIR}/question_quality_eval.csv", index=False)
    else:
        logger.warning("No assessment questions found. Skipping question quality eval.")

    clarification_df = evaluate_clarifications(clarification_tests)
    clarification_df.to_csv(f"{OUTPUT_DIR}/clarification_eval.csv", index=False)

    # Summary
    summary = {
        "retrieval_recall_at_k": float(retrieval_df["recall_at_k"].mean()),
        "retrieval_avg_semantic_sim": float(retrieval_df["avg_semantic_similarity"].mean()),
        "question_completeness": float(question_df["complete"].mean()) if question_df is not None else None,
        "clarification_conciseness": float(clarification_df["is_concise"].mean()),
    }
    with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Evaluation complete. Results saved to output/evaluation/")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_full_evaluation()
