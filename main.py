"""
Dementia App - RAG System Entrypoint
=====================================
CLI tool for:
  - Ingesting PDFs (admin workflow / FR-204)
  - Generating assessment questions (FR-201)
  - Building Vapi voice agent context (FR-202, FR-205)
  - Running evaluation
  - Interactive query mode (dev/testing)

Usage:
    python main.py ingest --pdf clinical_guidelines.pdf
    python main.py generate --domains Memory,Stress --questions 20
    python main.py vapi
    python main.py evaluate
    python main.py query --q "What are early dementia symptoms?"
    python main.py serve          # starts FastAPI admin server on :8000
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
key = os.environ.get("OPENAI_API_KEY")

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def cmd_ingest(args):
    from rag_pipeline import get_store
    store = get_store()
    store.ingest_pdf(args.pdf)
    print(f"✅ PDF ingested: {args.pdf}")
    print(f"   Images indexed: {len(store.image_store)}")


def cmd_generate(args):
    from assessment_generator import generate_full_assessment, ASSESSMENT_DOMAINS
    domains = [d.strip() for d in args.domains.split(",")] if args.domains else ASSESSMENT_DOMAINS
    assessment = generate_full_assessment(
        domains=domains,
        questions_per_domain=args.questions,
        save=True,
    )
    total = sum(len(v) for v in assessment.values())
    print(f"✅ Generated {total} questions across {len(assessment)} domains.")
    print("   Saved to output/questions/")


def cmd_vapi(args):
    from voice_agent_context import build_vapi_context
    prompt = build_vapi_context(save=True)
    print("✅ Vapi system prompt built and saved to output/vapi_system_prompt.txt")
    print("\n--- Preview (first 500 chars) ---")
    print(prompt[:500])


def cmd_evaluate(args):
    from evaluation import run_full_evaluation
    summary = run_full_evaluation(k=args.k)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


def cmd_query(args):
    """Interactive retrieval + generation for dev/testing."""
    from rag_pipeline import get_store, build_multimodal_message
    from llm_factory import get_llm

    store = get_store()
    llm = get_llm()

    query = args.q or input("Enter query: ")
    retrieved = store.retrieve(query, k=args.k)

    print(f"\n📄 Retrieved {len(retrieved)} documents:")
    for i, d in enumerate(retrieved):
        dtype = d.metadata.get("type", "?")
        page = d.metadata.get("page", "?")
        preview = d.page_content[:80] if dtype == "text" else f"[image: {d.page_content}]"
        print(f"  {i+1}. [{dtype}][page {page}] {preview}")

    message = build_multimodal_message(query, retrieved, store.image_store)
    response = llm.invoke([message])
    print(f"\n🤖 Answer:\n{response.content}")


def cmd_serve(args):
    import uvicorn
    uvicorn.run(
        "admin_upload:app",
        host="0.0.0.0",
        port=args.port,
        reload=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Dementia App RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a PDF into the RAG index")
    p_ingest.add_argument("--pdf", required=True, help="Path to PDF file")
    p_ingest.set_defaults(func=cmd_ingest)

    # generate
    p_gen = sub.add_parser("generate", help="Generate MCQ assessment questions")
    p_gen.add_argument("--domains", default=None, help="Comma-separated domain list")
    p_gen.add_argument("--questions", type=int, default=20, help="Questions per domain")
    p_gen.set_defaults(func=cmd_generate)

    # vapi
    p_vapi = sub.add_parser("vapi", help="Build Vapi voice agent system prompt")
    p_vapi.set_defaults(func=cmd_vapi)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run evaluation suite")
    p_eval.add_argument("--k", type=int, default=5, help="Top-k for retrieval eval")
    p_eval.set_defaults(func=cmd_evaluate)

    # query
    p_query = sub.add_parser("query", help="Interactive RAG query (dev/testing)")
    p_query.add_argument("--q", default=None, help="Query string")
    p_query.add_argument("--k", type=int, default=5)
    p_query.set_defaults(func=cmd_query)

    # serve
    p_serve = sub.add_parser("serve", help="Start FastAPI admin server")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()