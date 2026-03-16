"""
Dementia App - LLM Factory
===========================
Single place that decides which LLM to use based on environment variables.

Usage in any file:
    from llm_factory import get_llm
    llm = get_llm()

Control via environment variables:
    set LLM_BACKEND=openai    → uses OpenAI API (default, needs OPENAI_API_KEY)
    set LLM_BACKEND=ollama    → uses local Ollama (free, offline, no key needed)

No other file needs to change when you switch LLM provider.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Default is now OpenAI
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai").lower()


def get_llm():
    """
    Return the configured LLM instance.
    Controlled by LLM_BACKEND environment variable.
    """

    # ── OpenAI (default) ──────────────────────────────────────────────────────
    if LLM_BACKEND == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Run:\n"
                "  pip install langchain-openai"
            )

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set.\n"
                "Get your key at: https://platform.openai.com/api-keys\n"
                "Then set it:  set OPENAI_API_KEY=sk-..."
            )

        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        logger.info(f"LLM: OpenAI ({model})")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0,        # deterministic output — important for JSON generation
            max_tokens=4096,
        )

    # ── Ollama (local fallback, no API key needed) ────────────────────────────
    elif LLM_BACKEND == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "Ollama package not installed. Run:\n"
                "  pip install langchain-ollama"
            )

        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        logger.info(f"LLM: Ollama ({model}) — local")
        return ChatOllama(model=model)

    else:
        raise ValueError(
            f"Unknown LLM_BACKEND='{LLM_BACKEND}'.\n"
            "Supported values: openai, ollama"
        )