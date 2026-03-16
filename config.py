# config.py
from pathlib import Path
import os
from dotenv import load_dotenv
from pydantic import SecretStr

# Load .env from project root
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# ── OpenAI ────────────────────────────────────────────────────────────────────
_openai_key = os.getenv("OPENAI_API_KEY")
if not _openai_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found.\n"
        "Add it to your .env file:  OPENAI_API_KEY=sk-...\n"
        "Get a key at: https://platform.openai.com/api-keys"
    )

# Exposed as SecretStr to satisfy Pydantic / Pylance type checks
OPENAI_API_KEY: SecretStr = SecretStr(_openai_key)