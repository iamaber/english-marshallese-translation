import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# File paths
PROJECT_ROOT = Path(__file__).parent
VOCAB_FILE_PATH = PROJECT_ROOT / os.getenv("VOCAB_FILE_PATH", "Data/marshallese.json")

import warnings

if not OPENAI_API_KEY:
    warnings.warn("OPENAI_API_KEY not found in environment variables. LLM features will be disabled until provided.")

if not VOCAB_FILE_PATH.exists():
    warnings.warn(f"Vocabulary file not found at {VOCAB_FILE_PATH}. Fallback translator may not work.")


def is_llm_configured() -> bool:
    """Return True when an OpenAI/GPT API key is present and a model is configured."""
    return bool(OPENAI_API_KEY and MODEL_NAME)


def ensure_vocab_path() -> Path:
    """Return the resolved vocabulary path. Raises FileNotFoundError only when called explicitly.

    Use this helper when code needs to be strict about the presence of the JSON file.
    """
    if VOCAB_FILE_PATH.exists():
        return VOCAB_FILE_PATH
    raise FileNotFoundError(f"Vocabulary file not found at {VOCAB_FILE_PATH}")
