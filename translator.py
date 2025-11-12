import json
import re
import time
from typing import Dict
from pathlib import Path
import warnings

from config import (
    OPENAI_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    VOCAB_FILE_PATH,
    is_llm_configured,
)
from prompts import (
    EN_TO_MH_SYSTEM_PROMPT,
    MH_TO_EN_SYSTEM_PROMPT,
    EN_TO_MH_USER_PROMPT,
    MH_TO_EN_USER_PROMPT,
)


def _load_vocab_dict(vocab_path: Path) -> Dict[str, str]:
    """Load the JSON vocabulary and return a mapping english_lower -> marshallese."""
    start_time = time.time()
    if not Path(vocab_path).exists():
        warnings.warn(
            f"Vocab file {vocab_path} not found; fallback translator will be empty."
        )
        return {}

    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs: Dict[str, str] = {}

    def extract(obj):
        if isinstance(obj, dict):
            # common pair keys
            if "english" in obj and (
                "marshallese" in obj or "base" in obj or "my" in obj
            ):
                en = obj.get("english", "").strip()
                # prefer an explicit 'marshallese' value, then 'base', then 'my'
                mh = obj.get("marshallese") or obj.get("base") or obj.get("my")
                if en and mh:
                    pairs[en.lower()] = mh
            # also check for body_parts entries with 'base' etc.
            for v in obj.values():
                extract(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item)

    extract(data)
    end_time = time.time()
    print(f"_load_vocab_dict took {end_time - start_time:.4f} seconds")
    return pairs


def _longest_phrase_replace(text: str, mapping: Dict[str, str]) -> str:
    """Replace phrases from mapping into text using longest-first replacement (case-insensitive)."""
    if not mapping:
        return text

    start_time = time.time()
    # Build regex alternation with longest-first to match multi-word phrases
    phrases = sorted(mapping.keys(), key=lambda s: -len(s))
    # Escape regex meta characters in phrases
    escaped = [re.escape(p) for p in phrases]
    pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)
    compile_time = time.time()
    print(f"Regex compilation took {compile_time - start_time:.4f} seconds")

    def repl(m):
        matched = m.group(0)
        # lookup by lowercased key
        return mapping.get(matched.lower(), matched)

    result = pattern.sub(repl, text)
    end_time = time.time()
    print(f"Pattern sub took {end_time - compile_time:.4f} seconds")
    return result


def _try_use_llm(prompt_system: str, prompt_user: str, text: str) -> str:
    """Attempt to use LLM if configured. Returns None-like string on failure."""
    if not is_llm_configured():
        return None

    try:
        # Lazy import so that projects without langchain won't fail at import-time
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
    except Exception as e:
        warnings.warn(f"LLM imports failed: {e}. Falling back to local translator.")
        return None

    llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=TEMPERATURE)
    context = json.dumps(
        _load_vocab_dict(VOCAB_FILE_PATH), indent=2, ensure_ascii=False
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_system),
            ("user", prompt_user),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    try:
        start_invoke = time.time()
        result = chain.invoke({"context": context, "text": text})
        end_invoke = time.time()
        print(f"LLM invoke took {end_invoke - start_invoke:.4f} seconds")
        return result.strip()
    except Exception as e:
        warnings.warn(f"LLM invocation failed: {e}. Falling back to local translator.")
        return None


def english_to_marshallese(text: str) -> str:
    """Translate English to Marshallese.

    If an LLM is configured, attempt to use it. Otherwise use a simple phrase-mapping
    fallback built from `Data/marshallese.json`.
    """
    # Try LLM first when configured
    llm_result = _try_use_llm(EN_TO_MH_SYSTEM_PROMPT, EN_TO_MH_USER_PROMPT, text)
    if llm_result:
        return llm_result

    mapping = _load_vocab_dict(VOCAB_FILE_PATH)
    # Use longest-phrase replacement to preserve multi-word entries
    return _longest_phrase_replace(text, mapping)


def marshallese_to_english(text: str) -> str:
    """Translate Marshallese to English using fallback inversion or LLM when available."""
    llm_result = _try_use_llm(MH_TO_EN_SYSTEM_PROMPT, MH_TO_EN_USER_PROMPT, text)
    if llm_result:
        return llm_result

    mapping = _load_vocab_dict(VOCAB_FILE_PATH)
    # invert mapping, but if multiple English strings map to same Marshallese, keep first
    start_invert = time.time()
    inverted: Dict[str, str] = {}
    for en, mh in mapping.items():
        if mh and mh.lower() not in inverted:
            inverted[mh.lower()] = en
    end_invert = time.time()
    print(f"Inverting mapping took {end_invert - start_invert:.4f} seconds")

    return _longest_phrase_replace(text, inverted)
