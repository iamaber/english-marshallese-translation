from fastapi import FastAPI, Query
from pydantic import BaseModel
import openai
import asyncio
from typing import Optional, List, Dict
import json
import os
import re
import difflib
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()


def load_data():
    with open("Data/marshallese.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    eng_to_mar = {}
    mar_to_eng = {}

    # General vocabulary
    for section in ["greetings", "common_phrases"]:
        for item in data["general_vocabulary"][section]:
            eng = item["english"].lower()
            mar = item["marshallese"].lower()
            if eng not in eng_to_mar:
                eng_to_mar[eng] = []
            eng_to_mar[eng].append(mar)
            if mar not in mar_to_eng:
                mar_to_eng[mar] = []
            mar_to_eng[mar].append(eng)

    # Body parts
    for item in data["body_parts"]:
        eng = item["english"].lower()
        if "marshallese" in item:
            mar = item["marshallese"].lower()
        elif "base" in item:
            mar = item["base"].lower()
        else:
            continue
        if eng not in eng_to_mar:
            eng_to_mar[eng] = []
        eng_to_mar[eng].append(mar)
        if mar not in mar_to_eng:
            mar_to_eng[mar] = []
        mar_to_eng[mar].append(eng)
        # Add variations
        for key in ["my", "your", "their"]:
            if key in item:
                mar_var = item[key].lower()
                if mar_var not in mar_to_eng:
                    mar_to_eng[mar_var] = []
                mar_to_eng[mar_var].append(eng)

    # Medical conditions
    for item in data["medical_conditions"]:
        eng = item["english"].lower()
        mar = item["marshallese"].lower()
        if eng not in eng_to_mar:
            eng_to_mar[eng] = []
        eng_to_mar[eng].append(mar)
        if mar not in mar_to_eng:
            mar_to_eng[mar] = []
        mar_to_eng[mar].append(eng)

    # Pain and symptoms
    for item in data["pain_and_symptoms"]["descriptors"]:
        eng = item["english"].lower()
        mar = item.get("marshallese", "").strip()
        if mar:
            mar = mar.lower()
            if eng not in eng_to_mar:
                eng_to_mar[eng] = []
            eng_to_mar[eng].append(mar)
            if mar not in mar_to_eng:
                mar_to_eng[mar] = []
            mar_to_eng[mar].append(eng)

    for item in data["pain_and_symptoms"]["questions"]:
        eng = item["english"].lower()
        mar = item["marshallese"].lower()
        if eng not in eng_to_mar:
            eng_to_mar[eng] = []
        eng_to_mar[eng].append(mar)
        if mar not in mar_to_eng:
            mar_to_eng[mar] = []
        mar_to_eng[mar].append(eng)

    # Review of systems
    for category, items in data["review_of_systems"].items():
        if isinstance(items, list):
            for item in items:
                eng = item["english"].lower()
                mar = item.get("marshallese", "").strip()
                if mar:
                    mar = mar.lower()
                    if eng not in eng_to_mar:
                        eng_to_mar[eng] = []
                    eng_to_mar[eng].append(mar)
                    if mar not in mar_to_eng:
                        mar_to_eng[mar] = []
                    mar_to_eng[mar].append(eng)

    # Physical examination
    for item in data["physical_examination"]:
        eng = item["english"].lower()
        mar = item["marshallese"].lower()
        if eng not in eng_to_mar:
            eng_to_mar[eng] = []
        eng_to_mar[eng].append(mar)
        if mar not in mar_to_eng:
            mar_to_eng[mar] = []
        mar_to_eng[mar].append(eng)

    return eng_to_mar, mar_to_eng


eng_to_mar, mar_to_eng = load_data()


# ==== API Models and Endpoints  ====


class TranslationResponse(BaseModel):
    translation: str
    context: Optional[str] = None
    found_words: Optional[List[str]] = None
    rag_used: bool = True


async def fuzzy_search(
    word: str, dictionary: dict, threshold: float = 0.8
) -> Optional[str]:
    matches = difflib.get_close_matches(
        word.lower(), [k.lower() for k in dictionary.keys()], n=1, cutoff=threshold
    )
    if matches:
        key = next(k for k in dictionary if k.lower() == matches[0])
        return dictionary[key]
    return None


async def retrieve_translations(words: List[str], source_dict: dict) -> Dict[str, str]:
    retrieved = {}
    for word in words:
        match = await fuzzy_search(word, source_dict)
        if match:
            retrieved[word] = match
    return retrieved


async def ai_translate(
    text: str,
    from_lang: str,
    to_lang: str,
    retrieved: Dict[str, str],
    temperature: float = 0.0,
) -> str:
    context_str = "\n".join([f"{word}: {trans}" for word, trans in retrieved.items()])
    prompt = f"""Translate the following text from {from_lang} to {to_lang}.
Text: {text}

Use this dictionary context for known words:
{context_str}

If only some words are known, make sense of the full sentence by inferring from context and known translations. Provide a coherent translation.
Only provide the translated text without additional explanations.
"""
    response = await asyncio.to_thread(
        openai.chat.completions.create,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


async def ai_context(text: str, from_lang: str, max_tokens: int = 300) -> str:
    prompt = f"Provide context or explanation for the text '{text}' in {from_lang}. Limit to 200-500 tokens."
    response = await asyncio.to_thread(
        openai.chat.completions.create,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


@app.get("/translate", response_model=TranslationResponse)
async def translate(
    text: str = Query(..., description="Text (word or sentence) to translate"),
    from_lang: str = Query("english"),
    to_lang: str = Query("marshallese"),
    provide_context: bool = Query(False, description="Whether to provide context"),
):
    from_lang = from_lang.lower()
    to_lang = to_lang.lower()
    temperature = 0.3

    if from_lang == "english" and to_lang == "marshallese":
        source_dict = eng_to_mar
    elif from_lang == "marshallese" and to_lang == "english":
        source_dict = mar_to_eng
    else:
        raise ValueError(
            "Unsupported language pair. Use english to marshallese or vice versa."
        )

    # Split into words for retrieval
    words = re.findall(r"\b\w+\b", text.lower())

    # Retrieve translations for known words
    retrieved = await retrieve_translations(words, source_dict)
    found_words = list(retrieved.keys())
    rag_used = bool(retrieved)

    if len(found_words) == len(words):
        # All words found: compose translation directly
        translations = []
        for word in words:
            trans = retrieved.get(word)
            if isinstance(trans, list):
                translations.append(", ".join(trans))
            else:
                translations.append(trans or word)  # Fallback to original if missing
        translation = " ".join(
            translations
        )  # Simple join; improve for grammar if needed
    else:
        # Partial or no matches: use AI with RAG
        translation = await ai_translate(
            text,
            from_lang.capitalize(),
            to_lang.capitalize(),
            retrieved,
            temperature,
        )

    context = None
    if provide_context:
        context = await ai_context(text, from_lang.capitalize())

    return TranslationResponse(
        translation=translation,
        context=context,
        found_words=found_words,
        rag_used=rag_used,
    )
