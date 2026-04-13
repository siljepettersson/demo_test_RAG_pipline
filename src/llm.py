"""
LLM answer generation for the agency client knowledge base demo.

This module exposes a single ``generate_answer`` entry point that dispatches
to one of three providers (Anthropic, OpenAI, MiniMax). Each provider reads
its API key from the environment so the app stays provider-independent.
"""

from __future__ import annotations

import argparse
import os
from typing import Callable

import httpx
from dotenv import load_dotenv
from langchain_core.documents import Document


load_dotenv()


SYSTEM_PROMPT = (
    "Du er en assistent som hjelper ansatte i et markedsføringsbyrå "
    "med å finne informasjon om byråets kunder. "
    "Svar kort, presist og på norsk, og kun basert på kontekstutdragene nedenfor. "
    "Hvis konteksten ikke inneholder svaret, si tydelig at du ikke vet. "
    "Ikke dikt opp tall, navn eller datoer. "
    "Referer til kildene du bruker som (Kilde 1), (Kilde 2), osv."
)

ANSWER_MAX_TOKENS = 500
ANSWER_TEMPERATURE = 0.1

ANTHROPIC_MODEL = "claude-haiku-4-5"
OPENAI_MODEL = "gpt-4o-mini"

# MiniMax endpoint and model. The exact values may drift; verify against the
# current MiniMax API docs before demoing with this provider.
MINIMAX_API_URL = "https://api.minimax.io/v1/text/chatcompletion_v2"
MINIMAX_MODEL = "MiniMax-Text-01"

PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}


def format_context(results: list[Document]) -> str:
    """Format retrieved chunks as numbered context blocks for the LLM prompt."""
    if not results:
        return "(ingen kontekst tilgjengelig)"

    blocks: list[str] = []
    for index, document in enumerate(results, start=1):
        metadata = document.metadata
        title = metadata.get("document_title") or metadata.get("source") or "ukjent"
        client = metadata.get("client", "ukjent")
        source = metadata.get("source", "ukjent")
        blocks.append(
            f"[Kilde {index}] {title} — kunde: {client} — kilde: {source}\n"
            f"{document.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, results: list[Document]) -> str:
    """Combine the user question and retrieved context into one prompt string."""
    context = format_context(results)
    return (
        f"Spørsmål: {question}\n\n"
        f"Kontekst:\n{context}\n\n"
        "Svar kort og presist, og referer til kildene du bruker."
    )


def require_api_key(provider: str) -> str:
    """Read the required API key for a provider or raise a clear error."""
    env_key = PROVIDER_ENV_KEYS[provider]
    api_key = os.environ.get(env_key)
    if not api_key:
        raise RuntimeError(
            f"{env_key} is not set. Add it to .env to use the '{provider}' provider."
        )
    return api_key


def generate_with_anthropic(question: str, results: list[Document]) -> str:
    """Generate an answer using the Anthropic Claude API."""
    from anthropic import Anthropic

    client = Anthropic(api_key=require_api_key("anthropic"))
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=ANSWER_MAX_TOKENS,
        temperature=ANSWER_TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": build_user_prompt(question, results)},
        ],
    )
    return response.content[0].text.strip()


def generate_with_openai(question: str, results: list[Document]) -> str:
    """Generate an answer using the OpenAI Chat Completions API."""
    from openai import OpenAI

    client = OpenAI(api_key=require_api_key("openai"))
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=ANSWER_TEMPERATURE,
        max_tokens=ANSWER_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, results)},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def generate_with_minimax(question: str, results: list[Document]) -> str:
    """Generate an answer using the MiniMax chat completion HTTP API."""
    api_key = require_api_key("minimax")
    payload = {
        "model": MINIMAX_MODEL,
        "temperature": ANSWER_TEMPERATURE,
        "max_tokens": ANSWER_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, results)},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = httpx.post(
        MINIMAX_API_URL, headers=headers, json=payload, timeout=30.0
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


PROVIDERS: dict[str, Callable[[str, list[Document]], str]] = {
    "anthropic": generate_with_anthropic,
    "openai": generate_with_openai,
    "minimax": generate_with_minimax,
}


def generate_answer(
    provider: str, question: str, results: list[Document]
) -> str:
    """Dispatch answer generation to the chosen LLM provider."""
    provider_key = provider.lower()
    if provider_key not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Available: {sorted(PROVIDERS)}."
        )
    return PROVIDERS[provider_key](question, results)


def available_providers() -> list[str]:
    """Return providers whose API keys are currently set in the environment."""
    return [
        provider
        for provider, env_key in PROVIDER_ENV_KEYS.items()
        if os.environ.get(env_key)
    ]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a standalone provider smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDERS),
        default="anthropic",
        help="Which LLM provider to call.",
    )
    parser.add_argument(
        "--question",
        default="Hva er Fjordmats tone of voice?",
        help="Question to answer.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of chunks to retrieve as context.",
    )
    return parser.parse_args()


def _run_cli() -> None:
    """Run a small end-to-end smoke test against a chosen provider."""
    from src.rag_pipeline import (
        get_embeddings,
        load_vectorstore,
        query_vectorstore,
    )

    args = parse_args()
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings)
    results = query_vectorstore(vectorstore, args.question, k=args.k)

    print(f"\nQ: {args.question}")
    print(f"Provider: {args.provider}")
    print(f"Retrieved {len(results)} chunks.")
    print("\n--- Answer ---")
    print(generate_answer(args.provider, args.question, results))


if __name__ == "__main__":
    _run_cli()
