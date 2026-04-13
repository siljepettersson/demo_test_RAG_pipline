"""Streamlit MVP for the agency client knowledge base demo."""

from __future__ import annotations

from typing import Any

import streamlit as st
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.rag_pipeline import (
    DATA_DIR,
    build_vectorstore,
    get_embeddings,
    load_vectorstore,
    query_vectorstore,
)


APP_TITLE = "Kundeinnsikt"
APP_DESCRIPTION = (
    "Still et spørsmål om en kunde, og få et kort svar med tydelige kilder."
)
CLIENT_OPTIONS = [
    ("Alle kunder", None),
    ("Fjordmat", "fjordmat"),
    ("Spareklar", "spareklar"),
    ("Nordvik", "nordvik"),
    ("Skytjenester AS", "skytjenester"),
]
DOCUMENT_TYPE_LABELS = {
    "brand_guidelines": "Merkevareretningslinjer",
    "campaign_brief": "Kampanjebrief",
    "campaign_report": "Kampanjerapport",
    "meeting_notes": "Møtereferat",
    "social_media_strategy": "Sosiale medier-strategi",
    "influencer_strategy": "Influencer-strategi",
    "customer_case": "Kundecase",
    "seo_report": "SEO-rapport",
    "other": "Annet dokument",
}


def format_document_type(document_type: str) -> str:
    """Convert internal document type keys into UI-friendly Norwegian labels."""
    return DOCUMENT_TYPE_LABELS.get(document_type, document_type)


def snippet_from_chunk(content: str, limit: int = 240) -> str:
    """Create a compact snippet from retrieved chunk text."""
    single_line = " ".join(content.split())
    if len(single_line) <= limit:
        return single_line
    return f"{single_line[: limit - 3].rstrip()}..."


@st.cache_resource(show_spinner=False)
def get_vectorstore() -> Chroma:
    """Load the existing vector store or build it if missing."""
    embeddings = get_embeddings()
    try:
        return load_vectorstore(embeddings)
    except FileNotFoundError:
        return build_vectorstore()


def rebuild_vectorstore() -> Chroma:
    """Rebuild the persisted vector store and refresh the cache."""
    get_vectorstore.clear()
    return get_vectorstore()


def answer_from_results(question: str, results: list[Document]) -> str:
    """Create a short grounded answer from retrieved chunks without an external LLM."""
    if not results:
        return (
            "Jeg fant ikke et tydelig svar i kunnskapsbasen. "
            "Prøv å være mer spesifikk eller velg en kunde i sidepanelet."
        )

    top_result = results[0]
    title = top_result.metadata["document_title"]
    snippet = snippet_from_chunk(top_result.page_content, limit=260)
    return f"Basert på {title}: {snippet}"


def render_source_card(document: Document, index: int) -> None:
    """Render one source card with metadata and optional chunk text."""
    metadata = document.metadata
    with st.container(border=True):
        st.markdown(f"**Kilde {index}: {metadata['document_title']}**")
        st.caption(
            " | ".join(
                [
                    metadata["client"].capitalize(),
                    format_document_type(metadata["document_type"]),
                    metadata["source"],
                ]
            )
        )
        st.write(snippet_from_chunk(document.page_content))
        with st.expander("Vis utdrag"):
            st.write(document.page_content)


def render_assistant_message(message: dict[str, Any]) -> None:
    """Render one assistant response with answer and source cards."""
    st.write(message["answer"])
    if not message["sources"]:
        return
    st.markdown("**Kilder**")
    for index, document in enumerate(message["sources"], start=1):
        render_source_card(document, index)


def ensure_chat_state() -> None:
    """Initialize chat state once per session."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    """Render the Streamlit MVP."""
    st.set_page_config(page_title=APP_TITLE, page_icon=":speech_balloon:", layout="wide")
    ensure_chat_state()

    st.title(APP_TITLE)
    st.caption(APP_DESCRIPTION)

    with st.sidebar:
        st.header("Innstillinger")
        client_label = st.selectbox(
            "Kunde",
            options=[label for label, _ in CLIENT_OPTIONS],
            index=0,
        )
        selected_client = dict(CLIENT_OPTIONS)[client_label]
        top_k = st.slider("Antall kilder", min_value=2, max_value=5, value=3, step=1)
        if st.button("Bygg kunnskapsbase på nytt", use_container_width=True):
            with st.spinner("Oppdaterer vektorindeks..."):
                rebuild_vectorstore()
            st.success("Kunnskapsbasen er oppdatert.")

        st.markdown("**Eksempler**")
        st.caption(
            "\n".join(
                [
                    "- Hva er Fjordmats tone of voice?",
                    "- Hva var ROAS for Spareklars Google Ads i Q4 2024?",
                    "- Hvilke influencere samarbeider Nordvik med?",
                ]
            )
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                render_assistant_message(message)

    prompt = st.chat_input("Spør om en kunde, kampanje eller rapport")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Henter relevant kundekunnskap..."):
            vectorstore = get_vectorstore()
            results = query_vectorstore(
                vectorstore,
                prompt,
                k=top_k,
                client=selected_client,
            )
            answer = answer_from_results(prompt, results)

        st.write(answer)
        if results:
            st.markdown("**Kilder**")
            for index, document in enumerate(results, start=1):
                render_source_card(document, index)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "answer": answer,
            "sources": results,
        }
    )


if __name__ == "__main__":
    main()
