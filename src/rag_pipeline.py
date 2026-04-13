"""
RAG pipeline for the agency client knowledge base demo.

This module loads Norwegian client documents from ``data/``, chunks them,
embeds them locally, and stores them in Chroma for retrieval.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "vectorstore"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "agency_knowledge_base"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
REQUIRED_DOCUMENT_FIELDS = [
    "client",
    "filename",
    "document_title",
    "document_type",
    "source",
    "source_path",
]
REQUIRED_CHUNK_FIELDS = REQUIRED_DOCUMENT_FIELDS + ["chunk_index", "chunk_id"]
TEST_QUESTIONS = [
    "Hva er Fjordmats tone of voice?",
    "Hva var ROAS for Spareklars Google Ads i Q4 2024?",
    "Hvilke influencere samarbeider Nordvik med?",
    "Hva sier kundecaset om Skytjenester sine resultater hos LogistikkPartner?",
]
CLIENT_ALIASES = {
    "fjordmat": "fjordmat",
    "fjordmats": "fjordmat",
    "spareklar": "spareklar",
    "spareklars": "spareklar",
    "nordvik": "nordvik",
    "nordviks": "nordvik",
    "skytjenester": "skytjenester",
    "skytjenesters": "skytjenester",
}
DOCUMENT_TYPE_HINTS = {
    "brand_guidelines": [
        "tone of voice",
        "merkevare",
        "merkevareretningslinjer",
        "visuell identitet",
    ],
    "campaign_report": [
        "roas",
        "rapport",
        "kampanjerapport",
        "google ads",
        "resultater",
        "resultat",
    ],
    "campaign_brief": ["brief", "kampanjebrief", "lansering"],
    "meeting_notes": ["møtereferat", "meeting notes", "referat"],
    "social_media_strategy": ["sosiale medier", "social media strategy"],
    "influencer_strategy": ["influencer", "influencere"],
    "customer_case": ["kundecase", "kundecaset", "case study"],
    "seo_report": ["seo", "organisk trafikk"],
}


@dataclass(frozen=True)
class QueryContext:
    """Structured hints derived from the user's query."""

    client: str | None = None
    document_type: str | None = None


@dataclass(frozen=True)
class RetrievalExpectation:
    """Expected retrieval target for a sample evaluation question."""

    question: str
    expected_client: str
    expected_document_type: str
    expected_source: str


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    """Compact retrieval evaluation outcome for one question."""

    expectation: RetrievalExpectation
    top_result_source: str | None
    top_result_client: str | None
    top_result_document_type: str | None
    client_match: bool
    document_type_match: bool
    source_match: bool


EVALUATION_CASES = [
    RetrievalExpectation(
        question="Hva er Fjordmats tone of voice?",
        expected_client="fjordmat",
        expected_document_type="brand_guidelines",
        expected_source="fjordmat/merkevareretningslinjer.md",
    ),
    RetrievalExpectation(
        question="Hva var ROAS for Spareklars Google Ads i Q4 2024?",
        expected_client="spareklar",
        expected_document_type="campaign_report",
        expected_source="spareklar/kampanjerapport-google-ads-q4-2024.md",
    ),
    RetrievalExpectation(
        question="Hvilke influencere samarbeider Nordvik med?",
        expected_client="nordvik",
        expected_document_type="influencer_strategy",
        expected_source="nordvik/influencer-strategi.md",
    ),
    RetrievalExpectation(
        question="Hva sier kundecaset om Skytjenester sine resultater hos LogistikkPartner?",
        expected_client="skytjenester",
        expected_document_type="customer_case",
        expected_source="skytjenester/kundecase-logistikkpartner.md",
    ),
]


def build_search_kwargs(
    question: str,
    k: int,
    client: str | None = None,
) -> tuple[QueryContext, dict[str, object]]:
    """Build retrieval search kwargs from the user question and optional filter."""
    context = infer_query_context(question)
    resolved_client = client or context.client
    search_kwargs: dict[str, object] = {"k": max(k * 3, 6)}
    if resolved_client:
        search_kwargs["filter"] = {"client": resolved_client}
    return context, search_kwargs


def extract_title(content: str, fallback: str) -> str:
    """Extract the first markdown H1 title or fall back to the filename."""
    for line in content.splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return fallback


def infer_document_type(filename: str) -> str:
    """Infer a stable document type from the filename."""
    document_type_map = {
        "merkevareretningslinjer": "brand_guidelines",
        "kampanjebrief": "campaign_brief",
        "kampanjerapport": "campaign_report",
        "motereferat": "meeting_notes",
        "sosiale-medier-strategi": "social_media_strategy",
        "influencer-strategi": "influencer_strategy",
        "seo-rapport": "seo_report",
        "kundecase": "customer_case",
    }

    for prefix, document_type in document_type_map.items():
        if filename.startswith(prefix):
            return document_type

    return "other"


def infer_query_context(question: str) -> QueryContext:
    """Infer client and document-type hints from a Norwegian question."""
    normalized_question = question.casefold()

    client = next(
        (
            canonical_client
            for alias, canonical_client in CLIENT_ALIASES.items()
            if alias in normalized_question
        ),
        None,
    )
    document_type_scores: list[tuple[int, int, str]] = []
    for candidate_type, hints in DOCUMENT_TYPE_HINTS.items():
        matched_hints = [hint for hint in hints if hint in normalized_question]
        if matched_hints:
            document_type_scores.append(
                (
                    len(matched_hints),
                    max(len(hint) for hint in matched_hints),
                    candidate_type,
                )
            )

    document_type = (
        max(document_type_scores)[2] if document_type_scores else None
    )

    return QueryContext(client=client, document_type=document_type)


def rerank_results(
    results: list[Document],
    context: QueryContext,
) -> list[Document]:
    """Promote chunks whose metadata matches explicit query hints."""
    def score(document: Document) -> tuple[int, int]:
        client_match = int(
            context.client is not None and document.metadata["client"] == context.client
        )
        document_type_match = int(
            context.document_type is not None
            and document.metadata["document_type"] == context.document_type
        )
        return (client_match + document_type_match, document_type_match)

    return sorted(results, key=score, reverse=True)


def load_documents() -> list[Document]:
    """Load all markdown documents and attach normalized source metadata."""
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = sorted(
        loader.load(),
        key=lambda document: str(Path(document.metadata["source"]).resolve()),
    )

    for doc in documents:
        source_path = Path(doc.metadata["source"]).resolve()
        relative_path = source_path.relative_to(DATA_DIR.resolve())
        filename = relative_path.name

        doc.metadata["client"] = relative_path.parts[0]
        doc.metadata["filename"] = filename
        doc.metadata["document_title"] = extract_title(doc.page_content, filename)
        doc.metadata["document_type"] = infer_document_type(filename.removesuffix(".md"))
        doc.metadata["source_path"] = str(source_path)
        doc.metadata["source"] = str(relative_path)

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(documents)

    chunk_counters: Counter[str] = Counter()
    for chunk in chunks:
        source = chunk.metadata["source"]
        chunk_index = chunk_counters[source]
        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["chunk_id"] = f"{source}#chunk-{chunk_index:03d}"
        chunk_counters[source] += 1

    return sorted(chunks, key=lambda chunk: chunk.metadata["chunk_id"])


def print_corpus_summary(documents: list[Document], chunks: list[Document]) -> None:
    """Print a compact baseline summary of the indexed corpus."""
    documents_by_client = Counter(doc.metadata["client"] for doc in documents)
    chunks_by_client = Counter(chunk.metadata["client"] for chunk in chunks)
    chunks_by_source = Counter(chunk.metadata["source"] for chunk in chunks)

    print("\n--- Corpus Summary ---")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Average chunks per document: {len(chunks) / len(documents):.2f}")
    print("\nDocuments by client:")
    for client, count in sorted(documents_by_client.items()):
        print(f"  - {client}: {count}")

    print("\nChunks by client:")
    for client, count in sorted(chunks_by_client.items()):
        print(f"  - {client}: {count}")

    print("\nTop chunk-heavy documents:")
    for source, count in chunks_by_source.most_common(5):
        print(f"  - {source}: {count}")


def print_metadata_audit(documents: list[Document], chunks: list[Document]) -> None:
    """Print a compact audit of required source metadata fields."""
    print("\n--- Metadata Audit ---")

    missing_document_fields = {
        field: sum(1 for doc in documents if field not in doc.metadata)
        for field in REQUIRED_DOCUMENT_FIELDS
    }
    missing_chunk_fields = {
        field: sum(1 for chunk in chunks if field not in chunk.metadata)
        for field in REQUIRED_CHUNK_FIELDS
    }

    print("Missing document fields:")
    for field, count in missing_document_fields.items():
        print(f"  - {field}: {count}")

    print("\nMissing chunk fields:")
    for field, count in missing_chunk_fields.items():
        print(f"  - {field}: {count}")

    print("\nDocument types:")
    for document_type, count in sorted(
        Counter(doc.metadata["document_type"] for doc in documents).items()
    ):
        print(f"  - {document_type}: {count}")

    print("\nSample chunk metadata:")
    sample_chunk = chunks[0]
    for field in REQUIRED_CHUNK_FIELDS:
        print(f"  - {field}: {sample_chunk.metadata[field]}")


def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize the multilingual local embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def reset_vectorstore() -> None:
    """Remove any persisted Chroma data so indexing stays deterministic."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)


def create_vectorstore(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> Chroma:
    """Create and persist a Chroma vector store from chunked documents."""
    return Chroma.from_documents(
        documents=chunks,
        ids=[chunk.metadata["chunk_id"] for chunk in chunks],
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )


def load_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Load a persisted Chroma vector store from disk."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"No vector store found at {CHROMA_DIR}. Run build_vectorstore() first."
        )

    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def build_vectorstore(reset: bool = True) -> Chroma:
    """Build the full pipeline from raw documents to a persisted vector store."""
    if reset:
        reset_vectorstore()

    documents, chunks = inspect_corpus()

    print("Initializing embedding model...")
    embeddings = get_embeddings()

    print("Creating vector store...")
    vectorstore = create_vectorstore(chunks, embeddings)
    print(f"  Stored {vectorstore._collection.count()} vectors in {CHROMA_DIR}")

    return vectorstore


def inspect_corpus() -> tuple[list[Document], list[Document]]:
    """Load and chunk documents using the same deterministic path as indexing."""
    documents = load_documents()
    chunks = chunk_documents(documents)

    print("Loading documents...")
    print(f"  Loaded {len(documents)} documents")

    print("Chunking documents...")
    print(f"  Created {len(chunks)} chunks")

    return documents, chunks


def query(question: str, k: int = 4, client: str | None = None) -> list[Document]:
    """
    Query the vector store and return relevant document chunks.

    ``client`` can be used to limit retrieval to a specific client folder.
    """
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings)
    return query_vectorstore(vectorstore, question, k=k, client=client)


def query_vectorstore(
    vectorstore: Chroma,
    question: str,
    k: int = 4,
    client: str | None = None,
) -> list[Document]:
    """Query an already-loaded vector store and rerank the results."""
    context, search_kwargs = build_search_kwargs(question, k=k, client=client)
    results = vectorstore.similarity_search(question, **search_kwargs)
    return rerank_results(results, context)[:k]


def print_query_results(questions: list[str], k: int = 2) -> None:
    """Run a small set of baseline retrieval checks and print the top matches."""
    print("\n--- Test Queries ---")
    for question in questions:
        results = query(question, k=k)
        print(f"\nQ: {question}")
        for index, doc in enumerate(results, start=1):
            print(f"  [{index}] {doc.metadata['source']}")
            print(f"      {doc.page_content[:150].strip()}...")


def evaluate_retrieval(
    evaluation_cases: list[RetrievalExpectation],
    vectorstore: Chroma,
) -> list[RetrievalEvaluationResult]:
    """Evaluate whether top-1 retrieval matches expected source metadata."""
    results: list[RetrievalEvaluationResult] = []

    for expectation in evaluation_cases:
        retrieved_documents = query_vectorstore(vectorstore, expectation.question, k=1)
        top_result = retrieved_documents[0] if retrieved_documents else None

        result = RetrievalEvaluationResult(
            expectation=expectation,
            top_result_source=top_result.metadata["source"] if top_result else None,
            top_result_client=top_result.metadata["client"] if top_result else None,
            top_result_document_type=(
                top_result.metadata["document_type"] if top_result else None
            ),
            client_match=(
                bool(top_result)
                and top_result.metadata["client"] == expectation.expected_client
            ),
            document_type_match=(
                bool(top_result)
                and top_result.metadata["document_type"]
                == expectation.expected_document_type
            ),
            source_match=(
                bool(top_result)
                and top_result.metadata["source"] == expectation.expected_source
            ),
        )
        results.append(result)

    return results


def print_evaluation_report(results: list[RetrievalEvaluationResult]) -> None:
    """Print a compact pass/fail retrieval evaluation report."""
    passed = sum(1 for result in results if result.source_match)

    print("\n--- Retrieval Evaluation ---")
    print(f"Top-1 source matches: {passed}/{len(results)}")

    for result in results:
        status = "PASS" if result.source_match else "FAIL"
        print(f"\n[{status}] {result.expectation.question}")
        print(f"  Expected source: {result.expectation.expected_source}")
        print(f"  Actual source:   {result.top_result_source}")
        print(
            "  Match summary: "
            f"client={result.client_match}, "
            f"document_type={result.document_type_match}, "
            f"source={result.source_match}"
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for baseline inspection commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a corpus summary after loading and chunking documents.",
    )
    parser.add_argument(
        "--skip-queries",
        action="store_true",
        help="Build the vector store without running the baseline test queries.",
    )
    parser.add_argument(
        "--metadata-audit",
        action="store_true",
        help="Print a compact audit of the normalized metadata fields.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run a compact retrieval evaluation against fixed sample questions.",
    )
    return parser.parse_args()


def run_cli(args: argparse.Namespace) -> None:
    """Run the CLI entrypoint for indexing, inspection, and evaluation."""
    documents, chunks = inspect_corpus()

    if args.summary:
        print_corpus_summary(documents, chunks)

    if args.metadata_audit:
        print_metadata_audit(documents, chunks)

    reset_vectorstore()

    print("Initializing embedding model...")
    embeddings = get_embeddings()

    print("Creating vector store...")
    vectorstore = create_vectorstore(chunks, embeddings)
    print(f"  Stored {vectorstore._collection.count()} vectors in {CHROMA_DIR}")

    if args.evaluate:
        print_evaluation_report(evaluate_retrieval(EVALUATION_CASES, vectorstore))

    if not args.skip_queries:
        print_query_results(TEST_QUESTIONS)


if __name__ == "__main__":
    run_cli(parse_args())
