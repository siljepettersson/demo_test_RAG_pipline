"""
RAG pipeline for the agency client knowledge base demo.

This module loads Norwegian client documents from ``data/``, chunks them,
embeds them locally, and stores them in Chroma for retrieval.
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
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
TEST_QUESTIONS = [
    "Hva er Fjordmats tone of voice?",
    "Hva var ROAS for Spareklars Google Ads i Q4 2024?",
    "Hvilke influencere samarbeider Nordvik med?",
    "Hva sier kundecaset om Skytjenester sine resultater hos LogistikkPartner?",
]


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
    required_document_fields = [
        "client",
        "filename",
        "document_title",
        "document_type",
        "source",
        "source_path",
    ]
    required_chunk_fields = required_document_fields + ["chunk_index", "chunk_id"]

    print("\n--- Metadata Audit ---")

    missing_document_fields = {
        field: sum(1 for doc in documents if field not in doc.metadata)
        for field in required_document_fields
    }
    missing_chunk_fields = {
        field: sum(1 for chunk in chunks if field not in chunk.metadata)
        for field in required_chunk_fields
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
    for field in required_chunk_fields:
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

    search_kwargs: dict[str, object] = {"k": k}
    if client:
        search_kwargs["filter"] = {"client": client}

    return vectorstore.similarity_search(question, **search_kwargs)


def print_query_results(questions: list[str], k: int = 2) -> None:
    """Run a small set of baseline retrieval checks and print the top matches."""
    print("\n--- Test Queries ---")
    for question in questions:
        results = query(question, k=k)
        print(f"\nQ: {question}")
        for index, doc in enumerate(results, start=1):
            print(f"  [{index}] {doc.metadata['source']}")
            print(f"      {doc.page_content[:150].strip()}...")


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

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

    if not args.skip_queries:
        print_query_results(TEST_QUESTIONS)
