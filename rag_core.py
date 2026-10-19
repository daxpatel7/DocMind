import os
import pymupdf
import chromadb
import re 
from collections import Counter

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_embedding_model():
    # Lazy import: torch tabhi load hoga jab model sach me chahiye
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def parse_pdf(file_bytes):
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    return [
        (i, page.get_text().strip())
        for i, page in enumerate(doc, start=1)
    ]


def chunk_text(text, size=800, overlap=150):
    chunks, start = [], 0

    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap

    return chunks

def remove_repeated_lines(pages, min_pages=4, ratio=0.5):
    """
    pages: list of (page_number, text).
    Removes header/footer-like lines that repeat (digits ignored)
    on at least `ratio` of the pages.
    """
    if len(pages) < min_pages:
        return pages

    def norm(line):
        return re.sub(r"\d+", "#", line.strip())

    counts = Counter()
    for _, text in pages:
        counts.update({norm(l) for l in text.splitlines() if l.strip()})

    limit = ratio * len(pages)
    repeated = {line for line, c in counts.items() if c >= limit}

    cleaned = []
    for page_number, text in pages:
        kept = [l for l in text.splitlines() if norm(l) not in repeated]
        cleaned.append((page_number, "\n".join(kept).strip()))

    return cleaned

def prepare_chunks(file_name, pages, carry=300):
    """
    pages: list of (page_number, text).
    Returns ids, documents, metadatas ready for Chroma.
    The first chunk of each page also gets the last `carry` characters
    of the previous page, so lists that cross a page break stay together.
    """
    pages = remove_repeated_lines(pages)

    ids, documents, metadatas = [], [], []
    prev_page, prev_tail = None, ""

    for page_number, text in pages:
        if len(text) < 30:
            prev_page, prev_tail = None, ""
            continue

        for chunk_number, chunk in enumerate(chunk_text(text)):
            if chunk_number == 0 and prev_tail:
                chunk = f"(continued from page {prev_page}) {prev_tail}\n{chunk}"

            ids.append(f"{file_name}-{page_number}-{chunk_number}")
            documents.append(chunk)
            metadatas.append({
                "file": file_name,
                "page": page_number
            })

        prev_page, prev_tail = page_number, text[-carry:]

    return ids, documents, metadatas


def build_collection(pdf_path, model=None):
    """
    Parse a PDF, chunk it, embed it, and return a fresh
    in-memory ChromaDB collection (cosine distance).
    """
    if model is None:
        model = load_embedding_model()

    client = chromadb.Client()
    collection = client.create_collection(
        name="docmind_eval",
        metadata={"hnsw:space": "cosine"}
    )

    with open(pdf_path, "rb") as pdf_file:
        file_bytes = pdf_file.read()

    pages = parse_pdf(file_bytes)
    file_name = os.path.basename(pdf_path)

    ids, documents, metadatas = prepare_chunks(file_name, pages)

    if documents:
        embeddings = model.encode(
            documents,
            normalize_embeddings=True
        ).tolist()

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    return collection


def retrieve(collection, model, question, top_k=5):
    """
    Return top-k chunks as dicts: text, file, page, distance.
    """
    query_embedding = model.encode(
        QUERY_PREFIX + question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )

    retrieved_chunks = []

    if not results["documents"] or not results["documents"][0]:
        return retrieved_chunks

    for doc, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved_chunks.append({
            "text": doc,
            "file": metadata["file"],
            "page": metadata["page"],
            "distance": distance
        })

    return retrieved_chunks