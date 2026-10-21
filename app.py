
import os
import time
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from rag_core import (
    MODEL_NAME,
    load_embedding_model,
    parse_pdf,
    prepare_chunks,
    retrieve_with_neighbors,
)

load_dotenv()

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "docmind"

# Examples are deliberately short and relevant to course PDFs.
EXAMPLES = [
    "What are the advantages and limitations of open-source AI?",
    "Explain the main concepts covered in this unit.",
    "Summarize the important points from the uploaded PDF.",
]


st.set_page_config(
    page_title="DocMind",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    """Read the small custom stylesheet on every app run."""
    css_path = Path(".streamlit/style.css")
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


load_css()


@st.cache_resource(show_spinner="Loading the embedding model...")
def load_model():
    return load_embedding_model()


@st.cache_resource
def get_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection():
    return get_client().get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def read_index_metadata(collection):
    """Build document counts from metadata actually stored in ChromaDB."""
    stored = collection.get(include=["metadatas"])
    metadatas = stored.get("metadatas") or []

    documents = {}
    for metadata in metadatas:
        if not metadata:
            continue

        filename = metadata.get("file")
        if not filename:
            continue

        document = documents.setdefault(
            filename,
            {"chunks": 0, "pages": set()},
        )
        document["chunks"] += 1

        page = metadata.get("page")
        if page is not None:
            document["pages"].add(page)

    return documents


def generate_answer(question, retrieved_chunks):
    """Keep the existing Groq answer-generation behavior and system prompt."""
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("LLM_MODEL")

    if not api_key or not model_name:
        raise ValueError("GROQ_API_KEY or LLM_MODEL is missing from .env")

    client = Groq(api_key=api_key)

    context_parts = []
    for item in retrieved_chunks:
        context_parts.append(
            f"Source: {item['file']}, Page: {item['page']}\n"
            f"Content:\n{item['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are DocMind, a document question-answering assistant. "
                    "Answer using only the provided document context. "
                    "If the context does not contain the answer, say: "
                    "'This information was not found in the uploaded document.' "
                    "Do not invent facts. Cite the source filename and page number."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Document context:\n{context}"
                ),
            },
        ],
        temperature=0,
        max_tokens=800,
    )

    return response.choices[0].message.content, response.usage


def index_uploaded_files(files, collection):
    """Index uploaded PDFs while showing real processing stages."""
    if not files:
        st.info("Choose one or more PDF files before indexing.")
        return

    try:
        with st.status("Indexing your PDFs...", expanded=True) as status:
            st.write("Step 1 of 3 · Reading PDF pages")

            prepared_files = []
            total_pages = 0
            total_chunks = 0

            for uploaded_file in files:
                pages = parse_pdf(uploaded_file.getvalue())
                empty_pages = [
                    page_number
                    for page_number, text in pages
                    if len(text.strip()) < 30
                ]

                if empty_pages:
                    st.warning(
                        f"{uploaded_file.name}: little or no extractable text "
                        f"on page(s) {empty_pages}. They may be scanned."
                    )

                ids, docs, metadata = prepare_chunks(
                    uploaded_file.name,
                    pages,
                )
                prepared_files.append(
                    {
                        "name": uploaded_file.name,
                        "pages": pages,
                        "ids": ids,
                        "docs": docs,
                        "metadata": metadata,
                    }
                )
                total_pages += len(pages)
                total_chunks += len(docs)

            st.write(
                f"Read {total_pages} pages and prepared "
                f"{total_chunks} chunks."
            )
            st.write("Step 2 of 3 · Creating embeddings")

            model = load_model()
            embedded_files = []

            for item in prepared_files:
                if item["docs"]:
                    embeddings = model.encode(
                        item["docs"],
                        normalize_embeddings=True,
                    ).tolist()
                else:
                    embeddings = []

                embedded_files.append((item, embeddings))

            st.write("Step 3 of 3 · Saving to persistent ChromaDB")

            for item, embeddings in embedded_files:
                if item["docs"]:
                    collection.upsert(
                        ids=item["ids"],
                        documents=item["docs"],
                        embeddings=embeddings,
                        metadatas=item["metadata"],
                    )

            status.update(
                label="Indexing complete",
                state="complete",
                expanded=False,
            )

        st.success(
            f"Finished indexing {len(files)} PDF(s). "
            f"Prepared {total_chunks} chunks from {total_pages} pages."
        )
        st.rerun()

    except Exception as error:
        st.error("DocMind couldn't finish indexing. Please check the details.")
        with st.expander("Technical error details"):
            st.code(f"{type(error).__name__}: {error}")


def render_document_cards(documents):
    if not documents:
        st.caption("No indexed documents yet.")
        return

    for filename, info in sorted(documents.items()):
        with st.container(border=True):
            st.markdown(f"**{filename}**")
            st.caption(
                f"{len(info['pages'])} pages · {info['chunks']} chunks"
            )


def render_chunk_panel(chunks):
    st.subheader("Chunks sent to the model")
    st.caption(
        "Retrieved context supplied to the answer generator. "
        "These are not necessarily all the pages cited in the answer."
    )

    if not chunks:
        st.info("Ask a question to see the retrieved context here.")
        return

    for rank, chunk in enumerate(chunks, start=1):
        is_neighbor = bool(chunk.get("neighbor", False))
        distance = chunk.get("distance")

        if isinstance(distance, (int, float)):
            distance_label = f"{distance:.3f}"
        else:
            distance_label = "N/A"

        badge = " · Neighbor" if is_neighbor else ""
        title = (
            f"#{rank} · {chunk.get('file', 'Unknown file')} · "
            f"Page {chunk.get('page', '?')}{badge}"
        )

        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(f"Distance: {distance_label}")
            with st.expander("View chunk text"):
                st.text(chunk.get("text", ""))


# Session state keeps the displayed conversation and latest retrieval
# context. The LLM is still called with only the current question.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_chunks" not in st.session_state:
    st.session_state.latest_chunks = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "confirm_clear" not in st.session_state:
    st.session_state.confirm_clear = False


# Sidebar
with st.sidebar:
    st.markdown("# 📚 DocMind")
    st.caption("Your course PDFs, ready to explore.")
    st.divider()

    st.subheader("Add documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can select multiple course PDFs.",
        key="pdf_uploader",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} PDF(s) selected.")
    else:
        st.caption("No PDFs selected.")

    if st.button(
        "Index PDFs",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        index_uploaded_files(uploaded_files, get_collection())

    st.divider()
    st.subheader("Indexed documents")

    try:
        collection = get_collection()
        document_metadata = read_index_metadata(collection)
        render_document_cards(document_metadata)
        total_chunks = collection.count()
        st.caption(f"{total_chunks} total indexed chunks")
    except Exception as error:
        st.error("Couldn't read the current index.")
        with st.expander("Technical error details"):
            st.code(f"{type(error).__name__}: {error}")

    st.divider()
    st.subheader("Clear index")

    if not st.session_state.confirm_clear:
        if st.button("Clear index", use_container_width=True):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("This removes all indexed chunks from DocMind.")
        confirm_col, cancel_col = st.columns(2)

        with confirm_col:
            if st.button("Confirm clear", type="primary", use_container_width=True):
                try:
                    client = get_client()
                    try:
                        client.delete_collection(COLLECTION_NAME)
                    except Exception:
                        pass

                    # Recreate the empty collection with the same settings.
                    client.get_or_create_collection(
                        COLLECTION_NAME,
                        metadata={"hnsw:space": "cosine"},
                    )
                    st.session_state.messages = []
                    st.session_state.latest_chunks = []
                    st.session_state.pending_question = None
                    st.session_state.confirm_clear = False
                    st.success("Index cleared.")
                    st.rerun()
                except Exception as error:
                    st.error("Couldn't clear the index.")
                    with st.expander("Technical error details"):
                        st.code(f"{type(error).__name__}: {error}")

        with cancel_col:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

    with st.expander("Settings", expanded=False):
        st.markdown("**Chunking**")
        st.caption("Chunk size: 800 · Overlap: 150")
        st.markdown("**Retrieval**")
        st.caption("Top-k: 8 · Neighbor expansion: 3")
        st.markdown("**Embedding model**")
        st.code(MODEL_NAME)
        st.markdown("**LLM model**")
        st.code(os.getenv("LLM_MODEL") or "Not configured")


# Main layout
chat_column, context_column = st.columns([1.8, 1], gap="large")

with chat_column:
    st.title("Ask your documents")
    st.caption(
        "Each question is answered independently (no chat memory)."
    )

    if not st.session_state.messages:
        with st.container(border=True):
            st.markdown("### Get started")
            st.markdown(
                "1. **Upload** your course PDFs in the sidebar.\n\n"
                "2. **Index** them so DocMind can search their content.\n\n"
                "3. **Ask** a question and review the answer with its citations."
            )

            st.markdown("**Examples for the course PDFs**")
            example_columns = st.columns(len(EXAMPLES))
            for index, example in enumerate(EXAMPLES):
                with example_columns[index]:
                    if st.button(
                        example,
                        key=f"example_{index}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_question = example
                        st.rerun()

    # Display previous turns from session state.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                if message.get("refusal"):
                    st.info(
                        "This information was not found in the uploaded document."
                    )
                else:
                    st.markdown(message["content"])

                if message.get("error"):
                    st.warning("DocMind couldn't generate an answer.")
                    with st.expander("Technical error details"):
                        st.code(message["error"])

                if message.get("metrics"):
                    metrics = message["metrics"]
                    metric_columns = st.columns(4)
                    metric_columns[0].metric(
                        "Input tokens", metrics["input_tokens"]
                    )
                    metric_columns[1].metric(
                        "Output tokens", metrics["output_tokens"]
                    )
                    metric_columns[2].metric(
                        "Total tokens", metrics["total_tokens"]
                    )
                    metric_columns[3].metric(
                        "Response time", f"{metrics['seconds']:.2f}s"
                    )
                    st.caption(f"Model: {metrics['model']}")

    prompt = st.chat_input(
        "Ask a question about your indexed PDFs...",
        disabled=(get_collection().count() == 0),
    )

    question = prompt
    if not question and st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question:
        if get_collection().count() == 0:
            st.info("Upload and index a PDF before asking a question.")
        else:
            st.session_state.messages.append(
                {"role": "user", "content": question}
            )

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    start_time = time.perf_counter()
                    model = load_model()
                    retrieved_chunks = retrieve_with_neighbors(
                        collection=get_collection(),
                        model=model,
                        question=question,
                        top_k=8,
                        expand_top=3,
                    )

                    with st.spinner("Searching documents and generating answer..."):
                        answer, usage = generate_answer(
                            question,
                            retrieved_chunks,
                        )

                    elapsed = time.perf_counter() - start_time
                    refusal = (
                        answer.strip()
                        == "This information was not found in the uploaded document."
                    )

                    if refusal:
                        st.info(
                            "This information was not found in the uploaded document."
                        )
                    else:
                        st.markdown(answer)

                    metrics = {
                        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                        "seconds": elapsed,
                        "model": os.getenv("LLM_MODEL") or "Unknown",
                    }

                    metric_columns = st.columns(4)
                    metric_columns[0].metric("Input tokens", metrics["input_tokens"])
                    metric_columns[1].metric("Output tokens", metrics["output_tokens"])
                    metric_columns[2].metric("Total tokens", metrics["total_tokens"])
                    metric_columns[3].metric("Response time", f"{elapsed:.2f}s")
                    st.caption(f"Model: {metrics['model']}")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "refusal": refusal,
                            "metrics": metrics,
                        }
                    )
                    st.session_state.latest_chunks = retrieved_chunks

                except Exception as error:
                    error_text = f"{type(error).__name__}: {error}"
                    st.warning(
                        "DocMind couldn't generate an answer. "
                        "Please check the technical details."
                    )
                    with st.expander("Technical error details"):
                        st.code(error_text)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": "",
                            "error": error_text,
                        }
                    )
                    st.session_state.latest_chunks = []

            st.rerun()

    # Download the visible chat as Markdown.
    if st.session_state.messages:
        markdown_lines = ["# DocMind Chat", ""]

        for message in st.session_state.messages:
            if message["role"] == "user":
                markdown_lines.extend(
                    ["## You", message["content"], ""]
                )
            else:
                markdown_lines.extend(
                    ["## DocMind", message.get("content", ""), ""]
                )
                if message.get("metrics"):
                    metrics = message["metrics"]
                    markdown_lines.append(
                        f"*Tokens: {metrics['input_tokens']} input, "
                        f"{metrics['output_tokens']} output, "
                        f"{metrics['total_tokens']} total · "
                        f"{metrics['seconds']:.2f}s · "
                        f"Model: {metrics['model']}*"
                    )
                    markdown_lines.append("")

        st.download_button(
            "Download chat as Markdown",
            data="\n".join(markdown_lines),
            file_name="docmind-chat.md",
            mime="text/markdown",
            use_container_width=True,
        )


with context_column:
    st.subheader("Context")
    render_chunk_panel(st.session_state.latest_chunks)
