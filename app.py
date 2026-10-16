import os
import streamlit as st
import chromadb
from groq import Groq
from dotenv import load_dotenv

from rag_core import (
    load_embedding_model,
    parse_pdf,
    prepare_chunks,
    retrieve,
)

load_dotenv()


@st.cache_resource(show_spinner="Model is loading (first time may take time)...")
def load_model():
    return load_embedding_model()


@st.cache_resource
def get_collection():
    client = chromadb.Client()
    return client.get_or_create_collection(
        "docmind",
        metadata={"hnsw:space": "cosine"}
    )


def generate_answer(question, retrieved_chunks):
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("LLM_MODEL")

    if not api_key or not model_name:
        raise ValueError("GROQ_API_KEY ya LLM_MODEL .env me missing hai")

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
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Document context:\n{context}"
                )
            }
        ],
        temperature=0,
        max_tokens=800
    )

    return (
        response.choices[0].message.content,
        response.usage
    )


st.title("DocMind")

col = get_collection()

files = st.file_uploader(
    "PDF upload karo",
    type="pdf",
    accept_multiple_files=True
)

if files and st.button("Index karo"):
    model = load_model()

    for f in files:
        pages = parse_pdf(f.read())

        empty = [p for p, t in pages if len(t) < 30]

        if empty:
            st.warning(
                f"{f.name}: pages {empty} me text nahi mila "
                "(scanned ho sakte hain)"
            )

        ids, docs, metas = prepare_chunks(f.name, pages)

        if docs:
            with st.spinner(f"{f.name} index ho rahi hai..."):
                embs = model.encode(
                    docs,
                    normalize_embeddings=True
                ).tolist()

                col.upsert(
                    ids=ids,
                    documents=docs,
                    embeddings=embs,
                    metadatas=metas
                )

        st.success(f"{f.name}: {len(pages)} pages, {len(docs)} chunks")

st.caption(f"Total chunks indexed: {col.count()}")

q = st.text_input("Ask Question")

if q and col.count() > 0:
    model = load_model()
    retrieved_chunks = retrieve(col, model, q, top_k=5)

    try:
        with st.spinner("DocMind answer generate kar raha hai..."):
            answer, usage = generate_answer(q, retrieved_chunks)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Token Usage")
        c1, c2, c3 = st.columns(3)
        c1.metric("Input Tokens", usage.prompt_tokens)
        c2.metric("Output Tokens", usage.completion_tokens)
        c3.metric("Total Tokens", usage.total_tokens)

    except Exception as e:
        st.error(f"Answer generate nahi hua: {e}")

    st.subheader("Retrieved chunks (jo LLM ko bheje gaye)")
    for chunk in retrieved_chunks:
        with st.expander(
            f"{chunk['file']} | Page {chunk['page']} | distance {chunk['distance']:.3f}"
        ):
            st.write(chunk["text"])