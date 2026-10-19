import os
import sys

import chromadb

from rag_core import load_embedding_model, parse_pdf, prepare_chunks, retrieve

question = sys.argv[1]
pdf_paths = sys.argv[2:]

model = load_embedding_model()
client = chromadb.Client()
col = client.create_collection("debug", metadata={"hnsw:space": "cosine"})

for path in pdf_paths:
    with open(path, "rb") as f:
        pages = parse_pdf(f.read())

    name = os.path.basename(path)
    ids, docs, metas = prepare_chunks(name, pages)
    embs = model.encode(docs, normalize_embeddings=True).tolist()
    col.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

print("Total chunks:", col.count())

for rank, c in enumerate(retrieve(col, model, question, top_k=30), start=1):
    preview = c["text"][:150].replace("\n", " ")
    print(f"{rank}. {c['file']} | page {c['page']} | {c['distance']:.3f} | {preview}")
