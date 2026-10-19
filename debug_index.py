import sys

import chromadb

from rag_core import load_embedding_model, retrieve

question = sys.argv[1]
top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 30

client = chromadb.PersistentClient(path="data/chroma")
col = client.get_collection("docmind")
model = load_embedding_model()

print("Total chunks:", col.count())

for rank, c in enumerate(retrieve(col, model, question, top_k=top_k), start=1):
    preview = c["text"][:150].replace("\n", " ")
    print(f"{rank}. {c['file']} | page {c['page']} | {c['distance']:.3f} | {preview}")
