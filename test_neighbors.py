import sys

from rag_core import load_embedding_model, build_collection_multi, retrieve_with_neighbors

question = sys.argv[1]
expand_top = int(sys.argv[2])
paths = sys.argv[3:]

model = load_embedding_model()
col = build_collection_multi(paths, model)

chunks = retrieve_with_neighbors(col, model, question, top_k=8, expand_top=expand_top)

for rank, c in enumerate(chunks, start=1):
    short = c["file"].split("_")[1]
    tag = "NEIGHBOUR" if c["neighbor"] else f"{c['distance']:.3f}"
    print(f"{rank}. {short}:{c['page']} | {tag}")
