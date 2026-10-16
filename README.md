# DocMind

A small PDF question-answering app (RAG): upload PDFs, ask questions, get an answer with the source file and page number. Built to learn how retrieval quality is actually measured.

**Stack:** Streamlit, PyMuPDF, `BAAI/bge-small-en-v1.5` (sentence-transformers), ChromaDB (in-memory, cosine), Groq API for answer generation.

## How it works
1. PDF is parsed page by page; lines repeated on most pages (headers/footers) are removed.
2. Text is split into ~800-character chunks (150 overlap), each tagged with file and page.
3. Chunks are embedded and stored in ChromaDB.
4. A question retrieves the top 5 chunks, which are sent to the LLM with the instruction to answer only from the context and to say "not found" otherwise.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Copy `.env.example` to `.env` and fill in `GROQ_API_KEY` and `LLM_MODEL`. PDFs are not included in this repo; use your own.

```bash
streamlit run app.py
```

## Evaluation
`eval/questions.json` has 32 questions (23 answerable, 9 not answerable) written against one 25-page course PDF. `eval/run_eval.py` runs retrieval only (no LLM):

```bash
python eval/run_eval.py path/to/your.pdf
```

| Config | hit@5 | hit@1 |
|---|---|---|
| 1500/200 chunks | 22/23 | 17/23 |
| 800/150 chunks | 22/23 | 19/23 |
| 800/150 + header removal | 23/23 | 19/23 |

Caveat: the questions were written by looking at the PDF, so expect lower numbers on unseen questions. 23 questions is too few to call small differences real.

Distances (cosine): answerable questions' best match max 0.42; clearly unrelated questions min 0.53; "topic-adjacent but not in the document" questions sit at 0.30-0.35, overlapping answerable ones, so a distance threshold cannot separate them. That case is left to the LLM's refusal.

Manual end-to-end checks (Groq, temperature 0): 4/4 topic-adjacent questions were refused correctly; table-based and definition questions returned correct answers with the right page. This is a small sample, not a benchmark.

## Known limitations
- Lists that continue across a page break can be answered incompletely (a dataset lifecycle of 11 stages was answered with 7, because chunking is per page).
- The index is in memory and is lost when the app restarts.
- Multi-PDF retrieval has only been spot-checked on two PDFs.
- No OCR: scanned PDFs return no text (a warning is shown).
- English embedding model; Hinglish/Hindi questions have not been tested.
- Cited pages are not automatically verified against the answer.
- The retrieval eval counts a hit if any expected page is in the top 5, so it cannot detect incomplete answers.
