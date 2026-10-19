# DocMind

A small PDF question-answering app (RAG): upload PDFs, ask questions, get an answer with the source file and page number. Built to learn how retrieval quality is actually measured, and where it breaks.

**Stack:** Streamlit, PyMuPDF, `BAAI/bge-small-en-v1.5` (sentence-transformers), ChromaDB (persistent, cosine), Groq API for answer generation.

## How it works
1. PDF is parsed page by page; lines repeated on most pages (headers/footers) are removed.
2. Text is split into ~800-character chunks (150 overlap), each tagged with file and page. The first chunk of every page also carries the last 300 characters of the previous page, so lists that cross a page break are not cut in half.
3. Chunks are embedded and stored in a persistent ChromaDB folder (`data/chroma`), so each PDF is indexed once. The sidebar lists indexed files and has a "Clear index" button (needed after changing chunking or the embedding model).
4. A question retrieves the top 8 chunks, which are sent to the LLM with the instruction to answer only from the context and to say "not found" otherwise.

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
`eval/questions.json` has 32 questions (23 answerable, 9 not answerable) written against one 25-page course PDF. `eval/run_eval.py` runs retrieval only (no LLM, top 5, one PDF):

```bash
python eval/run_eval.py path/to/your.pdf
```

| Config | hit@5 | hit@1 |
|---|---|---|
| 1500/200 chunks | 22/23 | 17/23 |
| 800/150 chunks | 22/23 | 19/23 |
| 800/150 + header removal | 23/23 | 19/23 |
| + previous-page carry | 23/23 | 18/23 |

Caveat: the questions were written by looking at the PDF, so expect lower numbers on unseen questions. 23 questions is too few to call small differences real.

Cosine distances: the best match of answerable questions is at most 0.46; clearly unrelated questions start at 0.53; "topic-adjacent but not in the document" questions sit at 0.30-0.35, overlapping answerable ones, so a distance threshold cannot separate them. That case is left to the LLM's refusal.

Manual end-to-end checks (Groq, temperature 0) on three course PDFs (239 chunks), about a dozen questions: definitions, lists and table lookups returned correct answers with the right file and page; topic-adjacent questions (e.g. retrieval-augmented generation, LSTM) were refused or answered only with what the document actually says. Problems found and what happened:
- A list crossing a page break was answered with 7 of 11 items. Fixed by carrying the previous page's tail into the next chunk.
- "What is LoRA?" was answered from a single True/False line with top 5. Fixed by raising the app to top 8 (about +47% tokens per query).
- One cited page was wrong (a diagram at the end of page 3 was cited as page 4, caused by the carried text).
- A pretraining vs fine-tuning answer missed a comparison table on another page.

These are hand checks, not a benchmark.

## Known limitations
- Tables and rare-acronym queries are the weakest spot; there is no keyword/hybrid search.
- A cited page can be wrong for text carried over from the previous page.
- Multi-PDF behaviour is only hand-checked on three PDFs; `run_eval.py` handles one PDF.
- No OCR: scanned PDFs return no text (a warning is shown).
- English embedding model; Hinglish/Hindi questions have not been tested.
- Cited pages are not automatically verified against the answer.
- The retrieval eval counts a hit if any expected page is in the top 5, so it cannot detect incomplete answers.
