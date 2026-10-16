import sys
import json
from pathlib import Path
from datetime import datetime

# Project root ko Python import path mein add karo
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_core import load_embedding_model, build_collection, retrieve


def main():
    if len(sys.argv) != 2:
        print("Usage: python eval/run_eval.py PDF_PATH")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    questions_path = PROJECT_ROOT / "eval" / "questions.json"
    results_dir = PROJECT_ROOT / "eval"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"results_{timestamp}.txt"

    # Console output aur file output dono save karo
    with open(results_path, "w", encoding="utf-8") as output_file:
        class Tee:
            def write(self, text):
                sys.__stdout__.write(text)
                output_file.write(text)
                output_file.flush()

            def flush(self):
                sys.__stdout__.flush()
                output_file.flush()

        old_stdout = sys.stdout
        sys.stdout = Tee()

        try:
            if not pdf_path.is_file():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            if not questions_path.is_file():
                raise FileNotFoundError(
                    f"Questions file not found: {questions_path}"
                )

            with open(questions_path, "r", encoding="utf-8") as file:
                questions = json.load(file)

            if not isinstance(questions, list):
                raise ValueError("questions.json must contain a JSON list.")

            valid_questions = []
            for item in questions:
                question = str(item.get("question", ""))

                if question.lstrip().upper().startswith("TODO"):
                    print(
                        f"WARNING: Skipping question ID "
                        f"{item.get('id', '?')}: TODO question."
                    )
                    continue

                valid_questions.append(item)

            print("DocMind Retrieval Evaluation")
            print("=" * 100)
            print(f"PDF: {pdf_path.name}")
            print()

            # Model ek hi baar load hoga
            print("Loading embedding model...")
            model = load_embedding_model()

            # Collection ek hi baar banegi
            print("Building collection...")
            collection = build_collection(str(pdf_path), model)
            print()

            headers = [
                "ID", "TYPE", "QUESTION (first 40 chars)",
                "TOP-5 PAGES", "TOP-1 DISTANCE",
                "HIT@5", "FIRST HIT RANK"
            ]

            rows = []
            in_doc_count = 0
            hit_at_5 = 0
            hit_at_1 = 0
            in_doc_distances = []
            out_distances = []

            for item in valid_questions:
                question_id = item.get("id", "")
                question_type = item.get("type", "")
                question = str(item.get("question", ""))
                expected_pages = {
                    str(page) for page in (item.get("expected_pages") or [])
                }

                # Retrieval only: LLM/Groq call nahi hota
                results = retrieve(
                    collection=collection,
                    model=model,
                    question=question,
                    top_k=5
                )

                pages = [str(result["page"]) for result in results]
                pages_text = ", ".join(pages) if pages else "(none)"

                distance = results[0]["distance"] if results else None
                distance_text = (
                    f"{distance:.4f}"
                    if isinstance(distance, (int, float))
                    else "N/A"
                )

                hit_status = "-"
                hit_rank = "-"

                if question_type == "in_doc":
                    in_doc_count += 1
                    matching_ranks = [
                        rank
                        for rank, page in enumerate(pages, start=1)
                        if page in expected_pages
                    ]

                    if matching_ranks:
                        hit_at_5 += 1
                        hit_status = "HIT"
                        hit_rank = matching_ranks[0]

                        if hit_rank == 1:
                            hit_at_1 += 1
                    else:
                        hit_status = "MISS"

                    if isinstance(distance, (int, float)):
                        in_doc_distances.append(distance)

                elif question_type in (
                    "out_of_doc", "out_of_doc_adjacent"
                ):
                    out_distances.append(
                        (question_id, question_type, distance)
                    )

                rows.append([
                    str(question_id),
                    str(question_type),
                    question[:40].replace("\n", " "),
                    pages_text,
                    distance_text,
                    hit_status,
                    str(hit_rank)
                ])

            # Table print karo
            widths = [
                max(
                    len(headers[i]),
                    max((len(row[i]) for row in rows), default=0)
                )
                for i in range(len(headers))
            ]

            def print_row(row):
                print(" | ".join(
                    value.ljust(widths[i])
                    for i, value in enumerate(row)
                ))

            print_row(headers)
            print("-+-".join("-" * width for width in widths))

            for row in rows:
                print_row(row)

            print()
            print("=" * 100)
            print("EVALUATION SUMMARY")
            print(f"In-document count: {in_doc_count}")
            print(f"hit@5: {hit_at_5}/{in_doc_count}")
            print(f"hit@1: {hit_at_1}/{in_doc_count}")

            print()
            print("Out-of-document top-1 distances:")
            if out_distances:
                for qid, qtype, dist in out_distances:
                    value = (
                        f"{dist:.4f}"
                        if isinstance(dist, (int, float))
                        else "N/A"
                    )
                    print(f"ID {qid} ({qtype}): {value}")
            else:
                print("None")

            if in_doc_distances:
                print(
                    "Max in_doc top-1 distance: "
                    f"{max(in_doc_distances):.4f}"
                )
            else:
                print("Max in_doc top-1 distance: N/A")

            print(f"\nResults saved to: {results_path}")

        finally:
            sys.stdout = old_stdout


if __name__ == "__main__":
    main()