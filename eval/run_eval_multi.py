
import argparse
import json
import re
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path


# Add project root to Python import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_core import (
    load_embedding_model,
    build_collection_multi,
    retrieve,
    retrieve_with_neighbors,
)


QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions_multi.json"
RESULTS_DIR = PROJECT_ROOT / "eval"

EVALUATED_TYPES = ("in_doc", "in_doc_hinglish")
OUT_TYPES = ("out_of_doc", "out_of_doc_adjacent")


class Tee:
    """Write output to both the terminal and the results file."""

    def __init__(self, console, file_handle):
        self.console = console
        self.file_handle = file_handle

    def write(self, text):
        self.console.write(text)
        self.file_handle.write(text)
        self.file_handle.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.file_handle.flush()


def resolve_pdf_path(path_string):
    """Resolve relative paths from the current directory or project root."""
    path = Path(path_string).expanduser()

    if not path.is_absolute() and not path.is_file():
        path = PROJECT_ROOT / path

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    return path


def unit_label(file_name):
    """
    Convert a PDF filename to a unit label.
    Example: NEW_U2_GenAI_Sem7.pdf -> U2
    """
    match = re.search(r"_U(\d+)_", Path(file_name).name, re.IGNORECASE)

    if match:
        return f"U{match.group(1)}"

    return Path(file_name).stem


def result_label(result):
    """Convert a retrieved chunk into a label such as U2:5."""
    unit = unit_label(result["file"])
    page = int(result["page"])
    return f"{unit}:{page}"


def normalize_expected(item):
    """Validate and normalize expected entries such as U2:5."""
    expected = item.get("expected", [])

    if not isinstance(expected, list):
        raise ValueError(
            f"Question ID {item.get('id', '?')}: "
            "'expected' must be a list, e.g. ['U2:5']."
        )

    normalized = []

    for entry in expected:
        entry = str(entry).strip().upper()

        if not re.fullmatch(r"U\d+:\d+", entry):
            raise ValueError(
                f"Question ID {item.get('id', '?')}: "
                f"invalid expected entry {entry!r}; use U2:5 format."
            )

        normalized.append(entry)

    return list(dict.fromkeys(normalized))


def format_distance(value):
    if isinstance(value, (int, float)):
        return f"{value:.4f}"

    return "N/A"


def percentage(numerator, denominator):
    if denominator == 0:
        return "N/A"

    return f"{numerator}/{denominator} ({numerator / denominator * 100:.1f}%)"


def print_summary(questions, evaluated):
    print()
    print("=" * 110)
    print("EVALUATION SUMMARY")
    print("=" * 110)

    for question_type in EVALUATED_TYPES:
        subset = [
            row for row in evaluated
            if row["type"] == question_type
        ]

        count = len(subset)
        hits_top_k = sum(row["hit"] for row in subset)
        hits_top_1 = sum(row["hit_top_1"] for row in subset)

        need_all_rows = [
            row for row in subset
            if row["need_all"]
        ]

        complete_count = sum(
            row["hit_all"] for row in need_all_rows
        )

        print()
        print(f"Type: {question_type}")
        print(f"Question count: {count}")
        print(
            "hit@top_k: "
            f"{percentage(hits_top_k, count)}"
        )
        print(
            "hit@1: "
            f"{percentage(hits_top_1, count)}"
        )
        print(
            "complete@top_k (need_all only): "
            f"{percentage(complete_count, len(need_all_rows))}"
        )
        print(f"need_all question count: {len(need_all_rows)}")

    print()
    print("Out-of-document top-1 distances:")

    out_rows = [
        row for row in evaluated
        if row["type"] in OUT_TYPES
    ]

    if out_rows:
        for row in out_rows:
            print(
                f"ID {row['id']} ({row['type']}): "
                f"{format_distance(row['top1_distance'])}"
            )
    else:
        print("None")

    in_doc_rows = [
        row for row in evaluated
        if row["type"] in EVALUATED_TYPES
        and isinstance(row["top1_distance"], (int, float))
    ]

    print()
    if in_doc_rows:
        maximum = max(row["top1_distance"] for row in in_doc_rows)
        print(
            "Max top-1 distance among in_doc and "
            f"in_doc_hinglish: {maximum:.4f}"
        )
    else:
        print("Max top-1 distance among in_doc and in_doc_hinglish: N/A")

    for question_type in EVALUATED_TYPES:
        distances = [
            row["top1_distance"]
            for row in in_doc_rows
            if row["type"] == question_type
        ]

        if distances:
            print(
                f"Max {question_type} top-1 distance: "
                f"{max(distances):.4f}"
            )
        else:
            print(f"Max {question_type} top-1 distance: N/A")


def run_evaluation(pdf_paths, top_k, expand_top):
    if not QUESTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Questions file not found: {QUESTIONS_PATH}"
        )

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("questions_multi.json must contain a JSON list.")

    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("Every question entry must be a JSON object.")

        if "question" not in item:
            raise ValueError(
                f"Question entry {item.get('id', '?')} has no 'question'."
            )

        item["_expected_normalized"] = normalize_expected(item)

    print("DocMind Multi-PDF Retrieval Evaluation")
    print("=" * 110)
    print(f"PDFs: {len(pdf_paths)}")
    for path in pdf_paths:
        print(f"  - {path.name}")

    print(f"Questions: {len(questions)}")
    print(f"top_k: {top_k}")
    print(f"expand_top: {expand_top}")
    print("Mode: Retrieval only — no LLM/Groq calls")
    print()

    # Load the embedding model exactly once.
    print("Loading embedding model...")
    model = load_embedding_model()

    # Build one fresh collection from all supplied PDFs.
    print()
    print("Building multi-PDF collection...")
    collection = build_collection_multi(
        [str(path) for path in pdf_paths],
        model=model
    )
    print(f"Collection count: {collection.count()}")
    print()

    headers = [
        "ID",
        "TYPE",
        "QUESTION (first 40 chars)",
        "TOP-K RESULTS",
        "TOP-1 DISTANCE",
        "HIT/MISS",
        "FIRST HIT RANK",
        "NEED_ALL",
    ]

    rows = []
    evaluated = []

    for item in questions:
        question_id = str(item.get("id", "?"))
        question_type = str(item.get("type", ""))
        question = str(item["question"]).replace("\n", " ")
        expected = item["_expected_normalized"]
        need_all = item.get("need_all", False)

        if not isinstance(need_all, bool):
            raise ValueError(
                f"Question ID {question_id}: 'need_all' must be true or false."
            )

        if expand_top > 0:
            results = retrieve_with_neighbors(
                collection=collection,
                model=model,
                question=question,
                top_k=top_k,
                expand_top=expand_top,
            )
        else:
            results = retrieve(
                collection=collection,
                model=model,
                question=question,
                top_k=top_k
            )

        # Keep normal top-k hits separate from appended neighbour chunks.
        normal_results = [
            result for result in results
            if not result.get("neighbor", False)
        ]
        display_labels = [
            result_label(result) + ("*" if result.get("neighbor", False) else "")
            for result in results
        ]
        normal_labels = [result_label(result) for result in normal_results]
        all_labels = [result_label(result) for result in results]
        retrieved_set = set(all_labels)

        top1_distance = (
            normal_results[0].get("distance") if normal_results else None
        )

        # HIT/MISS includes normal hits and neighbour chunks.
        hit = any(label in expected for label in all_labels)

        # First-hit rank uses only the normal top-k hits, never neighbours.
        matching_ranks = [
            rank
            for rank, label in enumerate(normal_labels, start=1)
            if label in expected
        ]
        first_hit_rank = matching_ranks[0] if matching_ranks else None

        # HIT_ALL includes both normal hits and neighbour chunks.
        missing_entries = [
            entry for entry in expected
            if entry not in retrieved_set
        ]

        hit_all = not missing_entries and bool(expected)

        if need_all:
            need_all_status = (
                "HIT_ALL"
                if hit_all
                else "MISSING " + ", ".join(missing_entries)
            )
        else:
            need_all_status = "-"

        if question_type in EVALUATED_TYPES:
            hit_status = "HIT" if hit else "MISS"
        else:
            hit_status = "-"

        rows.append([
            question_id,
            question_type,
            question[:40],
            ", ".join(display_labels) if display_labels else "(none)",
            format_distance(top1_distance),
            hit_status,
            str(first_hit_rank) if first_hit_rank is not None else "-",
            need_all_status,
        ])

        evaluated.append({
            "id": question_id,
            "type": question_type,
            "top1_distance": top1_distance,
            "hit": hit,
            "hit_top_1": bool(normal_labels and normal_labels[0] in expected),
            "need_all": need_all,
            "hit_all": hit_all,
            "missing_entries": missing_entries,
        })

    widths = [
        max(
            len(headers[index]),
            max((len(row[index]) for row in rows), default=0)
        )
        for index in range(len(headers))
    ]

    def print_row(row):
        print(
            " | ".join(
                value.ljust(widths[index])
                for index, value in enumerate(row)
            )
        )

    print_row(headers)
    print("-+-".join("-" * width for width in widths))

    for row in rows:
        print_row(row)

    print_summary(questions, evaluated)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DocMind retrieval across multiple PDFs."
    )

    parser.add_argument(
        "pdfs",
        nargs="+",
        help="PDF paths, for example U1.pdf U2.pdf U3.pdf"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=8,
        help="Number of retrieved chunks to evaluate (default: 8)"
    )

    parser.add_argument(
        "--expand",
        type=int,
        default=0,
        help="Number of neighbour chunks to append (default: 0)"
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top_k must be at least 1.")

    if args.expand < 0:
        parser.error("--expand must be 0 or greater.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"results_multi_{timestamp}.txt"

    # Resolve paths before starting evaluation.
    try:
        pdf_paths = [resolve_pdf_path(path) for path in args.pdfs]

        if not QUESTIONS_PATH.is_file():
            raise FileNotFoundError(
                f"Questions file not found: {QUESTIONS_PATH}"
            )

        with open(results_path, "w", encoding="utf-8") as output_file:
            tee = Tee(sys.__stdout__, output_file)

            with redirect_stdout(tee):
                try:
                    run_evaluation(pdf_paths, args.top_k, args.expand)
                except Exception as exc:
                    print()
                    print(f"ERROR: {type(exc).__name__}: {exc}")
                    raise

                print()
                print(f"Full results saved to: {results_path}")

    except Exception as exc:
        print(f"Evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"Check partial output, if any: {results_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
