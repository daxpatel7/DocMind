"""
PDF validation helpers for DocMind (no Streamlit imports).

validate_pdf() lowers the risk of a bad upload before a file is parsed and
indexed. It cannot prove that a file is free of malware: no check can.

Layers, in order:
1. Cheap checks in the main process: size, "%PDF-" header, file name.
2. Optional Windows Defender scan (best effort, Windows only).
3. Structural checks in a separate process with a timeout, so a parser crash
   or hang cannot take the app down.

The worker runs as:  python validation.py --worker --max-pages N --max-text N
and reads the file bytes from stdin.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

MAX_BYTES = 25 * 1024 * 1024
MAX_PAGES = 300
MAX_TEXT_CHARS = 5_000_000
MIN_TEXT_CHARS = 30
MAX_NAME_LEN = 100
WORKER_TIMEOUT_S = 20
AV_TIMEOUT_S = 60

RESULT_MARKER = "DOCMIND_RESULT:"

# PDF names that mean active or embedded content.
DANGEROUS_KEYS = (
    "JS",
    "JavaScript",
    "Launch",
    "EmbeddedFile",
    "RichMedia",
    "XFA",
    "SubmitForm",
    "ImportData",
)
WARN_KEYS = ("GoToR", "GoToE")


def _key_regex(keys):
    # A PDF name starts with "/" and ends at a delimiter. The lookahead stops
    # "/JS" from matching names such as "/JSON".
    return re.compile(r"/(" + "|".join(keys) + r")(?![A-Za-z0-9_#.\-])")


DANGEROUS_RE = _key_regex(DANGEROUS_KEYS)
WARN_RE = _key_regex(WARN_KEYS)

INJECTION_PATTERNS = [
    re.compile(
        r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+"
        r"(instructions?|prompts?|rules?)",
        re.I,
    ),
    re.compile(r"disregard\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)", re.I),
    re.compile(r"forget\s+(all\s+|your\s+)?(previous|prior|earlier)\s+(instructions?|prompts?)", re.I),
    re.compile(r"\bsystem\s+prompt\b", re.I),
    re.compile(r"\byou\s+are\s+now\b", re.I),
    re.compile(r"reveal\s+(your|the)\s+(system\s+)?(prompt|instructions?)", re.I),
]


@dataclass
class ValidationResult:
    ok: bool = False
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    sha256: str = ""
    safe_name: str = ""
    page_count: int = 0
    av_status: str = "not run"


# --------------------------------------------------------------------------
# File name
# --------------------------------------------------------------------------
def clean_filename(filename):
    """Return (clean_name, error). Keeps only a plain base name ending in .pdf."""
    base = re.split(r"[\\/]", str(filename or ""))[-1].strip()
    if not base.lower().endswith(".pdf"):
        return "", "The file name must end in .pdf."

    stem = base[:-4]
    stem = re.sub(r"[^A-Za-z0-9 ._-]", "_", stem)
    stem = re.sub(r"\.{2,}", ".", stem)
    stem = stem.strip(" .")
    if not stem:
        stem = "document"
    stem = stem[: MAX_NAME_LEN - 4].strip(" .") or "document"
    return stem + ".pdf", None


# --------------------------------------------------------------------------
# Structural checks (run inside the worker process)
# --------------------------------------------------------------------------
def _inspect(data, max_pages=MAX_PAGES, max_text=MAX_TEXT_CHARS):
    import pymupdf

    try:
        pymupdf.TOOLS.mupdf_display_errors(False)
    except Exception:
        pass

    reasons, warnings = [], []
    doc = pymupdf.open(stream=data, filetype="pdf")
    try:
        if doc.needs_pass or doc.is_encrypted:
            return {
                "reasons": ["Encrypted or password-protected PDFs are not supported."],
                "warnings": [],
                "page_count": 0,
            }

        page_count = doc.page_count
        if page_count == 0:
            reasons.append("The PDF has no pages.")
        elif page_count > max_pages:
            reasons.append(f"The PDF has {page_count} pages; the limit is {max_pages}.")

        # Active or embedded content. xref_object() returns the decoded object,
        # so names written as /J#61vaScript are caught and objects stored inside
        # compressed object streams are covered too.
        dangerous, warned = set(), set()
        for xref in range(1, doc.xref_length()):
            try:
                obj = doc.xref_object(xref, compressed=False)
            except Exception:
                continue
            for match in DANGEROUS_RE.finditer(obj):
                dangerous.add(match.group(1))
            for match in WARN_RE.finditer(obj):
                warned.add(match.group(1))

        try:
            if doc.embfile_count() > 0:
                dangerous.add("EmbeddedFile")
        except Exception:
            pass

        if dangerous:
            reasons.append(
                "The PDF contains active or embedded content ("
                + ", ".join(sorted(dangerous))
                + ") and was rejected."
            )
        if warned:
            warnings.append(
                "The PDF links to other files (" + ", ".join(sorted(warned)) + "); these links are ignored."
            )

        # Text, only when the document passed the checks above.
        if not reasons:
            total_chars = 0
            no_text_pages = []
            injection_pages = {}
            for index in range(page_count):
                text = doc[index].get_text()
                total_chars += len(text.strip())
                if len(text.strip()) < MIN_TEXT_CHARS:
                    no_text_pages.append(index + 1)
                for pattern in INJECTION_PATTERNS:
                    if pattern.search(text):
                        injection_pages.setdefault(index + 1, True)
                if total_chars > max_text:
                    reasons.append("The PDF contains too much text and was rejected.")
                    break

            if not reasons and total_chars < MIN_TEXT_CHARS:
                reasons.append(
                    "No readable text was found. Scanned PDFs are not supported (no OCR)."
                )
            elif not reasons:
                if no_text_pages:
                    shown = ", ".join(str(p) for p in no_text_pages[:10])
                    more = "" if len(no_text_pages) <= 10 else " and more"
                    warnings.append(
                        f"Pages with little or no text: {shown}{more}. They may be scanned images."
                    )
                if injection_pages:
                    pages = ", ".join(str(p) for p in list(injection_pages)[:5])
                    warnings.append(
                        "Some text looks like an instruction aimed at an AI model "
                        f"(page {pages}). It will be treated as document text, not as instructions."
                    )

        return {"reasons": reasons, "warnings": warnings, "page_count": page_count}
    finally:
        try:
            doc.close()
        except Exception:
            pass


def _worker_main(argv):
    max_pages, max_text = MAX_PAGES, MAX_TEXT_CHARS
    if "--max-pages" in argv:
        max_pages = int(argv[argv.index("--max-pages") + 1])
    if "--max-text" in argv:
        max_text = int(argv[argv.index("--max-text") + 1])

    data = sys.stdin.buffer.read()
    try:
        result = _inspect(data, max_pages=max_pages, max_text=max_text)
    except Exception as exc:  # the file could not be read as a PDF
        result = {
            "reasons": [f"The file could not be read as a PDF ({type(exc).__name__})."],
            "warnings": [],
            "page_count": 0,
        }
    sys.stdout.write("\n" + RESULT_MARKER + json.dumps(result) + "\n")
    sys.stdout.flush()


def _run_worker(data, timeout_s):
    """Return (payload, error). Any failure of the worker means rejection."""
    command = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--max-pages",
        str(MAX_PAGES),
        "--max-text",
        str(MAX_TEXT_CHARS),
    ]
    try:
        proc = subprocess.run(command, input=data, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None, "The file took too long to check and was rejected."
    except Exception:
        return None, "The file could not be checked and was rejected."

    output = proc.stdout.decode("utf-8", "replace")
    marker = output.rfind(RESULT_MARKER)
    if proc.returncode != 0 or marker == -1:
        return None, "The check crashed on this file and it was rejected."
    try:
        line = output[marker + len(RESULT_MARKER):].strip().splitlines()[0]
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError
        return payload, None
    except Exception:
        return None, "The check returned an unreadable result and the file was rejected."


# --------------------------------------------------------------------------
# Antivirus (best effort, Windows Defender only)
# --------------------------------------------------------------------------
def _find_defender():
    if os.name != "nt":
        return None
    candidates = []
    program_data = os.environ.get("ProgramData")
    if program_data:
        pattern = os.path.join(
            program_data, "Microsoft", "Windows Defender", "Platform", "*", "MpCmdRun.exe"
        )
        candidates += sorted(glob.glob(pattern), reverse=True)
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidates.append(os.path.join(program_files, "Windows Defender", "MpCmdRun.exe"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def av_scan(data):
    """Return "clean", "threat found" or "not available".

    Exit codes follow Microsoft's MpCmdRun documentation (0: no threat found,
    2: threat found). Check them on your own machine before relying on them.
    "clean" only means this scanner found nothing.
    """
    exe = _find_defender()
    if not exe:
        return "not available"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        proc = subprocess.run(
            [exe, "-Scan", "-ScanType", "3", "-File", tmp_path],
            capture_output=True,
            timeout=AV_TIMEOUT_S,
        )
        if not os.path.exists(tmp_path) or proc.returncode == 2:
            return "threat found"
        if proc.returncode == 0:
            return "clean"
        return "not available"
    except Exception:
        return "not available"
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def validate_pdf(file_bytes, filename, timeout_s=WORKER_TIMEOUT_S):
    """Validate an uploaded PDF. Never raises: any internal error means reject."""
    result = ValidationResult()
    try:
        data = bytes(file_bytes) if file_bytes is not None else b""
        result.sha256 = hashlib.sha256(data).hexdigest()

        clean_name, name_error = clean_filename(filename)
        result.safe_name = clean_name
        if name_error:
            result.reasons.append(name_error)

        if not data:
            result.reasons.append("The file is empty.")
        else:
            if len(data) > MAX_BYTES:
                limit_mb = MAX_BYTES / (1024 * 1024)
                result.reasons.append(f"The file is larger than the {limit_mb:g} MB limit.")
            if b"%PDF-" not in data[:1024]:
                result.reasons.append("This does not look like a PDF file (no %PDF- header).")
            if b"%%EOF" not in data[-2048:]:
                result.warnings.append(
                    "The end-of-file marker is missing; the file may be cut off."
                )

        if result.reasons:
            return result  # do not scan or parse anything that failed the cheap checks

        result.av_status = av_scan(data)
        if result.av_status == "threat found":
            result.reasons.append(
                "The antivirus scan reported a threat (or removed the file), so it was rejected."
            )
            return result

        payload, error = _run_worker(data, timeout_s)
        if error:
            result.reasons.append(error)
            return result

        result.reasons.extend(payload.get("reasons", []))
        result.warnings.extend(payload.get("warnings", []))
        result.page_count = int(payload.get("page_count", 0) or 0)
        result.ok = not result.reasons
        return result
    except Exception:
        result.ok = False
        result.reasons.append("Validation failed unexpectedly, so the file was rejected.")
        return result


# --------------------------------------------------------------------------
# Model answer sanitiser
# --------------------------------------------------------------------------
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)|!\[[^\]]*\]\[[^\]]*\]")
_HTML_TAG_RE = re.compile(
    r"<\s*/?\s*(img|script|iframe|object|embed|style|link|svg|form|meta|base)\b[^>]*>", re.I
)
_LINK_RE = re.compile(r"\[([^\]]+)\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
_AUTOLINK_RE = re.compile(r"<(https?://[^>\s]+)>")
_BARE_URL_RE = re.compile(r"(?<![`(\[])(https?://[^\s)`<>\]]+)")


def _sanitize_plain(segment):
    segment = _IMAGE_RE.sub("", segment)
    segment = _HTML_TAG_RE.sub("", segment)
    segment = _LINK_RE.sub(lambda m: f"{m.group(1)} (`{m.group(2)}`)", segment)
    segment = _AUTOLINK_RE.sub(lambda m: f"`{m.group(1)}`", segment)
    segment = _BARE_URL_RE.sub(lambda m: f"`{m.group(1)}`", segment)
    return segment


def sanitize_answer(text):
    """Remove images and raw HTML tags and make links non-clickable.

    Text inside backticks is left unchanged. Other text is kept as is.
    """
    if not text:
        return ""
    parts = str(text).split("`")
    for index in range(0, len(parts), 2):  # even parts are outside inline code
        parts[index] = _sanitize_plain(parts[index])
    return "`".join(parts)


# --------------------------------------------------------------------------
# Duplicate handling (registry kept in data/registry.json by the app)
# --------------------------------------------------------------------------
def load_registry(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_registry(path, registry):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2)
    os.replace(tmp_path, path)


def classify_upload(registry, indexed_names, safe_name, sha256):
    """Return (action, message): action is "new", "skip" or "reject"."""
    for name, info in registry.items():
        if isinstance(info, dict) and info.get("sha256") == sha256:
            return "skip", f"Already indexed as {name}."
    if safe_name in registry:
        return (
            "reject",
            f"A different file named {safe_name} is already indexed. "
            "Rename the file or press Clear index.",
        )
    if safe_name in set(indexed_names or []):
        return (
            "reject",
            f"A file named {safe_name} is already in the index (older entry without a record). "
            "Rename the file or press Clear index.",
        )
    return "new", ""


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker_main(sys.argv)