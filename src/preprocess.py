import os
import re
import json
import argparse
from typing import List, Tuple, Dict, Any


def _try_extract_with_pymupdf(path: str) -> List[str]:
    try:
        import fitz  # type: ignore
    except Exception:
        return []
    try:
        doc = fitz.open(path)
        pages = []
        for p in doc:
            # "text" preserves reading order reasonably well
            pages.append(p.get_text("text") or "")
        return pages
    except Exception:
        return []


def _try_extract_with_pypdf(path: str) -> List[str]:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception:
        return []
    try:
        reader = PdfReader(path)
        return [(page.extract_text() or "") for page in reader.pages]
    except Exception:
        return []


def extract_pages(path: str) -> List[str]:
    """Extract text pages using available backends.

    Tries PyMuPDF (pymupdf) first for better layout, then PyPDF2.
    Returns list of page strings (may contain newlines), or raises if none work.
    """
    pages = _try_extract_with_pymupdf(path)
    if not pages:
        pages = _try_extract_with_pypdf(path)
    if not pages:
        raise RuntimeError(
            "PDF extraction requires either 'pymupdf' or 'PyPDF2'. "
            "Please install one of them."
        )
    return pages


def _normalize_whitespace(text: str) -> str:
    # Normalize mixed whitespace while preserving paragraph breaks
    # Collapse multiple spaces, normalize Windows/Mac newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace tabs with spaces
    text = text.replace("\t", " ")
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \u00A0]{2,}", " ", text)
    return text.strip()


def _dehyphenate(text: str) -> str:
    # Join hyphenated line breaks: e.g., "astro-\nphysics" -> "astrophysics"
    return re.sub(r"-\n(?=[a-z])", "", text)


def remove_headers_footers(pages: List[str]) -> List[str]:
    """Remove repeated header/footer lines seen on many pages.

    Heuristic: look at first 2 and last 2 non-empty lines across pages;
    if a line appears on >= 50% of pages, strip occurrences.
    """
    first_last_lines = []
    for pg in pages:
        lines = [ln.strip() for ln in pg.splitlines() if ln.strip()]
        head = lines[:2]
        tail = lines[-2:] if len(lines) >= 2 else lines[-len(lines):]
        first_last_lines.extend(head + tail)
    if not pages:
        return pages

    from collections import Counter

    counts = Counter(first_last_lines)
    threshold = max(1, int(len(pages) * 0.5))
    common = {ln for ln, c in counts.items() if c >= threshold}

    cleaned = []
    for pg in pages:
        lines = [ln for ln in pg.splitlines() if ln.strip() and ln.strip() not in common]
        cleaned.append("\n".join(lines))
    return cleaned


SECTION_PATTERNS = [
    ("Abstract", re.compile(r"^abstract\b\.?$", re.IGNORECASE)),
    ("Introduction", re.compile(r"^(?:\d+\.?\s*)?introduction\b", re.IGNORECASE)),
    ("Methods", re.compile(r"^(?:\d+\.?\s*)?(?:materials\s+and\s+methods|methods)\b", re.IGNORECASE)),
    ("Results", re.compile(r"^(?:\d+\.?\s*)?results?\b", re.IGNORECASE)),
    ("Discussion", re.compile(r"^(?:\d+\.?\s*)?discussion\b", re.IGNORECASE)),
    ("Conclusion", re.compile(r"^(?:\d+\.?\s*)?conclusions?\b", re.IGNORECASE)),
    ("References", re.compile(r"^(?:\d+\.?\s*)?references\b", re.IGNORECASE)),
    ("Appendix", re.compile(r"^(?:appendix|supplementary)\b", re.IGNORECASE)),
]


def detect_sections(pages: List[str]) -> List[Dict[str, Any]]:
    """Detect common sections by scanning per-page lines.

    Returns a list of dicts: {name, page_start, page_end, text}.
    If no section headings found, returns a single 'Body' section spanning all pages.
    """
    # Prepare per-page lines
    page_lines = [[ln for ln in pg.splitlines()] for pg in pages]

    # Find headings
    hits: List[Tuple[int, int, str]] = []  # (page_idx, line_idx, name)
    for p_idx, lines in enumerate(page_lines):
        for l_idx, line in enumerate(lines):
            s = line.strip()
            if not s:
                continue
            for name, pat in SECTION_PATTERNS:
                if pat.match(s):
                    hits.append((p_idx, l_idx, name))
                    break

    # If no headings, return entire body as one section
    if not hits:
        return [{
            "name": "Body",
            "page_start": 0,
            "page_end": max(0, len(pages) - 1),
            "text": "\n\n".join(pages),
        }]

    # Sort and build slices between headings
    hits.sort()
    sections: List[Dict[str, Any]] = []
    for i, (sp, sl, sname) in enumerate(hits):
        # Determine end bound
        if i + 1 < len(hits):
            ep, el, _ = hits[i + 1]
        else:
            ep, el = len(pages) - 1, None  # to end of doc

        # Collect text from [sp:ep] with in-page slicing
        buf: List[str] = []
        for p in range(sp, ep + 1):
            lines = page_lines[p]
            if p == sp:
                start_line = sl
            else:
                start_line = 0
            if p == ep and el is not None:
                end_line = el
            else:
                end_line = len(lines)
            if 0 <= start_line < end_line <= len(lines):
                buf.append("\n".join(lines[start_line:end_line]))
            elif p == sp == ep and el is not None and el > sl:
                buf.append("\n".join(lines[sl:el]))

        sec_text = _normalize_whitespace(_dehyphenate("\n".join(buf)))
        if sec_text:
            sections.append({
                "name": sname,
                "page_start": sp,
                "page_end": ep,
                "text": sec_text,
            })

    # Merge adjacent duplicate-named sections (rare but possible)
    # Group all sections by name to merge non-adjacent parts
    grouped_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for sec in sections:
        if sec["name"] not in grouped_by_name:
            grouped_by_name[sec["name"]] = []
        grouped_by_name[sec["name"]].append(sec)

    merged: List[Dict[str, Any]] = []
    for sname, parts in grouped_by_name.items():
        if not parts: continue
        full_text = "\n\n".join(p["text"] for p in parts)
        page_start = min(p["page_start"] for p in parts)
        page_end = max(p["page_end"] for p in parts)
        merged.append({"name": sname, "page_start": page_start, "page_end": page_end, "text": full_text})

    # Re-sort merged sections by their first appearance
    merged.sort(key=lambda s: s["page_start"])
    return merged


def split_sentences(text: str) -> List[str]:
    # Basic sentence splitter, avoids heavy dependencies
    parts = re.split(r"(?<=[.!?])\s+(?=(?:[A-Z\[(]|\d))", text)
    return [p.strip() for p in parts if p.strip()]


def make_chunks(text: str, max_chars: int = 3000, overlap_ratio: float = 0.2) -> List[str]:
    sents = split_sentences(text)
    if not sents:
        return [text] if text else []

    chunks: List[str] = []
    i = 0
    min_overlap = max(1, int(len(sents) * overlap_ratio * 0.2))  # small floor
    while i < len(sents):
        buf = []
        length = 0
        j = i
        while j < len(sents) and (length + len(sents[j]) + 1) <= max_chars:
            buf.append(sents[j])
            length += len(sents[j]) + 1
            j += 1
        if not buf:
            # Single sentence longer than max_chars, hard cut
            buf = [sents[j]]
            j += 1
        chunks.append(" ".join(buf))
        # Overlap by sentences approx 20% of this window
        overlap = max(min_overlap, int(len(buf) * overlap_ratio))
        if overlap <= 0:
            i = j
        else:
            i = max(i + 1, j - overlap)
    return chunks


def extract_numeric_citations(text: str) -> List[int]:
    nums: List[int] = []
    for m in re.findall(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]", text):
        parts = re.split(r"\s*,\s*", m)
        for p in parts:
            try:
                nums.append(int(p))
            except ValueError:
                pass
    return sorted(set(nums))


def extract_author_year_citations(text: str) -> List[str]:
    matches = re.findall(r"\(([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?(?:\s*&\s*[A-Z][A-Za-z\-]+)?),\s*(\d{4}[a-z]?)\)", text)
    return sorted({f"{a}, {y}" for a, y in matches})


def parse_filename(path: str) -> Tuple[str, str]:
    """Return (doc_id, domain) from filename like '2509.08999v1 Astrophysics.pdf'."""
    base = os.path.basename(path)
    name, _ext = os.path.splitext(base)
    parts = name.split(" ", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return name, "Unknown"


def process_pdf(path: str, out_writer, doc_index: Dict[str, Any], max_chars: int = 3000, overlap: float = 0.2) -> int:
    doc_id, domain = parse_filename(path)
    pages = extract_pages(path)
    pages = [ _normalize_whitespace(_dehyphenate(p)) for p in pages ]
    pages = remove_headers_footers(pages)
    sections = detect_sections(pages)

    total_chunks = 0
    for sec in sections:
        sec_name = sec["name"]
        sec_text = sec["text"]
        sec_chunks = make_chunks(sec_text, max_chars=max_chars, overlap_ratio=overlap)
        for idx, ch in enumerate(sec_chunks):
            # Ensure globally unique chunk IDs by including page range
            ps = sec.get("page_start")
            pe = sec.get("page_end")
            chunk_id = f"{doc_id}:{sec_name}:{ps}-{pe}:{idx:04d}"
            item = {
                "id": chunk_id,
                "doc_id": doc_id,
                "domain": domain,
                "section": sec_name,
                "page_start": sec.get("page_start"),
                "page_end": sec.get("page_end"),
                "chunk_idx": idx,
                "chunk_type": "text",
                "citations_numeric": extract_numeric_citations(ch),
                "citations_author_year": extract_author_year_citations(ch),
                "text": ch,
                "source_path": path,
            }
            out_writer.write(json.dumps(item, ensure_ascii=False) + "\n")
            total_chunks += 1

    doc_index[doc_id] = {
        "doc_id": doc_id,
        "domain": domain,
        "n_pages": len(pages),
        "sections": [{"name": s["name"], "page_start": s["page_start"], "page_end": s["page_end"]} for s in sections],
        "chunk_count": total_chunks,
        "source_path": path,
    }
    return total_chunks


def run(data_dir: str, out_dir: str, max_chars: int = 3000, overlap: float = 0.2) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "chunks.jsonl")
    index_json = os.path.join(out_dir, "docs_index.json")

    pdf_paths = [
        os.path.join(data_dir, f)
        for f in sorted(os.listdir(data_dir))
        if f.lower().endswith(".pdf")
    ]
    if not pdf_paths:
        print(f"No PDFs found in: {data_dir}")
        return

    doc_index: Dict[str, Any] = {}
    total_docs = 0
    total_chunks = 0
    with open(out_jsonl, "w", encoding="utf-8") as w:
        for p in pdf_paths:
            try:
                c = process_pdf(p, w, doc_index, max_chars=max_chars, overlap=overlap)
                total_chunks += c
                total_docs += 1
                print(f"Processed {os.path.basename(p)} -> {c} chunks")
            except Exception as e:
                print(f"Failed {os.path.basename(p)}: {e}")

    with open(index_json, "w", encoding="utf-8") as f:
        json.dump(doc_index, f, ensure_ascii=False, indent=2)

    print(f"Done. Docs: {total_docs}, Chunks: {total_chunks}")
    print(f"Chunks: {out_jsonl}")
    print(f"Index:  {index_json}")


def main():
    ap = argparse.ArgumentParser(description="Minimal PDF -> chunks.jsonl preprocessor")
    ap.add_argument("--data-dir", default="/Users/jonch/Documents/NUS CS6101 RAG/Code/Shayamal Oct4 /data", help="Input PDFs directory")
    ap.add_argument("--out-dir", default="processed", help="Output directory")
    ap.add_argument("--max-chars", type=int, default=3000, help="Max characters per chunk")
    ap.add_argument("--overlap", type=float, default=0.2, help="Chunk overlap ratio (0-1)")
    args = ap.parse_args()
    run(args.data_dir, args.out_dir, max_chars=args.max_chars, overlap=args.overlap)


if __name__ == "__main__":
    main()
