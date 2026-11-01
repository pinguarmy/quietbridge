"""Tuning and replay module for Lipflow cleanup models.

Replays known-label onboarding clips or annotated raw-candidates JSONL against
basic/local/gemini backends with faithful/polish styles.
Computes:
- Normalized WER only for faithful style (avoids misleading WER on intentional polish/rewrite).
- P50/P95 cleanup and cleanup-path latencies (relabelled honestly; does not claim end-to-end key-release latency).
- Semantic risk flags (canonicalized negations, numbers/digits, entities/names).
- Outputs concise local JSON and Markdown reports with sample-level backend & fallback tracking.
- Local execution: video frames and clips are never uploaded; when cloud backends (gemini/claude)
  are evaluated, candidate and context text is transmitted to the configured endpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Callable, Iterable, Sequence

# Lazy imports to ensure --help and metrics can run without heavy models or PyTorch
def _words_of(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def normalize_for_wer(text: str) -> str:
    """Normalize text for speech recognition WER comparison (lowercase, standard numbers, stripped punctuation)."""
    from .cleanup import numbers_to_digits
    t = numbers_to_digits(text.strip().lower())
    words = _words_of(t)
    return " ".join(words)


def compute_wer(hyp: str, ref: str) -> tuple[int, int, float]:
    """Calculate edit distance, reference length, and WER.
    Returns (edits, ref_len, wer_rate).
    """
    h_words = normalize_for_wer(hyp).split()
    r_words = normalize_for_wer(ref).split()
    if not r_words:
        return len(h_words), 0, 1.0 if h_words else 0.0

    d = list(range(len(r_words) + 1))
    for i, a in enumerate(h_words, 1):
        prev, d[0] = d[0], i
        for j, b in enumerate(r_words, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a != b))
    edits = d[-1]
    return edits, len(r_words), edits / max(len(r_words), 1)


# Semantic risk detection
_BASE_NEGATIONS = {
    "not", "no", "never", "none", "neither", "nor", "nothing", "nowhere",
}

_CONTRACTION_TO_NEGATION = {
    "cannot": "not", "cant": "not", "can't": "not",
    "dont": "not", "don't": "not",
    "wont": "not", "won't": "not",
    "isnt": "not", "isn't": "not",
    "arent": "not", "aren't": "not",
    "wasnt": "not", "wasn't": "not",
    "werent": "not", "weren't": "not",
    "hasnt": "not", "hasn't": "not",
    "havent": "not", "haven't": "not",
    "hadnt": "not", "hadn't": "not",
    "doesnt": "not", "doesn't": "not",
    "didnt": "not", "didn't": "not",
    "shouldnt": "not", "shouldn't": "not",
    "wouldnt": "not", "wouldn't": "not",
    "couldnt": "not", "couldn't": "not",
    "mustnt": "not", "mustn't": "not",
    "mightnt": "not", "mightn't": "not",
    "aint": "not", "ain't": "not",
}


def extract_numbers(text: str) -> set[str]:
    from .cleanup import numbers_to_digits
    norm = numbers_to_digits(text.lower())
    digits = set(re.findall(r"\b\d+\b", norm))
    return digits


def extract_negations(text: str) -> set[str]:
    """Extract canonicalized negative markers (mapping contractions e.g. don't -> not)."""
    words = _words_of(text)
    negs = set()
    for w in words:
        if w in _CONTRACTION_TO_NEGATION:
            negs.add(_CONTRACTION_TO_NEGATION[w])
        elif w in _BASE_NEGATIONS:
            negs.add(w)
    return negs


def extract_potential_entities(text: str, custom_words: Iterable[str] | None = None) -> set[str]:
    """Find capitalized words/entities (or matching custom vocabulary), ignoring sentence-start capitalization."""
    entities = set()
    if custom_words:
        for w in custom_words:
            if re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE):
                entities.add(w.lower())

    # Split into sentences by punctuation or newlines
    sentences = re.split(r'(?:[.?!]+\s+|\n+)', text.strip())
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        words = list(re.finditer(r"\b[A-Za-z0-9'-]+\b", s))
        if not words:
            continue
        # The first word is capitalized due to grammar at sentence start; only inspect mid-sentence tokens
        for m in words[1:]:
            token = m.group(0)
            if token and token[0].isupper():
                tl = token.lower()
                if tl not in {"i", "a", "the", "this", "that", "it"}:
                    entities.add(tl)
    return entities


def detect_semantic_risks(
    reference: str,
    hypothesis: str,
    custom_words: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Flag semantic divergences:
    - negation_mismatch: negation flipped or dropped
    - number_mismatch: numbers added or lost
    - entity_mismatch: key custom entity or capitalized entity modified
    """
    ref_neg = extract_negations(reference)
    hyp_neg = extract_negations(hypothesis)
    neg_diff = (ref_neg - hyp_neg) | (hyp_neg - ref_neg)

    ref_num = extract_numbers(reference)
    hyp_num = extract_numbers(hypothesis)
    num_diff = (ref_num - hyp_num) | (hyp_num - ref_num)

    ref_ent = extract_potential_entities(reference, custom_words)
    hyp_ent = extract_potential_entities(hypothesis, custom_words)
    ent_diff = (ref_ent - hyp_ent)

    return {
        "negation_risk": bool(neg_diff),
        "negation_diff": sorted(list(neg_diff)),
        "number_risk": bool(num_diff),
        "number_diff": sorted(list(num_diff)),
        "entity_risk": bool(ent_diff),
        "entity_diff": sorted(list(ent_diff)),
        "has_risk": bool(neg_diff or num_diff or ent_diff),
    }


def percentile(vals: Sequence[float], p: float) -> float:
    """Calculate p-th percentile (0 <= p <= 100) on sorted values."""
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    d = k - f
    return round(s[f] + d * (s[c] - s[f]), 3)


def load_dataset(input_path: str | None = None) -> list[dict[str, Any]]:
    """Load evaluation items:
    If input_path is a JSONL file with lines {"candidates": [...], "reference": "..."} or {"raw": [...], "gold": "..."}.
    Unannotated history where 'text' is only app-generated output is skipped unless explicitly marked reference/gold/verified.
    If input_path is None or points to onboarding clips directory, loads saved clips npz files with raw predictions.
    Items without predictions are skipped to prevent using gold text as candidates.
    """
    if input_path and input_path.endswith((".jsonl", ".json")):
        items = []
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                d = json.loads(s)
                candidates = d.get("candidates") or d.get("raw") or []
                if isinstance(candidates, str):
                    candidates = [candidates] if candidates.strip() else []

                # In JSONL logs, 'text' is app-generated output, not ground truth.
                # Require explicit reference/gold or verified label marker; do not score unannotated history.
                ref = d.get("reference") or d.get("gold") or d.get("ground_truth")
                if not ref and (d.get("verified") or d.get("labeled")):
                    ref = d.get("label") or d.get("text")

                if not ref:
                    continue

                ref = str(ref).strip()
                item_id = str(d.get("id") or len(items) + 1)
                vsr_ms = d.get("vsr_ms")
                items.append({
                    "candidates": candidates,
                    "reference": ref,
                    "id": item_id,
                    "vsr_ms": vsr_ms,
                })
        return items

    # Load from onboarding clips
    from .practice import CLIPS
    import glob
    import numpy as np

    folder = input_path if (input_path and os.path.isdir(input_path)) else CLIPS
    items = []
    for p in sorted(glob.glob(os.path.join(folder, "*.npz"))):
        try:
            d = np.load(p, allow_pickle=True)
            text = str(d.get("text", "")).strip()

            candidates: list[str] = []
            if "candidates" in d:
                c = d["candidates"]
                if isinstance(c, np.ndarray):
                    c = c.tolist()
                if isinstance(c, str) and c.strip():
                    candidates = [c.strip()]
                elif isinstance(c, list):
                    candidates = [str(x).strip() for x in c if str(x).strip()]
            if not candidates and "raw" in d:
                r = d["raw"]
                if isinstance(r, np.ndarray):
                    r = r.tolist()
                if isinstance(r, list):
                    candidates = [str(x).strip() for x in r if str(x).strip()]
                elif isinstance(r, str) and r.strip():
                    candidates = [r.strip()]

            # Must skip items without raw predictions; never substitute reference as predicted candidate
            if not candidates:
                continue

            items.append({
                "candidates": candidates,
                "reference": text,
                "id": os.path.basename(p),
                "path": p,
            })
        except Exception:
            continue
    return items


def replay_evaluation(
    dataset: list[dict[str, Any]],
    backend: str = "auto",
    style: str = "faithful",
    cleaner_instance: Any = None,
    custom_words: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Replay dataset against the specified backend and style.
    Evaluates:
    - normalized WER (only computed when style == 'faithful' and reference is non-empty)
    - P50 and P95 cleanup-path latency (relabelled honestly; does not claim end-to-end key-release latency)
    - Semantic risk rate (canonicalized negations, numbers, entities)
    - Reports sample-level actual backend and fallback status from cleaner.last_result
    """
    if limit:
        dataset = dataset[:limit]

    # Instantiate or configure cleaner
    if cleaner_instance is not None:
        cleaner = cleaner_instance
    else:
        from .cleanup import Cleaner
        cleaner = Cleaner(backend=backend, style=style)

    # Words
    if custom_words is None:
        from .vocab import load as load_vocab
        custom_words = load_vocab()

    total_edits = 0
    total_ref_words = 0
    cleanup_latencies: list[float] = []
    total_latencies: list[float] = []
    samples_results: list[dict[str, Any]] = []

    neg_risk_count = 0
    num_risk_count = 0
    ent_risk_count = 0
    any_risk_count = 0

    for item in dataset:
        cands = item.get("candidates") or []
        ref = item.get("reference") or ""
        t0 = time.perf_counter()
        
        # Invoke cleaner
        out = cleaner(cands, words=custom_words)
        elapsed_total_ms = (time.perf_counter() - t0) * 1000.0

        # Retrieve last_result metadata if supported
        meta = getattr(cleaner, "last_result", {}) or {}
        backend_used = meta.get("backend", getattr(cleaner, "backend", backend))
        fallback_used = meta.get("fallback")
        elapsed_backend_ms = meta.get("cleanup_ms")
        if elapsed_backend_ms is None:
            elapsed_backend_ms = round(elapsed_total_ms, 1)

        cleanup_latencies.append(float(elapsed_backend_ms))
        total_latencies.append(round(elapsed_total_ms, 1))

        # Check for known VSR measured time in item (if available from npz/history)
        vsr_ms = item.get("vsr_ms")

        # Risk check (only if reference is available)
        risks = detect_semantic_risks(ref, out, custom_words) if ref else {
            "negation_risk": False, "negation_diff": [],
            "number_risk": False, "number_diff": [],
            "entity_risk": False, "entity_diff": [],
            "has_risk": False,
        }
        if risks["negation_risk"]:
            neg_risk_count += 1
        if risks["number_risk"]:
            num_risk_count += 1
        if risks["entity_risk"]:
            ent_risk_count += 1
        if risks["has_risk"]:
            any_risk_count += 1

        sample_stat: dict[str, Any] = {
            "id": item.get("id"),
            "backend": backend_used,
            "fallback": fallback_used,
            "candidates": cands,
            "reference": ref,
            "output": out,
            "cleanup_backend_ms": round(float(elapsed_backend_ms), 1),
            "cleanup_path_ms": round(elapsed_total_ms, 1),
            "total_ms": round(elapsed_total_ms + vsr_ms, 1) if vsr_ms is not None else round(elapsed_total_ms, 1),
            "latency_scope": "vsr_plus_cleanup_replay" if vsr_ms is not None else "cleanup_path_only",
            "risks": risks,
        }

        # Calculate WER only if faithful style
        if style == "faithful" and ref:
            edits, r_len, wer_rate = compute_wer(out, ref)
            total_edits += edits
            total_ref_words += r_len
            sample_stat["wer"] = round(wer_rate, 4)

        samples_results.append(sample_stat)

    n_samples = len(dataset)
    res: dict[str, Any] = {
        "backend": getattr(cleaner, "backend", backend),
        "style": style,
        "model": getattr(cleaner, "model", None),
        "sample_count": n_samples,
        "cleanup_latency_ms": {
            "p50": percentile(cleanup_latencies, 50),
            "p95": percentile(cleanup_latencies, 95),
            "mean": round(sum(cleanup_latencies) / max(n_samples, 1), 1),
        },
        "cleanup_path_latency_ms": {
            "p50": percentile(total_latencies, 50),
            "p95": percentile(total_latencies, 95),
            "mean": round(sum(total_latencies) / max(n_samples, 1), 1),
        },
        "total_latency_ms": {
            "p50": percentile(total_latencies, 50),
            "p95": percentile(total_latencies, 95),
            "mean": round(sum(total_latencies) / max(n_samples, 1), 1),
        },
        "latency_note": "Replay measures cleanup-path latency only. It does not represent end-to-end key-release latency (which includes camera capture, ROI extraction, and VSR decoding).",
        "semantic_risks": {
            "negation_risk_rate": round(neg_risk_count / max(n_samples, 1), 4),
            "number_risk_rate": round(num_risk_count / max(n_samples, 1), 4),
            "entity_risk_rate": round(ent_risk_count / max(n_samples, 1), 4),
            "any_risk_rate": round(any_risk_count / max(n_samples, 1), 4),
        },
        "samples": samples_results,
    }

    if style == "faithful":
        overall_wer = (total_edits / max(total_ref_words, 1)) if total_ref_words else 0.0
        res["wer"] = round(overall_wer, 4)
        res["wer_note"] = f"Calculated over {total_ref_words} reference words across {n_samples} items."
    else:
        res["wer"] = None
        res["wer_note"] = "WER omitted for 'polish' style to avoid misleading scoring on intentional rewrites."

    return res


def generate_markdown_report(report: dict[str, Any]) -> str:
    """Format evaluation results into a clean, concise Markdown summary."""
    lines = [
        f"# Lipflow Model Tuning & Replay Report",
        "",
        f"- **Backend**: `{report.get('backend')}` (Style: `{report.get('style')}`, Model: `{report.get('model')}`)",
        f"- **Evaluated Samples**: {report.get('sample_count')}",
        "",
        "## Performance & Latency",
        "> **Notice**: Replay measures cleanup-path latency only. It does not measure end-to-end key-release latency (which also includes camera frame capture, ROI extraction, and neural VSR decoding).",
        f"- **Backend Cleanup Latency**: P50 = {report['cleanup_latency_ms']['p50']} ms | P95 = {report['cleanup_latency_ms']['p95']} ms | Mean = {report['cleanup_latency_ms']['mean']} ms",
        f"- **Total Cleanup-Path Latency**: P50 = {report['total_latency_ms']['p50']} ms | P95 = {report['total_latency_ms']['p95']} ms",
        "",
        "## Accuracy & Semantic Fidelity",
    ]

    if report.get("style") == "faithful":
        wer_pct = f"{report.get('wer', 0.0) * 100:.2f}%" if report.get("wer") is not None else "N/A"
        lines.append(f"- **Normalized WER (Faithful)**: **{wer_pct}** ({report.get('wer_note')})")
    else:
        lines.append(f"- **WER**: *Omitted* ({report.get('wer_note')})")

    risks = report.get("semantic_risks", {})
    lines.extend([
        f"- **Semantic Risk Rate (Any)**: {risks.get('any_risk_rate', 0.0) * 100:.1f}%",
        f"  - Negation Mismatch: {risks.get('negation_risk_rate', 0.0) * 100:.1f}%",
        f"  - Number Mismatch: {risks.get('number_risk_rate', 0.0) * 100:.1f}%",
        f"  - Entity Mismatch: {risks.get('entity_risk_rate', 0.0) * 100:.1f}%",
        "",
        "## Sample Highlights (First 5)",
        "| ID | Backend (Fallback) | Reference | Output | Risks |",
        "|---|---|---|---|---|",
    ])

    for s in report.get("samples", [])[:5]:
        sid = s.get("id", "")
        bk = s.get("backend", report.get("backend", ""))
        fb = s.get("fallback")
        backend_str = f"{bk} (fallback from {fb})" if fb else str(bk)
        ref = s.get("reference", "").replace("|", "\\|")
        out = s.get("output", "").replace("|", "\\|")
        risk_flags = []
        r = s.get("risks", {})
        if r.get("negation_risk"):
            risk_flags.append(f"Neg:{','.join(r['negation_diff'])}")
        if r.get("number_risk"):
            risk_flags.append(f"Num:{','.join(r['number_diff'])}")
        if r.get("entity_risk"):
            risk_flags.append(f"Ent:{','.join(r['entity_diff'])}")
        risk_str = " ".join(risk_flags) if risk_flags else "None"
        lines.append(f"| {sid} | {backend_str} | {ref} | {out} | {risk_str} |")

    lines.append("")
    return "\n".join(lines)


def tune_handler(args: Any = None) -> int:
    """CLI handler for tuning and replay."""
    parser = argparse.ArgumentParser(
        prog="lipflow tune",
        description="Replay clips or candidate logs against cleanup backends and evaluate latency, WER, and semantic risks.",
    )
    parser.add_argument("--input", type=str, default=None, help="Path to input JSONL or clips directory")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "basic", "local", "gemini", "claude", "ollama"], help="Cleanup backend")
    parser.add_argument("--style", type=str, default="faithful", choices=["faithful", "polish"], help="Cleanup style")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--out-json", type=str, default=None, help="Path to save output JSON report")
    parser.add_argument("--out-md", type=str, default=None, help="Path to save output Markdown report")

    parsed = parser.parse_args(args)

    print(f"[tune] Loading dataset from {parsed.input or 'default onboarding clips'}...")
    ds = load_dataset(parsed.input)
    if not ds:
        print("[tune] No clips or items found to evaluate. You can provide --input <file.jsonl> or record onboarding practice clips first.")
        return 0

    print(f"[tune] Replaying {min(len(ds), parsed.limit or len(ds))} items with backend={parsed.backend}, style={parsed.style}...")
    report = replay_evaluation(ds, backend=parsed.backend, style=parsed.style, limit=parsed.limit)

    md_report = generate_markdown_report(report)
    print("\n" + md_report)

    if parsed.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(parsed.out_json)), exist_ok=True)
        with open(parsed.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[tune] Saved JSON report to {parsed.out_json}")

    if parsed.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(parsed.out_md)), exist_ok=True)
        with open(parsed.out_md, "w", encoding="utf-8") as f:
            f.write(md_report)
        print(f"[tune] Saved Markdown report to {parsed.out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(tune_handler(sys.argv[1:]))
