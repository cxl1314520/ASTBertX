"""
ASTBertX Periodic Scanner  —  Layer 3 (Scheduled / SAST)
=========================================================
Batch-scans an entire codebase including legacy code and third-party libraries.
Designed to run as a cron job or CI scheduled workflow.

Features
--------
• Recursive directory walk with extension filtering
• Automatic detection of third-party library directories
  (node_modules/, vendor/, site-packages/, venv/, gems/, …)
• Per-language and per-category accuracy breakdown
• Outputs: JSON (machine), CSV (spreadsheet), HTML (human)
• Incremental mode: skip files unchanged since last scan
• Webhook / email notification on new findings

Usage
-----
# Full scan of current repo
python tools/periodic_scan.py --root . --base-path BERTAST --out-dir scan_results

# Include third-party libraries too
python tools/periodic_scan.py --root . --base-path BERTAST --include-third-party

# Incremental scan (only files newer than last report)
python tools/periodic_scan.py --root . --base-path BERTAST --incremental

# Scan specific directories
python tools/periodic_scan.py --dirs src/ legacy/ vendor/ --base-path BERTAST

Cron example (weekly, every Sunday 03:00)
------------------------------------------
0 3 * * 0 cd /project && python BERTAST/tools/periodic_scan.py \\
              --root . --base-path BERTAST --out-dir /var/log/astbertx \\
              --webhook $SLACK_WEBHOOK_URL
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterator

# ── third-party directory heuristics ─────────────────────────────────────────

THIRD_PARTY_DIR_NAMES = {
    "node_modules", "vendor", "site-packages", "dist-packages",
    "venv", ".venv", "env", ".env", "gems", ".gem",
    "__pycache__", ".tox", ".eggs", "build", "dist",
    "bower_components", "jspm_packages",
}

IGNORE_DIRS = {".git", ".svn", ".hg", ".idea", ".vscode", "__pycache__"}

SUPPORTED_EXTENSIONS = {"py", "c", "rb", "pl", "php", "html"}


def _is_third_party(dirpath: str) -> bool:
    parts = set(Path(dirpath).parts)
    return bool(parts & THIRD_PARTY_DIR_NAMES)


def _walk_files(
    roots: list[str],
    include_third_party: bool,
    incremental_since: float | None = None,
) -> Iterator[str]:
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in IGNORE_DIRS
                and (include_third_party or d not in THIRD_PARTY_DIR_NAMES)
            ]
            for fname in filenames:
                ext = Path(fname).suffix.lstrip(".").lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue
                full = os.path.join(dirpath, fname)
                if incremental_since is not None:
                    try:
                        if os.path.getmtime(full) < incremental_since:
                            continue
                    except OSError:
                        pass
                yield full


# ── HTML report ───────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ASTBertX Scan Report — {timestamp}</title>
<style>
  body{{font-family:monospace;background:#0d1117;color:#c9d1d9;padding:2rem}}
  h1{{color:#58a6ff}} h2{{color:#79c0ff;margin-top:2rem}}
  table{{border-collapse:collapse;width:100%;margin-top:1rem}}
  th{{background:#161b22;padding:.5rem 1rem;text-align:left;color:#8b949e}}
  td{{padding:.4rem 1rem;border-bottom:1px solid #21262d}}
  tr:hover td{{background:#161b22}}
  .threat{{color:#f85149}} .warn{{color:#e3b341}} .clean{{color:#3fb950}}
  .badge{{display:inline-block;padding:.1rem .5rem;border-radius:4px;font-size:.85em}}
  .b-threat{{background:#f851491a;color:#f85149;border:1px solid #f85149}}
  .b-warn{{background:#e3b3411a;color:#e3b341;border:1px solid #e3b341}}
  .b-clean{{background:#3fb9501a;color:#3fb950;border:1px solid #3fb950}}
  .stat{{display:inline-block;margin-right:2rem;font-size:1.2em}}
</style>
</head>
<body>
<h1>🛡 ASTBertX Periodic Scan</h1>
<p style="color:#8b949e">{timestamp} &nbsp;|&nbsp; {elapsed}s &nbsp;|&nbsp;
   threshold {threshold:.0%} &nbsp;|&nbsp; mode {mode}</p>

<h2>Summary</h2>
<p>
  <span class="stat">📁 <strong>{total}</strong> files</span>
  <span class="stat threat">⛔ <strong>{n_threats}</strong> threats</span>
  <span class="stat warn">⚠ <strong>{n_warnings}</strong> warnings</span>
  <span class="stat clean">✓ <strong>{n_clean}</strong> clean</span>
</p>

<h2>Label distribution</h2>
<table><tr><th>Category</th><th>Count</th></tr>
{label_rows}
</table>

<h2>Language distribution</h2>
<table><tr><th>Language</th><th>Files</th><th>Threats</th></tr>
{lang_rows}
</table>

<h2>Threat details</h2>
{threat_table}

<h2>Warning details</h2>
{warn_table}
</body></html>
"""

def _file_table(rows: list[dict], base: str) -> str:
    if not rows:
        return "<p style='color:#8b949e'>None.</p>"
    header = ("<table><tr><th>File</th><th>Category</th>"
              "<th>Confidence</th><th>Mode</th></tr>")
    body = ""
    for r in rows:
        rel = os.path.relpath(r["file"], base)
        css = "b-threat" if r["is_threat"] else "b-warn"
        body += (f"<tr><td>{rel}</td>"
                 f"<td><span class='badge {css}'>{r['label']}</span></td>"
                 f"<td>{r['confidence']:.2%}</td>"
                 f"<td>{r.get('mode','?')}</td></tr>")
    return header + body + "</table>"


def _generate_html(report: dict, base: str) -> str:
    m = report["meta"]
    s = report["summary"]

    label_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in sorted(s["by_label"].items(), key=lambda x: -x[1])
    )
    lang_rows = "".join(
        f"<tr><td>{lang}</td><td>{info['total']}</td><td>{info['threats']}</td></tr>"
        for lang, info in sorted(s["by_language"].items())
    )
    return _HTML_TEMPLATE.format(
        timestamp=m["timestamp"],
        elapsed=m["elapsed_seconds"],
        threshold=m["threshold"],
        mode=m["mode"],
        total=m["total_files"],
        n_threats=s["threats"],
        n_warnings=s["warnings"],
        n_clean=s["clean"],
        label_rows=label_rows,
        lang_rows=lang_rows,
        threat_table=_file_table(report["threat_files"], base),
        warn_table=_file_table(report["warning_files"], base),
    )


# ── CSV ───────────────────────────────────────────────────────────────────────

def _write_csv(results: list[dict], path: str, base: str) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "confidence", "is_threat", "mode",
                    "p_benign", "p_overflow", "p_injection", "p_dos", "p_filepath"])
        for r in results:
            p = r.get("probabilities", {})
            w.writerow([
                os.path.relpath(r["file"], base),
                r["label"], f"{r['confidence']:.4f}", r["is_threat"], r.get("mode"),
                f"{p.get('Benign', 0):.4f}", f"{p.get('Overflow', 0):.4f}",
                f"{p.get('Injection', 0):.4f}",
                f"{p.get('Denial of Service', 0):.4f}",
                f"{p.get('File Path', 0):.4f}",
            ])


# ── webhook notification ──────────────────────────────────────────────────────

def _notify_webhook(url: str, report: dict) -> None:
    import urllib.request
    s = report["summary"]
    payload = {
        "text": (
            f"*ASTBertX Scan Complete* — {report['meta']['timestamp']}\n"
            f">Files: {report['meta']['total_files']}  "
            f"Threats: {s['threats']}  Warnings: {s['warnings']}  Clean: {s['clean']}\n"
            + (f">Top threats: " +
               ", ".join(t['label'] for t in report['threat_files'][:5])
               if s['threats'] else ">No threats found ✓")
        )
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:
        print(f"[ASTBertX] Webhook notification failed: {exc}", file=sys.stderr)


# ── main ──────────────────────────────────────────────────────────────────────

THREAT_IDS = {1, 2, 3, 4}
EXT_TO_LANG = {"py": "python", "c": "c", "rb": "ruby",
               "pl": "perl", "php": "php", "html": "html"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ASTBertX periodic batch scanner (Layer 3)")
    parser.add_argument("--base-path", required=True,
                        help="Absolute path to BERTAST/")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--root", help="Repository root to scan recursively")
    group.add_argument("--dirs", nargs="+", help="Explicit directories to scan")
    parser.add_argument("--include-third-party", action="store_true",
                        help="Also scan node_modules/, vendor/, venv/, etc.")
    parser.add_argument("--mode", choices=["fast", "full"], default="full")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--out-dir", default="astbertx_scan",
                        help="Directory to write report files")
    parser.add_argument("--incremental", action="store_true",
                        help="Only scan files changed since last report")
    parser.add_argument("--webhook", default=None,
                        help="Slack/Teams webhook URL for notification")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker threads (default 1)")
    args = parser.parse_args(argv)

    base_path = str(Path(args.base_path).resolve())
    sys.path.insert(0, base_path)

    try:
        from tools.inference_engine import ASTBertXInference
    except ImportError as exc:
        print(f"[ASTBertX] Import error: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Incremental: mtime of last JSON report
    since: float | None = None
    last_report = out_dir / "last_scan.json"
    if args.incremental and last_report.exists():
        since = last_report.stat().st_mtime
        print(f"[ASTBertX] Incremental mode: scanning files newer than "
              f"{datetime.fromtimestamp(since).isoformat()}")

    roots = [args.root] if args.root else args.dirs
    files = list(_walk_files(roots, args.include_third_party, since))

    if not files:
        print("[ASTBertX] No scannable files found.")
        return 0

    print(f"[ASTBertX] {len(files)} files to scan "
          f"(mode={args.mode}, threshold={args.threshold:.0%}) …")

    engine = ASTBertXInference(base_path=base_path,
                               confidence_threshold=args.threshold)
    predict = engine.predict_full if args.mode == "full" else engine.predict_fast

    results: list[dict] = []
    t0 = time.perf_counter()

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(predict, f): f for f in files}
            done = 0
            for fut in as_completed(futures):
                done += 1
                fpath = futures[fut]
                try:
                    r = fut.result()
                    results.append(r)
                    if done % 50 == 0 or r["is_threat"]:
                        tag = "THREAT" if r["is_threat"] and r["confidence"] >= args.threshold \
                              else ("WARN " if r["is_threat"] else "     ")
                        print(f"  [{done}/{len(files)}] {tag}  "
                              f"{r['label']:<22} {os.path.relpath(fpath)}")
                except Exception as exc:
                    print(f"  ERROR  {fpath}: {exc}", file=sys.stderr)
    else:
        for i, fpath in enumerate(files, 1):
            try:
                r = predict(fpath)
                results.append(r)
                if i % 100 == 0 or r["is_threat"]:
                    tag = "THREAT" if r["is_threat"] and r["confidence"] >= args.threshold \
                          else ("WARN " if r["is_threat"] else "     ")
                    print(f"  [{i}/{len(files)}] {tag}  "
                          f"{r['label']:<22} {os.path.relpath(fpath)}")
            except Exception as exc:
                print(f"  ERROR  {fpath}: {exc}", file=sys.stderr)

    elapsed = time.perf_counter() - t0

    # Build statistics
    threats   = [r for r in results if r["is_threat"] and r["confidence"] >= args.threshold]
    warnings  = [r for r in results if r["is_threat"] and r["confidence"] < args.threshold]
    clean     = [r for r in results if not r["is_threat"]]
    by_label: dict[str, int] = defaultdict(int)
    by_lang:  dict[str, dict] = defaultdict(lambda: {"total": 0, "threats": 0})
    for r in results:
        by_label[r["label"]] += 1
        ext = Path(r["file"]).suffix.lstrip(".").lower()
        lang = EXT_TO_LANG.get(ext, ext)
        by_lang[lang]["total"] += 1
        if r["is_threat"]:
            by_lang[lang]["threats"] += 1

    scan_base = str(Path(roots[0]).resolve())
    report = {
        "meta": {
            "tool": "ASTBertX",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "elapsed_seconds": round(elapsed, 2),
            "threshold": args.threshold,
            "mode": args.mode,
            "total_files": len(results),
            "roots": roots,
            "include_third_party": args.include_third_party,
        },
        "summary": {
            "threats": len(threats),
            "warnings": len(warnings),
            "clean": len(clean),
            "by_label": dict(by_label),
            "by_language": {k: dict(v) for k, v in by_lang.items()},
        },
        "threat_files": threats,
        "warning_files": warnings,
    }

    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    json_path = out_dir / f"scan_{stamp}.json"
    csv_path  = out_dir / f"scan_{stamp}.csv"
    html_path = out_dir / f"scan_{stamp}.html"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    last_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    _write_csv(results, str(csv_path), scan_base)
    html_path.write_text(_generate_html(report, scan_base), "utf-8")

    print(f"\n[ASTBertX] Scan complete in {elapsed:.1f}s")
    print(f"  Threats  : {len(threats)}")
    print(f"  Warnings : {len(warnings)}")
    print(f"  Clean    : {len(clean)}")
    print(f"  Reports  → {json_path}")
    print(f"             {csv_path}")
    print(f"             {html_path}\n")

    if args.webhook and (len(threats) + len(warnings)) > 0:
        _notify_webhook(args.webhook, report)

    return 1 if threats else 0


if __name__ == "__main__":
    sys.exit(main())
