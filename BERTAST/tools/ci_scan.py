"""
ASTBertX CI/CD Scanner  —  Layer 2 (Build Server)
==================================================
Full BERT + AST fusion scan for every changed file in a PR / push.
Outputs a machine-readable JSON report and a human-readable summary.

Exit codes
----------
  0  All files clean (or only low-confidence warnings)
  1  Threats found above threshold
  2  Configuration / runtime error

Usage
-----
# Scan files changed relative to main branch
python tools/ci_scan.py --diff-base origin/main --base-path /abs/path/to/BERTAST

# Scan an explicit list of files
python tools/ci_scan.py --files src/exploit.py src/upload.php --base-path ./BERTAST

# Fast mode (BERT only, skip AST)
python tools/ci_scan.py --diff-base origin/main --mode fast --base-path ./BERTAST

GitHub Actions example
----------------------
- name: ASTBertX Security Scan
  run: |
    python BERTAST/tools/ci_scan.py \\
      --diff-base origin/main \\
      --base-path BERTAST \\
      --report-out astbertx-report.json \\
      --threshold 0.70
- name: Upload scan report
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: astbertx-report
    path: astbertx-report.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _changed_files(diff_base: str) -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--name-only", "--diff-filter=ACM", diff_base + "...HEAD"],
        text=True,
    )
    repo_root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True).strip()
    return [str(Path(repo_root) / p) for p in out.splitlines() if p.strip()]


def _build_report(results: list[dict], threshold: float, elapsed: float) -> dict:
    threats = [r for r in results if r["is_threat"] and r["confidence"] >= threshold]
    warnings = [r for r in results if r["is_threat"] and r["confidence"] < threshold]
    clean = [r for r in results if not r["is_threat"]]

    by_label: dict[str, int] = {}
    for r in results:
        by_label[r["label"]] = by_label.get(r["label"], 0) + 1

    return {
        "meta": {
            "tool": "ASTBertX",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "elapsed_seconds": round(elapsed, 2),
            "threshold": threshold,
            "total_files": len(results),
        },
        "summary": {
            "threats": len(threats),
            "warnings": len(warnings),
            "clean": len(clean),
            "by_label": by_label,
        },
        "threats": threats,
        "warnings": warnings,
        "clean": clean,
    }


def _print_summary(report: dict, use_color: bool = True) -> None:
    RED    = "\033[31m" if use_color else ""
    GREEN  = "\033[32m" if use_color else ""
    YELLOW = "\033[33m" if use_color else ""
    CYAN   = "\033[36m" if use_color else ""
    BOLD   = "\033[1m"  if use_color else ""
    RESET  = "\033[0m"  if use_color else ""

    m = report["meta"]
    s = report["summary"]
    print(f"\n{CYAN}{BOLD}{'─'*60}{RESET}")
    print(f"{CYAN}{BOLD}  ASTBertX CI Scan Report{RESET}")
    print(f"{CYAN}  {m['timestamp']}   {m['elapsed_seconds']}s{RESET}")
    print(f"{CYAN}{'─'*60}{RESET}")
    print(f"  Total files scanned : {m['total_files']}")
    print(f"  {RED}Threats (blocked)   : {s['threats']}{RESET}")
    print(f"  {YELLOW}Warnings (review)   : {s['warnings']}{RESET}")
    print(f"  {GREEN}Clean               : {s['clean']}{RESET}")

    if s["threats"]:
        print(f"\n{RED}{BOLD}  THREATS:{RESET}")
        for r in report["threats"]:
            rel = os.path.relpath(r["file"])
            print(f"{RED}    ✗ [{r['confidence']:.0%}] {r['label']:<22} {rel}{RESET}")

    if s["warnings"]:
        print(f"\n{YELLOW}  WARNINGS:{RESET}")
        for r in report["warnings"]:
            rel = os.path.relpath(r["file"])
            print(f"{YELLOW}    ⚠ [{r['confidence']:.0%}] {r['label']:<22} {rel}{RESET}")

    print(f"\n{CYAN}{'─'*60}{RESET}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ASTBertX CI/CD scanner (Layer 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-path", required=True,
                        help="Absolute path to BERTAST/ directory")
    parser.add_argument("--diff-base", default=None,
                        help="Git ref to diff against (e.g. origin/main)")
    parser.add_argument("--files", nargs="*", default=None,
                        help="Explicit list of files to scan")
    parser.add_argument("--mode", choices=["fast", "full"], default="full",
                        help="fast=BERT only, full=BERT+AST (default: full)")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Confidence threshold for blocking (default: 0.70)")
    parser.add_argument("--report-out", default=None,
                        help="Write JSON report to this path")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)

    base_path = str(Path(args.base_path).resolve())
    sys.path.insert(0, base_path)

    try:
        from tools.inference_engine import ASTBertXInference
    except ImportError as exc:
        print(f"[ASTBertX] Import error: {exc}\n"
              f"  Ensure --base-path points to the BERTAST/ directory.", file=sys.stderr)
        return 2

    # Collect files
    if args.files:
        candidates = [str(Path(f).resolve()) for f in args.files]
    elif args.diff_base:
        try:
            candidates = _changed_files(args.diff_base)
        except subprocess.CalledProcessError as exc:
            print(f"[ASTBertX] git diff failed: {exc}", file=sys.stderr)
            return 2
    else:
        print("[ASTBertX] Provide --diff-base or --files.", file=sys.stderr)
        return 2

    engine = ASTBertXInference(base_path=base_path,
                               confidence_threshold=args.threshold)
    candidates = [f for f in candidates if engine.is_supported(f)]

    if not candidates:
        print("[ASTBertX] No scannable files found.")
        return 0

    color = sys.stdout.isatty() and not args.no_color
    print(f"[ASTBertX] Scanning {len(candidates)} file(s) "
          f"(mode={args.mode}, threshold={args.threshold:.0%}) …")

    import time
    t0 = time.perf_counter()
    results: list[dict] = []
    predict = engine.predict_full if args.mode == "full" else engine.predict_fast

    for fpath in candidates:
        try:
            r = predict(fpath)
        except Exception as exc:
            print(f"  ERROR  {fpath}: {exc}", file=sys.stderr)
            continue
        results.append(r)
        rel = os.path.relpath(fpath)
        tag = "THREAT" if (r["is_threat"] and r["confidence"] >= args.threshold) \
              else ("WARN " if r["is_threat"] else "CLEAN")
        print(f"  {tag}  [{r['confidence']:.0%}] {r['label']:<22} {rel}")

    elapsed = time.perf_counter() - t0
    report = _build_report(results, args.threshold, elapsed)
    _print_summary(report, use_color=color)

    if args.report_out:
        Path(args.report_out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[ASTBertX] Report saved → {args.report_out}")

    return 1 if report["summary"]["threats"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
