"""
ASTBertX Pre-commit Hook  —  Layer 1 (Developer Workstation)
=============================================================
Scans only staged files using the fast BERT-only path.
Blocks the commit when a threat exceeds the confidence threshold.

Install
-------
    bash tools/install_hooks.sh

Or manually:
    cp tools/precommit_hook.py .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

Configuration  (.astbertx.yml at repo root)
-------------------------------------------
precommit:
  base_path: /abs/path/to/BERTAST
  confidence_threshold: 0.75   # block if P(threat) >= this
  block_on_threat: true        # set false to warn-only
  skip_patterns:               # glob patterns to skip
    - "tests/**"
    - "*.min.js"
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ── locate repo root & load config ───────────────────────────────────────────

REPO_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"],
                            text=True).strip()
)

def _load_config() -> dict:
    cfg_path = REPO_ROOT / ".astbertx.yml"
    defaults = {
        "base_path": str(REPO_ROOT / "BERTAST"),
        "confidence_threshold": 0.75,
        "block_on_threat": True,
        "skip_patterns": [],
    }
    if not cfg_path.exists():
        return defaults
    try:
        import yaml  # type: ignore
        with open(cfg_path) as f:
            data = yaml.safe_load(f) or {}
        cfg = data.get("precommit", {})
        defaults.update(cfg)
    except Exception:
        pass
    return defaults

# ── terminal colours ──────────────────────────────────────────────────────────

RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"

def _c(code: str, text: str) -> str:
    return f"{code}{text}{RESET}" if _supports_color() else text

# ── staged files ──────────────────────────────────────────────────────────────

def _staged_files() -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        text=True,
    )
    return [str(REPO_ROOT / p) for p in out.splitlines() if p.strip()]

def _matches_skip(path: str, patterns: list[str]) -> bool:
    from fnmatch import fnmatch
    rel = os.path.relpath(path, str(REPO_ROOT))
    return any(fnmatch(rel, pat) for pat in patterns)

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = _load_config()
    base_path   = cfg["base_path"]
    threshold   = float(cfg["confidence_threshold"])
    block       = bool(cfg["block_on_threat"])
    skip_pats   = cfg["skip_patterns"]

    sys.path.insert(0, base_path)
    from tools.inference_engine import ASTBertXInference
    engine = ASTBertXInference(base_path=base_path,
                               confidence_threshold=threshold)

    candidates = [
        f for f in _staged_files()
        if engine.is_supported(f) and not _matches_skip(f, skip_pats)
    ]

    if not candidates:
        print(_c(GREEN, "[ASTBertX] No scannable files staged. ✓"))
        return 0

    print(_c(CYAN, f"\n[ASTBertX] Pre-commit scan ({len(candidates)} file(s)) …\n"))

    threats: list[dict] = []
    for fpath in candidates:
        try:
            result = engine.predict_fast(fpath)
        except Exception as exc:
            print(_c(YELLOW, f"  WARN  {fpath}: {exc}"))
            continue

        rel = os.path.relpath(fpath, str(REPO_ROOT))
        label = result["label"]
        conf  = result["confidence"]
        bar   = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))

        if result["is_threat"] and conf >= threshold:
            print(_c(RED,   f"  THREAT  [{bar}] {conf:.2%}  {label:<20}  {rel}"))
            threats.append(result)
        elif result["is_threat"]:
            print(_c(YELLOW, f"  WARN    [{bar}] {conf:.2%}  {label:<20}  {rel}"))
        else:
            print(_c(GREEN,  f"  CLEAN   [{bar}] {conf:.2%}  {label:<20}  {rel}"))

    print()
    if not threats:
        print(_c(GREEN, "[ASTBertX] All staged files are clean. Commit allowed. ✓\n"))
        return 0

    print(_c(RED, f"[ASTBertX] {len(threats)} threat(s) detected above threshold "
                  f"({threshold:.0%}):\n"))
    for t in threats:
        rel = os.path.relpath(t["file"], str(REPO_ROOT))
        print(_c(RED, f"  • {rel}  →  {t['label']}  ({t['confidence']:.2%})"))
    print()

    if block:
        print(_c(RED + BOLD,
                 "[ASTBertX] Commit BLOCKED. Fix or whitelist the file(s), "
                 "then retry.\n"
                 "  To bypass (not recommended): git commit --no-verify\n"))
        return 1

    print(_c(YELLOW,
             "[ASTBertX] WARNING mode: commit proceeds despite threats.\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
