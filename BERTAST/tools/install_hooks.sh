#!/usr/bin/env bash
# ASTBertX — install pre-commit hook and default config
# Usage: bash BERTAST/tools/install_hooks.sh [BERTAST_BASE_PATH]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
BERTAST_BASE="${1:-$SCRIPT_DIR/..}"
BERTAST_BASE="$(cd "$BERTAST_BASE" && pwd)"

echo "[ASTBertX] Repository root : $REPO_ROOT"
echo "[ASTBertX] BERTAST base    : $BERTAST_BASE"

# ── 1. Write pre-commit hook ──────────────────────────────────────────────────

HOOK_PATH="$REPO_ROOT/.git/hooks/pre-commit"

cat > "$HOOK_PATH" <<HOOK
#!/usr/bin/env bash
# ASTBertX pre-commit hook (auto-generated — do not edit manually)
set -euo pipefail
PYTHONPATH="$BERTAST_BASE" python "$BERTAST_BASE/tools/precommit_hook.py" "\$@"
HOOK

chmod +x "$HOOK_PATH"
echo "[ASTBertX] ✓ Pre-commit hook installed → $HOOK_PATH"

# ── 2. Write default .astbertx.yml (only if absent) ──────────────────────────

CONFIG_PATH="$REPO_ROOT/.astbertx.yml"

if [ ! -f "$CONFIG_PATH" ]; then
cat > "$CONFIG_PATH" <<YAML
# ASTBertX configuration
# See BERTAST/tools/precommit_hook.py for all options.

precommit:
  base_path: "$BERTAST_BASE"
  confidence_threshold: 0.75   # block commit if P(threat) >= this value
  block_on_threat: true         # set false to emit warning only
  skip_patterns:
    - "tests/**"
    - "test_*.py"
    - "**/*.min.js"
    - "docs/**"

ci:
  mode: full                    # fast | full
  threshold: 0.70

periodic:
  mode: full
  threshold: 0.65
  include_third_party: false
  out_dir: astbertx_scan
YAML
  echo "[ASTBertX] ✓ Config written → $CONFIG_PATH"
else
  echo "[ASTBertX] ⚠ Config already exists, not overwritten → $CONFIG_PATH"
fi

echo ""
echo "[ASTBertX] Installation complete."
echo "  Pre-commit hook : $HOOK_PATH"
echo "  Config file     : $CONFIG_PATH"
echo ""
echo "  Next steps:"
echo "  1. Edit $CONFIG_PATH to set base_path and thresholds."
echo "  2. Train ASTBertX models (see README)."
echo "  3. git commit — the hook will fire automatically."
