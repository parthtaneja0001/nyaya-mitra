#!/usr/bin/env bash
# deploys this repo to a hugging face space (docker sdk).
#
# usage:
#   export HF_TOKEN=hf_xxx                          # write-scope; huggingface.co/settings/tokens
#   export HF_SPACE_REPO=parthtaneja0001/nyaya-mitra-env  # optional; default below
#   ./scripts/deploy_space.sh
#
# what it does:
#   1. logs into hf with the provided token (uses the new `hf` cli; huggingface-cli is deprecated)
#   2. creates the space (docker sdk) if it doesn't exist
#   3. uploads the repo via `hf upload` (single-commit, handles auth + exclusions)
#   4. prints the space url; first docker build takes ~5 min
#
# requires: pip install -e ".[track_a]" (which pulls huggingface_hub)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SPACE_REPO="${HF_SPACE_REPO:-parthtaneja0001/nyaya-mitra-env}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "error: HF_TOKEN env var not set."
    echo "  get a write-scope token at https://huggingface.co/settings/tokens"
    echo "  then: export HF_TOKEN=hf_xxxxx"
    exit 1
fi

# prefer the project venv's hf if available, fall back to global
HF_CLI="hf"
if [ -x "$ROOT/.venv/bin/hf" ]; then
    HF_CLI="$ROOT/.venv/bin/hf"
fi

if ! command -v "$HF_CLI" >/dev/null 2>&1 && [ ! -x "$HF_CLI" ]; then
    echo "error: hf cli not found."
    echo "  install: uv pip install huggingface_hub"
    echo "  (note: huggingface-cli is deprecated; we use the new 'hf' command)"
    exit 1
fi

echo "logging into hugging face..."
"$HF_CLI" auth login --token "$HF_TOKEN" --add-to-git-credential 2>&1 | grep -v "^hf_" || true
"$HF_CLI" auth whoami

echo "ensuring space $SPACE_REPO exists (docker sdk)..."
"$HF_CLI" repos create "$SPACE_REPO" --type space --space-sdk docker --exist-ok 2>&1 | tail -2

# hf spaces require frontmatter in README.md (title, sdk, app_port). prepare a
# deploy-only README that has it; restore on exit so the github repo's README
# stays clean.
ORIG_README="$ROOT/README.md"
ORIG_BACKUP="$ROOT/.README.original.bak"

cleanup() {
    if [ -f "$ORIG_BACKUP" ]; then
        mv "$ORIG_BACKUP" "$ORIG_README"
    fi
}
trap cleanup EXIT

cp "$ORIG_README" "$ORIG_BACKUP"
TMP_README=$(mktemp)
cat > "$TMP_README" <<'YAML'
---
title: nyaya-mitra-env
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 8000
pinned: false
license: apache-2.0
---

YAML
cat "$ORIG_BACKUP" >> "$TMP_README"
mv "$TMP_README" "$ORIG_README"

echo "uploading to $SPACE_REPO..."
"$HF_CLI" upload "$SPACE_REPO" . . \
    --repo-type space \
    --commit-message "deploy: $(git rev-parse --short HEAD)" \
    --exclude ".venv/*" \
    --exclude ".git/*" \
    --exclude "**/__pycache__/*" \
    --exclude "**/*.pyc" \
    --exclude "training/dumps/*" \
    --exclude "training/checkpoints/*" \
    --exclude ".pytest_cache/*" \
    --exclude ".ruff_cache/*" \
    --exclude ".mypy_cache/*" \
    --exclude "wandb/*" \
    --exclude "*.egg-info/*" \
    --exclude "build/*" \
    --exclude "dist/*" \
    --exclude ".DS_Store" \
    --exclude "node_modules/*" \
    --exclude ".README.original.bak" 2>&1 | tail -5

SPACE_HOST=$(echo "$SPACE_REPO" | tr '/' '-')
echo ""
echo "deployed: https://huggingface.co/spaces/$SPACE_REPO"
echo "first build takes ~5 min. once up, smoke test:"
echo "  curl https://${SPACE_HOST}.hf.space/healthz"
