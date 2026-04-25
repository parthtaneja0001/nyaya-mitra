#!/usr/bin/env bash
# deploys this repo to a hugging face space (docker sdk).
#
# usage:
#   export HF_TOKEN=hf_xxx                      # write-scope token from huggingface.co/settings/tokens
#   export HF_SPACE_REPO=parthtaneja0001/nyaya-mitra-env   # optional; defaults below
#   ./scripts/deploy_space.sh
#
# what it does:
#   1. logs into hf with the provided token
#   2. creates the space (docker sdk) if it doesn't exist
#   3. pushes the current branch to the space's main
#   4. prints the space url; build takes ~2 minutes
set -euo pipefail

SPACE_REPO="${HF_SPACE_REPO:-parthtaneja0001/nyaya-mitra-env}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "error: HF_TOKEN env var not set."
    echo "  get a write-scope token at https://huggingface.co/settings/tokens"
    echo "  then: export HF_TOKEN=hf_xxxxx"
    exit 1
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "error: huggingface-cli not installed."
    echo "  install: pip install huggingface_hub"
    exit 1
fi

echo "logging into hugging face..."
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null

echo "ensuring space $SPACE_REPO exists (docker sdk)..."
huggingface-cli repo create "$SPACE_REPO" --type space --space_sdk docker -y >/dev/null 2>&1 || true

remote_url="https://huggingface.co/spaces/$SPACE_REPO"
if git remote get-url hf-space >/dev/null 2>&1; then
    git remote set-url hf-space "$remote_url"
else
    git remote add hf-space "$remote_url"
fi

current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "pushing $current_branch -> $SPACE_REPO main..."
git push hf-space "$current_branch:main" --force-with-lease

echo ""
echo "deployed: $remote_url"
echo "wait ~2 minutes for the docker build to finish."
echo "once up, smoke test: curl $remote_url/healthz"
