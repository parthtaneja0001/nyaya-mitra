# Deploy to Hugging Face Spaces

The env runs as a Docker container on a Hugging Face Space. **The Space URL is the entire submission artifact** — judges pull the env from there to evaluate it.

## One-time setup

1. Create a write-scope token at https://huggingface.co/settings/tokens.
2. Install `huggingface_hub`:
   ```
   pip install huggingface_hub
   ```
3. Set env vars (or add to `~/.zshrc`):
   ```
   export HF_TOKEN=hf_xxxxxxxxxxxxx
   export HF_SPACE_REPO=parthtaneja0001/nyaya-mitra-env
   ```

## Deploying

From the repo root, on the branch you want to deploy:
```
./scripts/deploy_space.sh
```

The script:
- Logs into HF with the token
- Creates the Space (Docker SDK) if it doesn't exist
- Pushes the current branch to the Space's `main`
- Prints the Space URL

The Space takes ~2 minutes to build the Docker image. Once up, smoke-test the canonical OpenEnv routes:

```
SPACE=https://huggingface.co/spaces/$HF_SPACE_REPO

curl $SPACE/healthz                          # → {"status":"ok"}
curl $SPACE/health                           # → {"status":"healthy"}  (canonical OpenEnv)
curl $SPACE/metadata                         # → EnvironmentMetadata JSON
curl $SPACE/schema                           # → JSON schemas for action/observation/state
curl -L $SPACE/docs                          # → Swagger UI HTML
```

Each request creates a fresh env on the HTTP transport (per OpenEnv's stateless-HTTP design). For multi-turn dialogue use the `/ws` WebSocket endpoint or `/mcp` MCP transport.

## Hitting the deployed env from training

```
import os
from nyaya_mitra.env.client import NyayaMitraClient

os.environ["NYAYA_ENV_URL"] = "https://huggingface.co/spaces/parthtaneja0001/nyaya-mitra-env"
client = NyayaMitraClient()
obs = client.reset(seed=0)
```

Or set `NYAYA_ENV_URL` once and the default-arg picks it up.

## Local docker test

To verify the Dockerfile builds and serves before pushing to a Space:
```
docker build -t nyaya-mitra-env:local .
docker run --rm -p 8000:8000 nyaya-mitra-env:local
# in another terminal:
curl http://localhost:8000/healthz
```

## Layout the Space sees

The Space's git repo is the same as this repo's contents pushed to its `main` branch. Hugging Face reads:
- `Dockerfile` — image build
- `README.md` (with frontmatter) — Space metadata
- `openenv.yaml` — OpenEnv manifest

The Space-level README frontmatter expected by HF:
```yaml
---
title: Nyaya Mitra Env
sdk: docker
app_port: 8000
pinned: false
---
```

This is added by `scripts/deploy_space.sh` (or done manually in the Space's UI).

## Troubleshooting

- **Build fails on `uv pip install`**: usually a missing system lib. Check the Space build logs; add the apt package to the Dockerfile's first stage.
- **Space crashes immediately**: check `docker logs` equivalent in the Space's "Logs" tab. Most common: missing dependency in `[env]` group.
- **Healthcheck flapping**: the healthcheck pings `/healthz` every 30s with a 3s timeout. If startup is slow, adjust `--start-period` in the Dockerfile.
