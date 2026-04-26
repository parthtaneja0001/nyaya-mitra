"""FastAPI server for the Nyaya Mitra OpenEnv environment.

Delegates entirely to openenv.core's `create_app` builder so we get the
canonical OpenEnv HTTP surface:

  POST /reset      → start an episode
  POST /step       → submit a NyayaAction, receive a NyayaObservation
  GET  /state      → public state (episode_id, step_count, elicited_facts)
  GET  /metadata   → EnvironmentMetadata (name, description, version)
  GET  /schema/*   → JSON schemas for action/observation/state
  GET  /healthz    → liveness probe (added below)
  GET  /docs       → Swagger UI
  GET  /redoc      → ReDoc UI

This is the entrypoint declared in openenv.yaml (`server.app:app` once you
package the env, or `nyaya_mitra.env.server:app` for in-repo dev).
"""

from __future__ import annotations

from openenv.core.env_server.http_server import create_app

from nyaya_mitra.env.openenv_env import (
    NyayaAction,
    NyayaEnvironment,
    NyayaObservation,
)


def _env_factory() -> NyayaEnvironment:
    return NyayaEnvironment()


app = create_app(
    env=_env_factory,
    action_cls=NyayaAction,
    observation_cls=NyayaObservation,
    env_name="nyaya-mitra",
)


@app.get("/healthz", tags=["Health"])
def healthz() -> dict[str, str]:
    return {"status": "ok"}
