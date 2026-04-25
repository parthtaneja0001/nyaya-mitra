"""fastapi wrapper for the nyaya mitra env. single-session for now; scale to per-session later."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, TypeAdapter, ValidationError

from nyaya_mitra.citizen.extractor import FactExtractor
from nyaya_mitra.citizen.simulator import CitizenSimulator
from nyaya_mitra.env.environment import NyayaMitraEnv
from nyaya_mitra.interface import AdvisorAction
from nyaya_mitra.knowledge.loader import KnowledgeBase

_env: NyayaMitraEnv | None = None
_action_adapter: TypeAdapter[AdvisorAction] = TypeAdapter(AdvisorAction)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _env
    _env = NyayaMitraEnv(KnowledgeBase(), CitizenSimulator(), FactExtractor())
    yield
    if _env is not None:
        _env.close()


app = FastAPI(title="nyaya-mitra", lifespan=lifespan)


class ResetRequest(BaseModel):
    seed: int = 0
    difficulty: str | None = None


@app.get("/")
def root() -> dict[str, Any]:
    """basic landing for hf spaces / smoke checks. lists endpoints."""
    return {
        "name": "nyaya-mitra",
        "description": "paralegal-cum-welfare-advisor RL environment",
        "endpoints": {
            "POST /reset": "start a new episode; body {seed, difficulty}",
            "POST /step": "submit an AdvisorAction; body matches interface.AdvisorAction",
            "GET /state": "debug snapshot; requires NYAYA_DEBUG=1",
            "POST /close": "release the current episode",
            "GET /healthz": "liveness probe",
        },
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest) -> dict[str, Any]:
    assert _env is not None
    obs = _env.reset(seed=req.seed, difficulty=req.difficulty)
    return obs.model_dump()


@app.post("/step")
def step(action: dict[str, Any]) -> dict[str, Any]:
    assert _env is not None
    try:
        parsed = _action_adapter.validate_python(action)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors()) from e
    result = _env.step(parsed)
    return {
        "observation": result.observation.model_dump() if result.observation else None,
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
def state() -> dict[str, Any]:
    assert _env is not None
    if not os.environ.get("NYAYA_DEBUG"):
        raise HTTPException(status_code=403, detail="state requires NYAYA_DEBUG=1")
    return _env.state()


@app.post("/close")
def close() -> dict[str, str]:
    assert _env is not None
    _env.close()
    return {"status": "closed"}
