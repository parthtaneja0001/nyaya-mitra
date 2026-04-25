# What this is not

Nyaya Mitra is an RL **environment** to train welfare/paralegal advisor LLMs. It is deliberately **not** any of the following.

## Not a substitute for a lawyer

Every `LegalRouteRecommendation` the trained agent produces is structurally required to include a `free_legal_aid_contact` pointing to a real NALSA / SLSA / DLSA office. Pydantic refuses to construct a route without one (see `tests/integration/test_action_plan_requires_legal_aid_contact`).

The agent **cannot** give "advice" without a route to a human lawyer. We made this a schema-level invariant rather than a reward signal so it cannot be optimized away during training.

## Not a benefits-application service

The env's `ActionPlan.schemes[].application_path` cites canonical sources (`pmkisan.gov.in`, `pmuy.gov.in`, `pmayg.nic.in`, `pmjay.gov.in`, `nrega.nic.in`, `jansuraksha.gov.in`) and named offline offices (Common Service Center, Gram Panchayat, Block Development Office, LPG distributor, etc.). The agent **routes** the citizen; the citizen still applies through the canonical channel.

## Not a chatbot

The agent's terminal output is a structured `ActionPlan`, not a free-text reply. Any free-text the agent emits during the conversation is via `Ask` / `Probe` (questions, not advice) or `Explain` (a bounded teaching utterance). The reward fn evaluates the structured plan, not chat aesthetics.

## Not exhaustive on coverage

Current KB covers 6 schemes + 4 legal frameworks + 20 DLSAs across 10 states. PLAN.md sets a 30/15/all-states target. Many real-world situations (e.g., agricultural input subsidies, MSP procurement, RTI requests, women-specific health scheme variants, state-specific labour codes) are out of scope for the v1 environment. The architecture supports growth — new entries are JSON files + a single Python checker — but the demo is **not** a comprehensive welfare oracle.

## Not a real citizen sim (yet)

The current `CitizenSimulator` is "smart-canned" — deterministic, profile-driven, templated utterances across en / hi / hinglish, with trust gating on sensitive disclosures. It is **not** a frozen LLM. PLAN.md A.4 calls for a Qwen 2.5 3B Instruct (or Llama 3.2 3B) frozen-llm sim with the explicit hard rules (never volunteer sensitive facts, speak at literacy level, never contradict, never invent). That's the next track-A pickup.

The smart-canned sim is sufficient to exercise the reward pipeline and demonstrate end-to-end training, but it produces a much narrower distribution of utterances than a real LLM would.

## Not a production system

This is hackathon-scale code. Specifically:
- Single-tenant FastAPI server (no per-session isolation)
- No rate limiting, no auth, no audit log
- Profile data is hand-authored synthetic — no real PII anywhere
- HF Space deploy uses a single Docker container; no horizontal scaling

A real deployment would add multi-tenancy, observability, content-safety filters on the citizen sim, human-in-the-loop review of plans, and integration with state-specific welfare directories.

## Not optimized for "looks good"

The reward function deliberately weights LLM-judged components (`dignity_judge`) at the 5% cap and deterministic components at 15% max. An advisor that produces *aesthetically* nice plans will not score well unless those plans are actually grounded, structurally valid, and procedurally correct. The judge's vote is a tiebreaker, not the goal.

## Not done

KB content (24 more schemes, 11 more frameworks), 46 more seed profiles, the real LLM citizen sim, and the actual GRPO training runs are remaining work. See `docs/kb_coverage.md` for the inventory and the open coord-board tasks for the queue.
