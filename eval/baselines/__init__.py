"""baseline advisors used in the eval harness as comparison floors.

three baselines:

- scripted_baseline: deterministic, no LLM, no GPU. uses the env's elicited_facts
  directly to produce plans. acts as a sanity floor that proves the pipeline works
  end-to-end before any LLM is involved.

- vanilla_baseline: LLM with minimal system prompt. honest "what does the base
  model do without any context?" lower bound.

- prompted_baseline: same LLM with hand-tuned system prompt + KB excerpts in
  context. the *honest* comparison — what a non-RL approach can extract from
  the same model.

all baselines implement the rollout.Advisor signature so they're swappable.
"""

from eval.baselines.prompted_baseline import build_prompted_baseline
from eval.baselines.scripted_baseline import build_scripted_baseline
from eval.baselines.vanilla_baseline import build_vanilla_baseline

__all__ = [
    "build_prompted_baseline",
    "build_scripted_baseline",
    "build_vanilla_baseline",
]
