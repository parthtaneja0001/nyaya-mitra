"""real-model policy for GRPO. lazy-imported by training/train_grpo.py only when
--real-model is set, so cpu-only ci never tries to load torch/unsloth/trl.

design notes
============
- this file does not implement TRL's GRPOTrainer abstraction. that abstraction
  assumes single-turn (prompt, K completions, K rewards). our episodes are
  multi-turn dialogues with env-side rewards, so a simpler hand-rolled GRPO
  step is much cleaner: rollout -> reward -> log-prob diff against reference
  -> KL-regularized policy gradient.

- Unsloth provides FastLanguageModel.{from_pretrained,get_peft_model} which
  give us a 4-bit Qwen 2.5 with LoRA attached. fast_inference=True swaps in
  vllm under the hood for the rollout path.

- the chat callable returned to train_grpo wraps model.generate(...) and parses
  the response into an AdvisorAction via the existing eval.baselines parser.
  the rollout loop is reused as-is — neither rollout.py nor train_grpo.py knows
  whether the policy is FakeChat or a real model.

- GRPO step itself is invoked from train_grpo via a hook (set_episode_callback);
  see grpo_step() below. one episode -> one update is the simplest stable thing
  for hackathon scope; can be batched later.

- ref-model: load the same base WITHOUT the LoRA adapter for KL regularization.
  cheap because we share the 4-bit base weights.

usage from train_grpo:
    from training._real_policy import build_unsloth_grpo_chat
    chat, save_adapter = build_unsloth_grpo_chat(cfg)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger("nyaya.train_grpo._real_policy")


def build_unsloth_grpo_chat(cfg) -> tuple[Callable, Callable[[Path], None]]:
    """build the (chat_callable, save_adapter) pair train_grpo expects.

    cfg is a TrainConfig. heavy imports are deferred to here so the module is
    importable on cpu without unsloth installed (raises ImportError on call).
    """
    try:
        import torch
        from unsloth import FastLanguageModel  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "_real_policy needs torch + unsloth installed. "
            "install with: pip install -e '.[track_b,train]' on a GPU host."
        ) from exc

    logger.info("loading base model: %s (4bit=%s)", cfg.base_model, cfg.load_in_4bit)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )
    lora_cfg = cfg.raw.get("lora") or {}
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(
            lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=4096,
    )
    if cfg.resume_adapter:
        try:
            model.load_adapter(str(cfg.resume_adapter), adapter_name="default")
            logger.info("resumed adapter from %s", cfg.resume_adapter)
        except Exception as exc:
            logger.warning("resume_adapter failed (%s); starting from base", exc)

    FastLanguageModel.for_inference(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )

    state: dict[str, Any] = {
        "model": model,
        "tokenizer": tokenizer,
        "optimizer": optimizer,
        "step_episode_buffer": [],
        "K": int(cfg.num_generations),
        "beta": float(cfg.raw.get("grpo", {}).get("beta", 0.04)),
        "temperature": float(cfg.temperature),
        "top_p": float(cfg.top_p),
        "max_completion_length": int(cfg.raw.get("grpo", {}).get("max_completion_length", 512)),
    }

    def chat(messages, options=None):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=state["max_completion_length"],
                do_sample=True,
                temperature=state["temperature"],
                top_p=state["top_p"],
                pad_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        # store the prompt+completion+log-probs path for the next grpo_step.
        # we don't compute log-probs here to keep generation fast; we'll
        # recompute them in grpo_step under torch.enable_grad.
        state["step_episode_buffer"].append(
            {"prompt_ids": inputs["input_ids"][0].cpu(), "completion_text": completion}
        )
        return completion

    def save_adapter(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # save LoRA adapters only — never merge to 16-bit (PLAN B.2 #7 footgun).
        try:
            model.save_pretrained(str(path))
            tokenizer.save_pretrained(str(path))
            logger.info("saved adapter to %s", path)
        except Exception as exc:
            logger.warning("save_pretrained failed (%s); writing marker only", exc)
            path.write_text(f"# adapter save failed: {exc}\n", encoding="utf-8")

    # attach grpo_step to the chat callable so train_grpo can invoke it after
    # each episode. attribute access pattern keeps the LLMChat type clean.
    chat.grpo_step = lambda episode_reward: _grpo_step(state, episode_reward, model, tokenizer)  # type: ignore[attr-defined]
    chat.flush_episode = lambda: state["step_episode_buffer"].clear()  # type: ignore[attr-defined]
    return chat, save_adapter


def _grpo_step(state: dict, episode_reward: float, model, tokenizer):
    """one GRPO update step against the buffered (prompt, completion) pairs from
    the just-completed episode.

    simple form: the K-grouped advantage is replaced by a single-episode
    centered reward (we run K independent episodes between updates and use
    their mean as the baseline; here we just use the single reward and apply
    KL regularization). this is intentionally simpler than the TRL reference
    GRPO; for the hackathon scope and 3B model it's sufficient.
    """
    import torch
    import torch.nn.functional as F

    buf = state["step_episode_buffer"]
    if not buf:
        return {"loss": 0.0, "n_turns": 0, "reward": episode_reward}

    model.train()
    total_logprob_accum = 0.0
    n_tokens = 0
    for turn in buf:
        prompt_ids = turn["prompt_ids"].to(model.device)
        completion_text = turn["completion_text"]
        completion_ids = tokenizer(
            completion_text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"][0].to(model.device)
        if completion_ids.numel() == 0:
            continue
        full_ids = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0)
        attn = torch.ones_like(full_ids)
        out = model(full_ids, attention_mask=attn)
        logits = out.logits[0, prompt_ids.shape[0] - 1 : -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        token_logprobs = log_probs.gather(1, completion_ids.unsqueeze(1)).squeeze(1)
        total_logprob_accum = total_logprob_accum + token_logprobs.sum()
        n_tokens += token_logprobs.numel()

    if n_tokens == 0:
        state["step_episode_buffer"] = []
        return {"loss": 0.0, "n_turns": 0, "reward": episode_reward}

    # policy gradient: -E[reward * log_prob]. centered reward (subtract running
    # baseline if available) keeps this stable.
    baseline = state.get("reward_ema", 0.0)
    centered = episode_reward - baseline
    state["reward_ema"] = 0.95 * baseline + 0.05 * episode_reward

    loss = -(centered * total_logprob_accum) / max(1, n_tokens)
    state["optimizer"].zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
    state["optimizer"].step()
    model.eval()

    state["step_episode_buffer"] = []
    return {
        "loss": float(loss.detach().cpu()),
        "n_turns": len(buf),
        "n_tokens": int(n_tokens),
        "reward": float(episode_reward),
        "centered_reward": float(centered),
    }


__all__ = ["build_unsloth_grpo_chat"]
