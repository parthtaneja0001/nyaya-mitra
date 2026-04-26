# Demo video script (target: 90 seconds, hard cap 2 minutes)

Beats sized to a 90-second video. The judging guidance: *"the demo should be engaging and easy to follow for a non-technical audience."*

## Pre-production checklist

- [ ] Phase 1 training has been run on Colab/A100 — `training/dumps/phase1_metrics.jsonl` exists with real numbers
- [ ] `python scripts/render_demo_plots.py --training-jsonl training/dumps/phase1_metrics.jsonl` rendered real-data PNGs into `demo/plots/`
- [ ] `eval/report_phase1.md` has real headline numbers (mean reward / integration solve rate / sensitivity F1)
- [ ] Trained adapter exists at `training/checkpoints/adapter_final_warmup.lora/`
- [ ] Side-by-side transcripts in `demo/transcripts/` show baseline vs trained on `int_001_scripted.md`-shaped cases (one per cohort: welfare / legal / integrated)
- [ ] HF Space deployed and `/healthz` returns 200

## Beats

### 0:00–0:08 — The problem (hook)

> "Six hundred million Indians are eligible for a welfare scheme they don't claim. The bottleneck isn't money — it's a conversation."

Visual: split-screen. Left: a Hindi conversation (subtitles) with a citizen describing a vague problem ("things at home aren't ok"). Right: a tangle of overlapping government scheme websites.

### 0:08–0:18 — Why an LLM is hard here

> "An LLM advisor can list schemes. But hallucinate one and you've sent someone to a non-existent office. Sound 'lawyerly' and you've taken on liability. Optimize for a judge model and you'll learn to be wrong, beautifully."

Visual: zoom in on three failure modes:
1. A fake `dlsa_xyz` contact_id flashing on screen
2. The phrase "in my professional opinion..."
3. A judge model's smiling avatar giving 100/100 to a totally wrong plan

### 0:18–0:35 — Our move: make reward hacking the centerpiece

> "We built `Nyaya Mitra` — an OpenEnv environment where the failure mode is the design. Four hard gates short-circuit the reward to negative one."

Visual: list four gates appearing one by one with a `-1.0` flashing on each:
- `format_violation`
- `hallucination` — any unknown scheme/framework/contact_id
- `contradiction` — citing what the citizen denied
- `sim_leak` — sensitive fact volunteered without matching probe

> "Plus a schema invariant: every legal route must include a real DLSA contact. The agent literally cannot give 'advice' — only routes."

Visual: a Pydantic ValidationError stamping itself on a plan that's missing `free_legal_aid_contact`.

### 0:35–0:52 — Live: the agent in action

> "Watch the trained agent talk to a wary, low-literacy, Hindi-speaking, pregnant garment worker with DV at home."

Visual: full-screen terminal-style transcript scrolling:
- Citizen (in Hindi): "मुझे काम पर परेशानी है, और घर में भी कुछ ठीक नहीं है।"
- Advisor: ASK (work) → citizen reveals pregnancy + denial
- Advisor: PROBE (sensitive_topic=dv) → AT TURN 3, AFTER TRUST BUILDING
- Advisor: EXPLAIN (low literacy)
- Advisor: FINALIZE — `pmsby`+`pmuy`+`apy`+`pmjjby` schemes, `maternity_benefit_act_1961` and `domestic_violence_act_2005` routed to **DLSA Muzaffarpur**

### 0:52–1:10 — The numbers

> "Phase 1 GRPO training. 500 episodes, Qwen 2.5 3B with Unsloth 4-bit and LoRA. Six hours on an A100."

Visual: `total_reward_curve.png` rising from baseline floor to >0.5. Then `integration_solve_rate.png` showing trained vs scripted-baseline bars.

> "Trained mean reward [X.XX] vs scripted baseline 0.51. Integration solve rate [Y%] up from [Z%]. Most importantly: hallucination gate fires twenty times less than at step zero — the agent learned to NOT make up contact IDs."

Visual: `gate_trigger_frequency.png` showing the hallucination line dropping over training.

### 1:10–1:25 — The architecture (technical credibility)

> "Two-track build. Track A — env, KB, citizen sim, fact extractor. Track B — rewards, gates, GRPO trainer. Single legitimate cross-track import is the wire-rewards bootstrap script. AST-walker test forbids the rest."

Visual: simplified architecture diagram with the seam highlighted.

### 1:25–1:30 — Close

> "Nyaya Mitra. Reward hacking is the headline. Hosted at huggingface.co/spaces/parthtaneja0001/nyaya-mitra-env. Code at github.com/parthtaneja0001/nyaya-mitra."

Visual: GitHub URL + HF Space URL with the dignity_judge clamp ribbon ("max 5%") in the corner.

## Recording notes

- **Voice:** clear, slow, no jargon for the first 30 seconds. Earn the term "GRPO" by 0:35.
- **Subtitles:** burn in for the Hindi conversation. Let the visual breathe.
- **Pace:** never spend more than 12 seconds on a static slide; the transcript scroll is the longest single shot at 17 sec.
- **No emojis on screen** (workspace style).
- **End cards:** repo URL + HF Space URL on a single quiet frame, 5+ seconds.

## Companion deliverables

- HF blog post: `demo/blog_post.md` (long-form storytelling)
- Slide deck (optional): same structure as the video, 8 slides
- README rewrite: already done; positioned around the same anti-reward-hacking story
