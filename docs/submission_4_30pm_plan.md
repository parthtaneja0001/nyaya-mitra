# 4:30 PM Submission plan (5h43m crunch)

You have 10:47 AM → 4:30 PM. This is the path that fits.

## Strategy (TL;DR)

- Use **free Colab T4** for training. Don't pay for a faster GPU — setup overhead exceeds the time saved.
- Reserve **$60 HF credits** as buffer for Space re-deploys (each costs <$1) and post-hackathon experiments.
- Trade model size + episode count for wall-clock: **Qwen 2.5 0.5B + 100 episodes ≈ 1h on T4.**
- A real upward curve is enough for the 20% "showing improvement" judging axis. Already strong on the other 80%.

## Why not use the $30 HF credit for faster training

| Option | Cost | Setup time | Training time | Net |
|---|---|---|---|---|
| Colab T4 (free) | $0 | 5min | ~1h | **best** |
| HF Space + a10g paid runtime | $1-3 | 30min | ~30min | break-even at best |
| HF Inference Endpoint | $5+ | 1h+ | unsuitable for training | net loss |
| Colab Pro+ A100 (you'd pay $50/mo) | $50 sub | 5min | ~30min | only worth it if you already subscribe |

The HF credits don't accelerate Colab. They DO power the live Space. Keep them as buffer.

## Step-by-step

### 10:50 — Open Colab + auth (5 min)

1. https://colab.research.google.com → File → Open notebook → GitHub → paste `https://github.com/parthtaneja0001/nyaya-mitra` → pick `training/train_grpo_colab.ipynb`.
2. Runtime → Change runtime type → **T4 GPU**. Save.
3. Cell 7 already uses `getpass()` — **don't edit the code**, just run the cell. A masked prompt appears; paste your **NEW** HF token there (after rotating the leaked one).
4. (Optional) skip the W&B prompt by hitting Enter.
5. Run cells 1-7 sequentially.
6. **Skip cell 11** (smoke run). Local FakeChat smoke covered orchestration.
7. **Replace cell 15 contents** with one line:
   ```
   !bash scripts/run_timeboxed.sh
   ```

### 11:00-12:00 — Training runs unattended (~1h)

100 episodes × ~30-40s each on T4 = ~1h. The script auto-runs `scripts/render_demo_plots.py` at the end so the 6 PNGs get regenerated with real metrics.

If reward curve looks flat at episode 100: don't extend, ship what you have. Document honestly in the blog.

### 12:00-12:30 — Push artifacts (30 min)

In Colab cell 25 (or new cell):
```
!cd /content/nyaya-mitra && \
  git config user.email 'parth.sankhla98@gmail.com' && \
  git config user.name 'parthtaneja0001' && \
  git add training/dumps/ training/checkpoints/ demo/plots/ eval/report_*.md && \
  git commit -m 'phase 1 timeboxed artifacts' && \
  git push https://parthtaneja0001:$HF_TOKEN@github.com/parthtaneja0001/nyaya-mitra HEAD:main
```

Note: GitHub does NOT accept HF tokens as auth. Use a GitHub PAT instead, OR push from the Mac after pulling Colab artifacts via `gcloud storage` / Colab's local download. Easiest path: in Colab, run `!zip -r /tmp/artifacts.zip training/dumps training/checkpoints demo/plots eval/report_*.md`, then download via the Files panel, then commit + push from your Mac.

### 12:30-13:00 — Re-deploy HF Space (30 min)

Back on your **Mac**:
```
cd ~/meta-hackathon
git pull origin main                              # pull the artifacts you pushed
export HF_TOKEN=hf_xxx_NEW_ROTATED                # the rotated token, not the leaked one
./scripts/deploy_space.sh
```

Wait ~3 min for Space rebuild. Verify:
```
curl https://parthtaneja0001-nyaya-mitra-env.hf.space/healthz
```

### 13:00-13:30 — Update README + blog with real numbers (30 min)

`eval/report_phase1_timeboxed.md` will have your headline numbers. Edit the `[fill in real numbers...]` placeholders in:
- `README.md` (the "## The numbers (training)" section)
- `demo/blog_post.md` (same section)

Push final commit.

### 13:30-14:30 — Record 90-sec video (1h)

Use `demo/video_script.md` as the storyboard. **One take if possible.** Tools: macOS QuickTime (Cmd+Shift+5 → Record Selected Portion) or Loom for screen + face. Upload to YouTube as **unlisted**.

### 14:30-15:30 — Video edits + buffer (1h)

Trim start/end. Add captions if Hindi audio is unclear. Re-upload if you re-record.

### 15:30-16:00 — Final commits + blog publish (30 min)

Push final README. To publish the blog to HF Hub (optional):
```
.venv/bin/hf upload parthtaneja0001/nyaya-mitra-blog demo/blog_post.md ./README.md --repo-type=dataset
```

Otherwise, link `demo/blog_post.md` directly from the GitHub repo's main page.

### 16:00-16:30 — Submit to portal (30 min)

Submit links:
- Repo: https://github.com/parthtaneja0001/nyaya-mitra
- HF Space: https://huggingface.co/spaces/parthtaneja0001/nyaya-mitra-env
- Video: YouTube unlisted URL
- Blog: HF Hub URL or GitHub blog markdown link

## Fallbacks

**If training crashes mid-run:** restart `!bash scripts/run_timeboxed.sh`. Will resume from latest LoRA snapshot if one exists in `training/checkpoints/`.

**If T4 unavailable:** Colab sometimes runs out of T4 quotas. Try the second HF account / different Google account / different IP. Or use Pro for $10 to guarantee.

**If reward curve is totally flat:** still a real run. Use existing FakeChat-smoke plots and mention in blog: "Phase 1 timeboxed proves the pipeline; full Phase 1 is post-hackathon work." Score 5-10/20 on "showing improvement" but stay strong on the other 80%.

**If video runs long (>2 min):** cut the architecture section (1:10-1:25 in script). Final 5-sec end-card stays.

## Note on the Colab token cell

Cell 7 looks like:
```python
import os
from getpass import getpass

os.environ["HF_TOKEN"] = getpass("HF token: ")
wb = getpass("W&B API key (optional, blank to skip): ").strip()
if wb:
    os.environ["WANDB_API_KEY"] = wb
!huggingface-cli login --token $HF_TOKEN
```

You **don't edit the code**. When you Run the cell:
1. A password-style input box appears at the bottom of Colab labeled `HF token:`.
2. Paste your token there. Press Enter. The token is set in `os.environ` but NEVER printed or saved to the notebook.
3. Same for the optional W&B key.

The `!huggingface-cli login` line is using the deprecated CLI. If it warns or errors, replace with:
```
!hf auth login --token $HF_TOKEN --add-to-git-credential
```
(I'm pushing a small Colab patch for this too.)
