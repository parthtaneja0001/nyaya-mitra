"""concatenate two training metrics jsonls into one combined file with
monotonic step numbers. used after a resumed-training run so the demo curve
spans both phases as a single trajectory.

usage:
    python scripts/merge_metrics.py \
        training/dumps/phase1_t4_metrics.jsonl \
        training/dumps/phase1_t4_metrics_resumed.jsonl \
        training/dumps/phase1_t4_metrics_combined.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    a_path = Path(sys.argv[1])
    b_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])

    rows: list[dict] = []
    for line in a_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    n_a = len(rows)
    print(f"loaded {n_a} rows from {a_path}")

    n_b = 0
    for line in b_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        # renumber so step is monotonic across both files.
        d["step"] = d.get("step", 0) + n_a
        rows.append(d)
        n_b += 1
    print(f"loaded {n_b} rows from {b_path} (renumbered to step {n_a}..{n_a + n_b - 1})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
