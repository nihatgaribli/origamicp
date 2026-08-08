#!/usr/bin/env python
"""Regenerate every number and figure the write-up depends on.

    python scripts/run_all.py --data ../data/synth768 --out ../results

Runs the stages in dependency order and writes each stage's console output next
to its figures, so a claim in the text can be traced to the run that produced
it. Training is not included: it takes far longer than everything else and its
checkpoints are inputs here, not outputs. Pass ``--checkpoint`` and
``--no-light-checkpoint`` from a completed training run.

Stages that need no checkpoint (the oracle and resolution studies) still run if
the checkpoints are missing, so a fresh clone can reproduce the parts that do
not depend on a trained model.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--no-light-checkpoint", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    # Selected on validation for a 768-pixel render, which is what --data points
    # at. It is passed here rather than left to the library default because the
    # two disagree on purpose: 30 is right at 512 and 35 at 768, and no rule
    # connecting them has been measured at enough sizes to be worth writing
    # down. Point this at a dataset rendered at another size and this is the
    # flag to re-sweep.
    parser.add_argument("--snap", type=float, default=35.0)
    # Extra no-light checkpoints for the coherence table. The claim they support
    # -- that context and capacity make coherence worse, and that a
    # flip-invariant loss fixes it -- needs several models, so it cannot be a
    # stage that runs off one checkpoint like the others.
    parser.add_argument("--coherence-checkpoints", type=Path, nargs="*", default=[])
    return parser.parse_args(argv)


def run(name: str, command: list[str], out: Path) -> bool:
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}", flush=True)
    result = subprocess.run(
        [sys.executable, *command], capture_output=True, text=True, cwd=Path(__file__).parent.parent
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, end="")
    (out / f"{name}.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
    return result.returncode == 0


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    data = str(args.data)
    limit = ["--limit", str(args.limit)] if args.limit else []
    snap = ["--snap", str(args.snap)]

    stages: list[tuple[str, list[str], bool]] = [
        (
            # Twenty-four tuning designs is not enough for a thirty-point grid:
            # at that size the sweep picked a configuration that lost to the
            # library default on held-out sheets at 1024. Sixty holds up.
            "resolution",
            ["scripts/resolution.py", "--n-tune", "60", "--out", str(args.out / "resolution.png")],
            True,
        ),
        (
            "oracle_ceiling",
            ["scripts/evaluate_graph.py", "--data", data, "--split", "test", "--oracle",
             *snap, *limit],
            True,
        ),
        (
            # Swept on validation by ``scripts/tune.py --classical``, over the
            # same grid the model got. Running it on the model's own settings
            # scored 0.087, which is a handicap the baseline did not choose: its
            # masks are noisier, so a density filter set against clean ones cuts
            # into real creases. On its own settings it reaches 0.260.
            #
            # Selected on crease F1 rather than on validity, against the rule
            # used everywhere else, because for this baseline the validity-
            # selected configuration is degenerate -- it returns 272 of 1920
            # interior vertices and buys its 0.229 by declining to answer.
            "classical_baseline",
            ["scripts/evaluate_graph.py", "--data", data, "--split", "test", "--classical",
             "--threshold", "0.80", "--min-density", "1.0", *snap, *limit],
            True,
        ),
    ]

    if args.checkpoint:
        checkpoint = str(args.checkpoint)
        stages += [
            ("model_pipeline",
             ["scripts/evaluate_graph.py", "--data", data, "--split", "test",
              "--checkpoint", checkpoint, *snap, *limit], False),
            ("difficulty",
             ["scripts/difficulty.py", "--data", data, "--checkpoint", checkpoint,
              "--out", str(args.out / "difficulty.png")], False),
            ("predictions",
             ["scripts/predict.py", "--data", data, "--checkpoint", checkpoint,
              "--threshold", "0.92", "--n", "3",
              "--out", str(args.out / "predictions.png")], False),
            ("constrained_decoding",
             ["scripts/constrained_decode.py", "--data", data, "--checkpoint", checkpoint,
              *snap, *limit], False),
        ]
    if args.checkpoint and args.no_light_checkpoint:
        stages.append(
            ("light_ablation",
             ["scripts/light_ablation.py", "--data", data,
              "--with-light", str(args.checkpoint),
              "--without-light", str(args.no_light_checkpoint),
              "--out", str(args.out / "light_ablation.png")], False)
        )

    if args.coherence_checkpoints:
        stages.append(
            ("coherence",
             ["scripts/coherence.py", "--data", data, "--split", "test",
              "--out", str(args.out / "coherence.png"),
              *[str(c) for c in args.coherence_checkpoints]], False)
        )

    failed = [name for name, command, _ in stages if not run(name, command, args.out)]

    print(f"\n{'=' * 72}")
    print(f"{len(stages) - len(failed)}/{len(stages)} stages completed -> {args.out}")
    if failed:
        print(f"failed: {', '.join(failed)}")
    skipped = []
    if not args.checkpoint:
        skipped.append("model stages (pass --checkpoint)")
    if not args.no_light_checkpoint:
        skipped.append("light ablation (pass --no-light-checkpoint)")
    if skipped:
        print(f"skipped: {'; '.join(skipped)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
