#!/usr/bin/env python
"""Train the crease extractor on synthetic scans.

    python scripts/make_synthetic.py --n 4000 --out ../data/synth
    python scripts/train.py --data ../data/synth --epochs 30

Reports crease detection and mountain/valley accuracy separately, because they
are different problems and the gap between them is the point. MV chance is 0.5;
a model that has latched onto texture rather than shading sits there.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from origamicp.models import Counts, CreaseUNet, SyntheticCreaseDataset, collate, losses


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=32)
    # Depth sets how far the receptive field reaches, which is the quantity the
    # mountain/valley ablation turns on: resolving the sign coherently means
    # comparing creases with each other, and a unit that cannot see two of them
    # at once has nothing to compare.
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--no-light",
        action="store_true",
        help="withhold the light-direction planes (identifiability ablation)",
    )
    # Scores each sheet against the better of the two global mountain/valley
    # assignments. Only meaningful together with --no-light, where the sign is
    # unrecoverable and a per-pixel loss is minimised by declining to choose.
    parser.add_argument(
        "--flip-invariant-mv",
        action="store_true",
        help="mountain/valley loss invariant to inverting every crease at once",
    )
    parser.add_argument("--limit-batches", type=int, default=0, help="smoke-test escape hatch")
    return parser.parse_args(argv)


def run_epoch(model, loader, device, optimiser=None, limit=0, flip_invariant_mv=False):
    training = optimiser is not None
    model.train(training)
    counts = Counts()
    totals = {"total": 0.0, "crease": 0.0, "mv": 0.0}
    batches = 0

    for batch in loader:
        batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        with torch.set_grad_enabled(training):
            outputs = model(batch["image"])
            loss = losses(outputs, batch, flip_invariant_mv=flip_invariant_mv)

        if training:
            optimiser.zero_grad(set_to_none=True)
            loss["total"].backward()
            optimiser.step()

        with torch.no_grad():
            counts.update(outputs, batch)
        for key in totals:
            totals[key] += loss[key].detach().item()
        batches += 1
        if limit and batches >= limit:
            break

    for key in totals:
        totals[key] /= max(batches, 1)
    return totals, counts


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    use_light = not args.no_light
    train_set = SyntheticCreaseDataset(
        args.data, "train", crop=args.crop, jitter=True, use_light=use_light
    )
    # Validation runs on whole sheets, uncropped: a crease that only makes sense
    # in the context of the full pattern should be scored that way.
    val_set = SyntheticCreaseDataset(
        args.data, "val", crop=None, jitter=False, use_light=use_light
    )
    print(f"train {len(train_set)} images | val {len(val_set)} images | device {device}", flush=True)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate
    )

    model = CreaseUNet(in_channels=3, width=args.width, depth=args.depth).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"model: {parameters / 1e6:.2f}M parameters\n")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_totals, train_counts = run_epoch(
            model, train_loader, device, optimiser, args.limit_batches,
            flip_invariant_mv=args.flip_invariant_mv,
        )
        val_totals, val_counts = run_epoch(
            model, val_loader, device, None, args.limit_batches,
            flip_invariant_mv=args.flip_invariant_mv,
        )
        scheduler.step()

        print(
            f"epoch {epoch:>3}/{args.epochs}  "
            f"loss {train_totals['total']:.4f} (crease {train_totals['crease']:.4f} "
            f"mv {train_totals['mv']:.4f})  {time.time() - started:.0f}s"
        )
        print(f"           train  {train_counts.summary()}")
        print(f"           val    {val_counts.summary()}", flush=True)

        if val_counts.f1 > best_f1:
            best_f1 = val_counts.f1
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "width": args.width,
                    "depth": args.depth,
                    "flip_invariant_mv": args.flip_invariant_mv,
                    "use_light": use_light,
                    "val_f1": val_counts.f1,
                    "val_mv_accuracy": val_counts.mv_accuracy,
                },
                args.out / "best.pt",
            )

    print(f"\nbest val F1 {best_f1:.3f} -> {args.out / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
