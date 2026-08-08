# OrigamiCP

Recovering an origami crease pattern from an image of the unfolded sheet: where
every fold line is, and which way each one folds.

The mountain/valley half of that question turns out to be limited by
identifiability rather than by difficulty. Under a Lambertian shading model a
mountain lit from one direction and a valley lit from the opposite direction
produce the same image, and neither Kawasaki's nor Maekawa's condition can break
the tie, since both survive inverting every crease at once. The assignment is
fixed only up to a global flip -- one bit for an entire sheet -- and a per-pixel
cross-entropy is therefore minimised by refusing to decide. Giving a network more
context makes its internal coherence *worse*; changing the objective to one
invariant to the symmetry fixes it.

**The full write-up, with every number and how it was measured, is in
[code/README.md](code/README.md).**

## What is here

```text
code/       the package, the experiments, and the write-up
  origamicp/  crease patterns, fold conditions, renderer, extractor, decoder
  scripts/    dataset generation, training, tuning, one script per experiment
  tests/      173 tests
data/       the six pilot designs, printable capture templates, manifest example
results/    every figure and every console output the write-up cites
```

Generated datasets and training checkpoints are not tracked: the datasets are
reproduced exactly from their seed by `scripts/make_synthetic.py`, and the
checkpoints are large. Everything else needed to rerun the study is here.

## Quick start

```bash
cd code
pip install -e .
python -m pytest -q
```

Reproducing the reported numbers, including which flags matter and why, is
documented in [code/README.md](code/README.md#reproducing).

## Status

All results are synthetic. The capture tooling for a real dataset is
implemented and tested end to end on synthetic scans, but no physical sheets
have been collected; that is the largest open gap and it is discussed in the
write-up's limitations.

## Licence

MIT, see [LICENSE](LICENSE).
