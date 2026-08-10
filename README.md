# OrigamiCP

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21848888.svg)](https://doi.org/10.5281/zenodo.21848888)

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

## Citing

Archived on Zenodo; the badge above resolves to the latest version. To cite the
exact revision behind the reported numbers, use the version DOI
[10.5281/zenodo.21848889](https://doi.org/10.5281/zenodo.21848889).
Machine-readable metadata is in [CITATION.cff](CITATION.cff).

## Related work

[origami-recognition-system](https://github.com/nihatgaribli/origami-recognition-system)
is a separate project of mine on the other side of the problem: it collects and
labels photographs of *folded* models, toward recognising which model a photo
shows. This repository goes the other way, from an image of an *unfolded* sheet
back to its crease pattern. The two share a domain, not a dataset.

## Licence

MIT, see [LICENSE](LICENSE).
