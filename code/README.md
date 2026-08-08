# OrigamiCP

Recovering an origami crease pattern from an image of the unfolded sheet: where
every fold line is, and which way each one folds.

A crease pattern is what is left in the paper after a model is unfolded. Reading
one back out of a photograph is a perception problem with an unusual property --
the answer has to obey known geometry. Every interior vertex must satisfy
Kawasaki's condition on its sector angles and Maekawa's on its fold directions,
so a recovered pattern can be checked for correctness without comparing it to
anything. This repository is built around that: a physically-based renderer, an
extractor, and an evaluation that asks whether the recovered graph could
actually be folded.

All results here are from synthetic data. The capture tooling for a real dataset
is included and tested, but no physical sheets were collected -- see
[Limitations](#limitations).

## Results

216 held-out sheets, 900 designs, rendered at 768 px (about 112 dpi of paper).
Splits are by design, never by image. Every number below comes from
`scripts/run_all.py`; the console output behind each one is in `results/`.

| | pixel F1 | crease F1 | crease MV | geometric validity |
|---|---|---|---|---|
| learning-free baseline | -- | 0.260 | 0.907 | 0.149 |
| U-Net | **0.851** | **0.653** | **0.970** | **0.314** |
| oracle (ground-truth masks) | -- | 0.708 | 0.978 | 0.518 |

Geometric validity is the fraction of interior vertices satisfying Kawasaki,
Maekawa and big-little-big at a 3-degree tolerance. Its components for the
U-Net: Kawasaki 0.359, Maekawa 0.513, big-little-big 0.788.

Both the baseline and the network were tuned on validation over the same grid,
by `scripts/tune.py`. That matters more than it sounds: run on the network's
settings the baseline scores 0.087, and a comparison quoted from that would be
measuring a handicap it did not choose.

## What was found

**Without the light direction the task is not solvable, and the obvious
explanation for how models fail at it is wrong.**

Under a Lambertian shading model with a symmetric crease profile, a mountain lit
from azimuth θ and a valley lit from θ+π produce the same image. The fold
theorems cannot break the tie either: inverting every crease at once maps
Maekawa's defect M−V to −(M−V), so |M−V| = 2 is preserved, and Kawasaki depends
on angles alone. So the assignment is fixed only up to inverting the whole
sheet, one bit of uncertainty for the entire pattern, and every *per-crease*
posterior is uniform. Per-pixel cross-entropy therefore has its Bayes floor at
ln 2. Trained without the light-direction planes, the network converges to
exactly that.

The interesting part is what a model does with the one bit it cannot recover.
Allowing the best global flip per sheet, it reaches 0.738 -- not the 1.0 the
ambiguity alone would permit. But 0.738 is not as close to coherent as it looks,
because a model flipping every crease independently already scores 0.623 on that
metric: with n creases the better of the two signs wins by a binomial
fluctuation. Measured on the actual crease-count distribution, that floor and
the ceiling bracket the model at 30% of the way across.

The column that matters practically is what one bit of side information buys.
The theory says a single bit resolves the whole sheet, so being told the true
label of one crease -- the longest, scored by majority over its pixels -- should
recover everything. Whether it does is a question about the model, not the
theory:

| | plain | up to a flip | one crease given | coherent sheets |
|---|---|---|---|---|
| creases flipped independently | 0.500 | 0.623 | -- | -- |
| ablated U-Net | 0.536 | 0.738 | 0.678 | 16.4% |
| trained on whole sheets | 0.511 | 0.647 | 0.582 | 1.9% |
| depth 6 | 0.523 | 0.698 | 0.643 | 7.9% |
| flip-invariant loss | 0.538 | **0.972** | **0.959** | **92.6%** |

Knowing one fold takes the standard model to 0.678 and the flip-invariant one to
0.959. The bit is worth what the rest of the sheet agrees with it about.

The natural explanation -- a convolution with a bounded receptive field has no
way to compare distant creases -- is testable, and it is false. Training on
whole 768-pixel sheets instead of 384-pixel crops *lowers* up-to-flip accuracy
to 0.643; a depth-6 network with sixteen times the parameters lowers it to
0.695. The fraction of sheets that come back all-right or all-wrong falls from
15.0% to 0.5% as context grows.

The ordering is monotone and it runs the wrong way, which is the finding. A
per-pixel loss is minimised by predicting one half on every crease -- that is
what the ln 2 floor means -- so coherence is never trained at all. What little
of it the small model had was an artifact of not reaching the optimum. Context
and capacity let the bigger models get closer to it, and the incidental
coherence dissolves.

**The fix is the objective, not the model.** If the target is determined only up
to a global flip, the loss should be invariant to that flip: score each sheet
against the better of the two global labellings rather than against the labelled
one. Its floor is not ln 2. Changing nothing else -- same depth, same 384-pixel
crops, same 24 epochs -- the MV loss converges to 0.125 and up-to-flip accuracy
goes to **0.972**, with 92.6% of sheets internally consistent against 16.4%.
Plain accuracy stays at 0.538, which is not a failure but the theorem: the
global sign is not in the image and no loss can put it there. That the two
numbers move apart is the point: the model now decides, it just cannot know
which way, and one observed fold is enough to tell it.

**Mountain/valley is not a learning problem.** Given the light direction, the
sign of the brightness step across a crease is the answer. The learning-free
rule scores 0.907 at crease level against 0.970 for the trained U-Net. What
learning buys is *detection*: crease F1 of 0.653 against 0.260, a factor of
about two and a half.

**Pixel metrics disagree with graph metrics, badly.** Pixel F1 0.851, crease F1
0.653, geometric validity 0.314 on the same predictions. Spurious pixels are
cheap; spurious *lines* are not. A false line crossing k true creases invents k
junctions and splits k+1 edges, so its cost at the graph level grows with crease
density while its cost at the pixel level is only its area. Measured per sheet,
that is exactly what happens: sheets with fewer than 30 creases come back at
1.00 precision and recall, a 112-crease corrugation at 0.36 and 0.56.

**Decoding under the fold theorems made things worse.** Enforcing Maekawa at
each vertex dropped MV accuracy from 0.970 to 0.821. Constraints only help when
the structure they constrain is right, and only 38% of recovered vertices had
the correct degree; on the rest, enforcing a vertex condition corrupts labels the
model already had correct. Reported as a negative result, with the decoder kept
in the repository for the case where structure improves.

**The bottleneck is the mask, not the graph.** The two can be separated because
the oracle runs ground-truth masks through the same vectoriser, and they point
opposite ways. Sweeping the vectoriser's three parameters over the full
validation split moves the network's geometric validity not at all -- snap over
20..50 gives 0.226 to 0.301, threshold over 0.50..0.92 gives 0.271 to 0.301,
density over 0..4.5 gives 0.268 to 0.303 -- while the same snap sweep moves the
oracle from 0.356 to 0.489. The oracle's 0.518 is where this ends.

What moves it is not the pixel metric. Training the same network half again as
long raises pixel F1 from 0.851 to 0.859 and leaves validity where it was, 0.314
to 0.310. A depth-5 network on that same budget reaches the same pixel F1, 0.858,
and lifts validity to 0.341 and crease F1 to 0.675. Two masks that score
identically per pixel are not equally foldable, which is the divergence above
seen from the model's side rather than the vectoriser's. These are single runs
without repeated seeds, so read the 0.03 as a direction and not as a
measurement.

**Capture resolution matters more than the earlier version of this study could
show.** Crease F1 rises from 0.52 at 37 dpi to 0.81 at 149 dpi on ground-truth
masks, and is still climbing at the top of the range.

An earlier version of this study reported a plateau above 112 dpi. That was an
artifact of holding one vectoriser configuration fixed across the range: two of
its parameters are pixel quantities rather than distances on the paper, so the
extractor went progressively out of tune as the render grew, and the curve
flattened for a reason that had nothing to do with the paper. The study now
sweeps them per render size, tuning and scoring on different designs, and
reports the single-configuration curve alongside so the gap is visible. At 37
dpi one fixed configuration scores 0.049 against 0.517 tuned.

**Three parameters were each tuned against a pipeline that no longer existed.**
A density filter was written, documented and left switched off; a snapping
radius was swept at 512 px and afterwards scaled by render size, which made it
wrong everywhere else; a mask threshold was chosen while the density filter was
off, so it had to keep the mask clean by itself. Each was defensible when set.
Fixing all three moved the oracle from crease F1 0.591 to 0.708 and validity
0.475 to 0.518. `scripts/tune.py` exists so the next such change is recorded
rather than assumed, and it refuses to run on the test split.

## Layout

```text
origamicp/
  core/       CreasePattern, FOLD-format I/O
  verify/     Kawasaki, Maekawa, big-little-big; exact single-vertex solver
  generate/   random foldable patterns, the pilot design ladder, corner fiducial
  render/     physically-based scan renderer, printable SVG templates
  capture/    real-capture tooling: manifest, registration, front/back flip
  models/     targets, dataset, U-Net, metrics, learning-free baseline
  vectorize/  masks -> line segments -> planar crease graph, crease matching
  decode/     re-decoding a graph under the fold theorems
scripts/      dataset generation, training, tuning, and one script per experiment
tests/        173 tests
```

## Reproducing

```bash
pip install -e .
python scripts/make_synthetic.py --n 900 --out ../data/synth768 --size-px 768
python scripts/train.py --data ../data/synth768 --out ../runs/baseline --epochs 24
python scripts/train.py --data ../data/synth768 --out ../runs/nolight --epochs 24 --no-light
python scripts/run_all.py --data ../data/synth768 --out ../results \
    --checkpoint ../runs/baseline/best.pt \
    --no-light-checkpoint ../runs/nolight/best.pt
```

The coherence table needs three more no-light runs. Only the last differs from
the ablation in anything but its architecture or its crops:

```bash
python scripts/train.py --data ../data/synth768 --out ../runs/nl_full \
    --epochs 24 --no-light --crop 768 --batch-size 2
python scripts/train.py --data ../data/synth768 --out ../runs/nl_deep \
    --epochs 24 --no-light --depth 6
python scripts/train.py --data ../data/synth768 --out ../runs/nl_flipinv \
    --epochs 24 --no-light --flip-invariant-mv

python scripts/run_all.py --data ../data/synth768 --out ../results \
    --checkpoint ../runs/baseline/best.pt \
    --no-light-checkpoint ../runs/nolight/best.pt \
    --coherence-checkpoints ../runs/nolight/best.pt ../runs/nl_full/best.pt \
                            ../runs/nl_deep/best.pt ../runs/nl_flipinv/best.pt
```

`run_all.py` writes every figure and the console output behind every number.
The resolution and oracle studies need no checkpoint and run on a fresh clone.

`run_all.py` passes `--snap 35`, selected on validation for a 768-pixel render.
The library default is 30, which is right at 512 and is what the tests exercise.
Pointed at a dataset rendered at another size, that is the flag to re-sweep.

## Two conventions that are easy to confuse

Mountain/valley is numbered differently in two places, and mixing them is
silent. `models.targets` uses a three-way map over pixels with background at
zero (`TARGET_BACKGROUND`, `TARGET_MOUNTAIN`, `TARGET_VALLEY` = 0, 1, 2). A
model's `mv` head is two-way over creases only (`PRED_MOUNTAIN`, `PRED_VALLEY` =
0, 1). They are named apart so the mismatch cannot type-check.

## Limitations

**Synthetic only.** No physical sheets were collected, so nothing here shows the
renderer matches real paper. The renderer is grounded in the shading physics and
its difficulty is calibrated by a signal-to-texture ratio measurable on real
scans (`measure_crease_snr`), which is what a transfer study would need -- but
that study has not been done, and its absence is the largest gap.

No public dataset substitutes for it. The nearest candidates are the document
dewarping benchmarks -- WarpDoc, DocUNet -- which do contain photographs of
creased paper, and they were downloaded and examined rather than assumed
unsuitable. They carry no crease-level ground truth, their illumination is
uncontrolled and its direction unrecorded, and the sheets are not flat, so a
shadow at a lifted corner produces a brightness step an order of magnitude
larger than a crease and nothing distinguishes the two. `scripts/real_crease_snr.py`
implements the measurement for the case where controlled photographs exist.

**The identifiability result is a property of the shading model.** Real creases
are not perfectly symmetric in cross-section and real paper is not perfectly
Lambertian, so on physical sheets the symmetry is only approximate and the
ambiguity might be broken by cues the model does not have. Nothing here tests
that.

**The capture pipeline is untested on paper.** `origamicp/capture/` and the
printable templates implement a full protocol -- print on the reverse, fold, cut
the corner fiducial, scan both faces at 600 dpi, register against the digital
pattern. It is tested end to end on synthetic scans and has never seen a real
one.

**Absolute numbers are modest.** Crease F1 is 0.653 and geometric validity
0.314. The vectoriser, not the network, is what limits both: the oracle reaches
only 0.708 and 0.518 from perfect masks. Improving mask-to-graph assembly is the
obvious next step and has not been done.

**Two architectures, not a survey.** A depth-4 and a depth-5 U-Net, which agree
qualitatively. Nothing here compares architecture families.
