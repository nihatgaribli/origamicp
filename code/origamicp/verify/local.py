"""Exact local flat-foldability conditions.

Scope note: everything here is a *necessary* condition that is checkable per
vertex in O(degree). Global flat-foldability is NP-hard (Bern & Hayes, 1996), so
we do not reimplement it -- global ground truth is delegated to Flat-Folder, the
same solver OrigamiBench uses. See ``origamicp/verify/global_solver.py``.

The three conditions below are what we need for the paper's evaluation metric:
a crease graph recovered from a photograph is *geometrically valid* if every
interior vertex satisfies all three. Unlike pixel IoU this metric is objective,
scale-free, and directly meaningful for downstream folding.

``tol`` matters. On synthetic patterns it can be ~1e-9; on graphs extracted from
photographs the vertex positions are noisy, so validity must be reported as a
function of angular tolerance (that sweep is itself a result, not a nuisance).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from origamicp.core.cp import FOLDED, MOUNTAIN, VALLEY, CreasePattern

DEFAULT_TOL = np.deg2rad(1.0)


def kawasaki_residual(angles: np.ndarray) -> float:
    """Alternating sum of sector angles; zero iff Kawasaki's condition holds.

    Kawasaki: an interior vertex is flat-foldable (i.e. *some* valid MV
    assignment exists) iff it has even degree and a1 - a2 + a3 - ... = 0.
    Returned in radians so it can be compared against an angular tolerance.
    """
    if len(angles) == 0:
        return 0.0
    signs = (-1.0) ** np.arange(len(angles))
    return float(abs(signs @ angles))


def maekawa_defect(mv: np.ndarray) -> int:
    """#mountains - #valleys. A flat-foldable interior vertex needs exactly +-2."""
    n_m = int(np.sum(mv == MOUNTAIN))
    n_v = int(np.sum(mv == VALLEY))
    return n_m - n_v


def big_little_big_violations(
    angles: np.ndarray, mv: np.ndarray, tol: float = DEFAULT_TOL
) -> list[int]:
    """Sectors that violate the big-little-big lemma.

    If sector ``i`` is strictly smaller than both neighbours, the two creases
    bounding it must fold in opposite directions -- otherwise the little sector
    cannot tuck between them. Returns the offending sector indices.

    This catches errors Maekawa and Kawasaki both miss, which matters here: a
    model that flips one crease's MV label usually leaves the angles untouched.
    """
    k = len(angles)
    if k < 3:
        return []
    violations = []
    for i in range(k):
        prev_a, this_a, next_a = angles[i - 1], angles[i], angles[(i + 1) % k]
        if this_a < prev_a - tol and this_a < next_a - tol:
            left, right = mv[i], mv[(i + 1) % k]
            if left in FOLDED and right in FOLDED and left == right:
                violations.append(i)
    return violations


@dataclass
class VertexReport:
    vertex: int
    degree: int
    interior: bool
    even_degree: bool
    kawasaki_residual: float
    kawasaki_ok: bool
    maekawa_defect: int
    maekawa_ok: bool
    blb_violations: list[int] = field(default_factory=list)

    @property
    def blb_ok(self) -> bool:
        return not self.blb_violations

    @property
    def ok(self) -> bool:
        """Boundary vertices are unconstrained, so they always pass."""
        if not self.interior:
            return True
        return self.even_degree and self.kawasaki_ok and self.maekawa_ok and self.blb_ok


@dataclass
class PatternReport:
    vertices: list[VertexReport]
    tol: float

    @property
    def interior(self) -> list[VertexReport]:
        return [r for r in self.vertices if r.interior]

    def _rate(self, predicate) -> float:
        interior = self.interior
        if not interior:
            return float("nan")
        return sum(bool(predicate(r)) for r in interior) / len(interior)

    @property
    def vertex_validity_rate(self) -> float:
        """Headline metric: fraction of interior vertices that are locally valid."""
        return self._rate(lambda r: r.ok)

    @property
    def kawasaki_rate(self) -> float:
        return self._rate(lambda r: r.kawasaki_ok)

    @property
    def maekawa_rate(self) -> float:
        return self._rate(lambda r: r.maekawa_ok)

    @property
    def blb_rate(self) -> float:
        return self._rate(lambda r: r.blb_ok)

    @property
    def valid(self) -> bool:
        """Strictest metric: every interior vertex passes."""
        return all(r.ok for r in self.vertices)

    def failures(self) -> list[VertexReport]:
        return [r for r in self.interior if not r.ok]

    def summary(self) -> str:
        n = len(self.interior)
        return (
            f"interior vertices: {n} | valid: {self.vertex_validity_rate:.3f} | "
            f"kawasaki: {self.kawasaki_rate:.3f} | maekawa: {self.maekawa_rate:.3f} | "
            f"big-little-big: {self.blb_rate:.3f} | tol: {np.rad2deg(self.tol):.2f} deg"
        )


def verify_vertex(cp: CreasePattern, v: int, tol: float = DEFAULT_TOL) -> VertexReport:
    angles = cp.sector_angles(v)
    mv = cp.mv_around(v)
    degree = len(cp.incident_edges(v))
    interior = not cp.is_boundary_vertex(v)

    residual = kawasaki_residual(angles)
    defect = maekawa_defect(mv)
    even = degree % 2 == 0

    return VertexReport(
        vertex=v,
        degree=degree,
        interior=interior,
        even_degree=even,
        kawasaki_residual=residual,
        # Kawasaki presupposes even degree; an odd-degree interior vertex fails
        # outright rather than accidentally passing on a near-zero residual.
        kawasaki_ok=even and residual <= tol,
        maekawa_defect=defect,
        maekawa_ok=abs(defect) == 2,
        blb_violations=big_little_big_violations(angles, mv, tol),
    )


def verify(cp: CreasePattern, tol: float = DEFAULT_TOL) -> PatternReport:
    """Check every vertex of a crease pattern against the local conditions."""
    return PatternReport(
        vertices=[verify_vertex(cp, v, tol) for v in range(cp.n_vertices)],
        tol=tol,
    )


def validity_vs_tolerance(
    cp: CreasePattern, tolerances_deg: np.ndarray | None = None
) -> list[tuple[float, float]]:
    """Sweep angular tolerance and report validity rate at each level.

    Extraction from photographs trades precision for recall; this sweep is how
    we report that honestly instead of quoting a single tuned tolerance.
    """
    if tolerances_deg is None:
        tolerances_deg = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
    return [
        (float(t), verify(cp, tol=np.deg2rad(t)).vertex_validity_rate)
        for t in tolerances_deg
    ]
