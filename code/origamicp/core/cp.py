"""Crease-pattern container and FOLD-format I/O.

We read and write the FOLD spec (Demaine et al.) directly instead of inventing a
format, because Flat-Folder, Origami Simulator and OrigamiBench all speak it.
That keeps our extracted patterns pluggable into existing solvers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MOUNTAIN, VALLEY, BOUNDARY, FLAT, UNASSIGNED = "M", "V", "B", "F", "U"
FOLDED = (MOUNTAIN, VALLEY)
VALID_ASSIGNMENTS = frozenset({MOUNTAIN, VALLEY, BOUNDARY, FLAT, UNASSIGNED})

TWO_PI = 2.0 * np.pi


@dataclass
class CreasePattern:
    """A planar crease pattern: vertices, straight creases, MV assignment.

    ``vertices``   (N, 2) float64 coordinates in paper space.
    ``edges``      (M, 2) int64 vertex-index pairs.
    ``assignment`` (M,) one of M / V / B / F / U per FOLD.
    """

    vertices: np.ndarray
    edges: np.ndarray
    assignment: np.ndarray

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 2)
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        self.assignment = np.asarray(self.assignment, dtype="<U1").reshape(-1)

        if len(self.edges) != len(self.assignment):
            raise ValueError(
                f"{len(self.edges)} edges but {len(self.assignment)} assignments"
            )
        if len(self.edges) and self.edges.max() >= len(self.vertices):
            raise ValueError("edge references a vertex index that does not exist")
        if (self.edges[:, 0] == self.edges[:, 1]).any():
            raise ValueError("self-loop edge")
        bad = set(np.unique(self.assignment)) - VALID_ASSIGNMENTS
        if bad:
            raise ValueError(f"unknown edge assignments: {sorted(bad)}")

        self._adjacency = [[] for _ in range(len(self.vertices))]
        for e, (a, b) in enumerate(self.edges):
            self._adjacency[a].append(e)
            self._adjacency[b].append(e)

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def incident_edges(self, v: int) -> list[int]:
        return self._adjacency[v]

    def other_end(self, edge: int, v: int) -> int:
        a, b = self.edges[edge]
        return int(b) if int(a) == v else int(a)

    def is_boundary_vertex(self, v: int) -> bool:
        """A vertex on the paper's border, where the fold theorems do not apply."""
        return any(self.assignment[e] == BOUNDARY for e in self._adjacency[v])

    def interior_vertices(self) -> list[int]:
        return [v for v in range(self.n_vertices) if not self.is_boundary_vertex(v)]

    def boundary_polygon(self) -> np.ndarray:
        """Vertex indices tracing the sheet outline, in order.

        The outline is not always a rectangle -- a corner may be cut off as an
        orientation fiducial -- so anything that needs the sheet's silhouette
        must follow the boundary edges rather than assume four corners.
        """
        neighbours: dict[int, list[int]] = {}
        for e, (a, b) in enumerate(self.edges):
            if self.assignment[e] != BOUNDARY:
                continue
            neighbours.setdefault(int(a), []).append(int(b))
            neighbours.setdefault(int(b), []).append(int(a))
        if not neighbours:
            return np.zeros(0, dtype=np.int64)

        start = min(neighbours)
        cycle, previous, current = [start], None, start
        while True:
            options = [v for v in neighbours[current] if v != previous]
            if not options:
                break
            previous, current = current, options[0]
            if current == start:
                break
            cycle.append(current)
        return np.array(cycle, dtype=np.int64)

    def sorted_creases(self, v: int) -> tuple[np.ndarray, np.ndarray]:
        """Edges incident to ``v``, sorted counter-clockwise.

        Returns ``(edge_indices, directions)`` where ``directions`` holds the
        outgoing angle of each crease in [0, 2pi).
        """
        edges = np.asarray(self._adjacency[v], dtype=np.int64)
        if len(edges) == 0:
            return edges, np.zeros(0)
        deltas = np.array(
            [self.vertices[self.other_end(e, v)] - self.vertices[v] for e in edges]
        )
        theta = np.mod(np.arctan2(deltas[:, 1], deltas[:, 0]), TWO_PI)
        order = np.argsort(theta)
        return edges[order], theta[order]

    def sector_angles(self, v: int) -> np.ndarray:
        """Angles of the paper sectors between consecutive creases around ``v``.

        For an interior vertex these sum to 2pi. ``sector_angles(v)[i]`` sits
        between ``sorted_creases(v)[0][i]`` and its cyclic successor, which is
        the indexing Kawasaki's and the big-little-big conditions assume.
        """
        _, theta = self.sorted_creases(v)
        if len(theta) < 2:
            return np.zeros(0)
        return np.mod(np.roll(theta, -1) - theta, TWO_PI)

    def mv_around(self, v: int) -> np.ndarray:
        """MV assignment of the creases around ``v``, in the same CCW order."""
        edges, _ = self.sorted_creases(v)
        return self.assignment[edges]

    @classmethod
    def from_fold(cls, path: str | Path) -> "CreasePattern":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_fold_dict(data)

    @classmethod
    def from_fold_dict(cls, data: dict) -> "CreasePattern":
        coords = np.asarray(data["vertices_coords"], dtype=np.float64)
        if coords.shape[1] > 2:  # tolerate 3D coords from folded-state files
            coords = coords[:, :2]
        edges = np.asarray(data["edges_vertices"], dtype=np.int64)
        assignment = data.get("edges_assignment")
        if assignment is None:
            assignment = [UNASSIGNED] * len(edges)
        return cls(coords, edges, np.asarray(assignment, dtype="<U1"))

    def to_fold_dict(self, name: str = "origamicp") -> dict:
        return {
            "file_spec": 1.1,
            "file_creator": "origamicp",
            "frame_title": name,
            "frame_classes": ["creasePattern"],
            "vertices_coords": self.vertices.tolist(),
            "edges_vertices": self.edges.tolist(),
            "edges_assignment": self.assignment.tolist(),
        }

    def to_fold(self, path: str | Path, name: str = "origamicp") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_fold_dict(name), indent=1), encoding="utf-8")
