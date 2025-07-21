from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class OptimizationState:
    """Represents a single state in an optimization problem."""

    x: np.ndarray
    q: np.ndarray
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    w: np.ndarray = field(default_factory=lambda: np.zeros(3))
    u: np.ndarray = field(default_factory=lambda: np.zeros(6))
    i: int = 0

    def __post_init__(self):
        # Ensure arrays are NumPy arrays with correct shapes
        self.x = np.array(self.x, dtype=float)
        self.q = np.array(self.q, dtype=float)
        self.v = np.array(self.v, dtype=float)
        self.w = np.array(self.w, dtype=float)
        self.u = np.array(self.u, dtype=float)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, OptimizationState):
            return NotImplemented
        return (
            np.allclose(self.x, other.x)
            and np.allclose(self.q, other.q)
            and np.allclose(self.v, other.v)
            and np.allclose(self.w, other.w)
            and np.allclose(self.u, other.u)
            and self.i == other.i
        )

    def get_state(self) -> np.ndarray:
        """Return the full state vector as a flat array."""
        return np.concatenate([self.x, self.v, self.q, self.w])

    def to_dict(self) -> dict[str, Any]:
        """Convert the object to a serializable dictionary."""
        return {
            "x": self.x.tolist(),
            "v": self.v.tolist(),
            "q": self.q.tolist(),
            "w": self.w.tolist(),
            "u": self.u.tolist(),
            "i": self.i,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> OptimizationState:
        """Reconstruct an OptimizationState from a dictionary."""
        return OptimizationState(
            x=np.array(d["x"]),
            v=np.array(d.get("v", [0.0, 0.0, 0.0])),
            q=np.array(d["q"]),
            w=np.array(d.get("w", [0.0, 0.0, 0.0])),
            u=np.array(d.get("u", [0.0] * 6)),
            i=int(d.get("i", 0)),
        )
