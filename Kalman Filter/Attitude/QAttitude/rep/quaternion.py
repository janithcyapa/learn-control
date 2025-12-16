from __future__ import annotations
from dataclasses import dataclass
import numpy as np

_EPS = 1e-12


def _as_vec3(v) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.size != 3:
        raise ValueError("Expected a 3D vector.")
    return v


@dataclass(frozen=True)
class Quaternion:
    """
    Quaternion stored as (w, x, y, z).

    Convention:
    - Unit quaternion represents rotation.
    - Rotating a 3D vector v: v' = q ⊗ [0,v] ⊗ q^{-1}
    """
    w: float
    x: float
    y: float
    z: float

    # ---------- Constructors ----------
    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion(1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_array(a) -> "Quaternion":
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.size != 4:
            raise ValueError("Expected array-like of length 4 (w,x,y,z).")
        return Quaternion(float(a[0]), float(a[1]), float(a[2]), float(a[3]))

    def as_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z], dtype=float)

    # ---------- Basic ops ----------
    def norm(self) -> float:
        return float(np.linalg.norm(self.as_array()))

    def normalized(self) -> "Quaternion":
        n = self.norm()
        if n < _EPS:
            raise ZeroDivisionError("Cannot normalize near-zero quaternion.")
        a = self.as_array() / n
        return Quaternion.from_array(a)

    def conjugate(self) -> "Quaternion":
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def inverse(self) -> "Quaternion":
        # For unit quaternions, inverse == conjugate
        a = self.as_array()
        n2 = float(a @ a)
        if n2 < _EPS:
            raise ZeroDivisionError("Cannot invert near-zero quaternion.")
        return Quaternion(self.w / n2, -self.x / n2, -self.y / n2, -self.z / n2)

    # Hamilton product (composition)
    def __mul__(self, other: "Quaternion") -> "Quaternion":
        if not isinstance(other, Quaternion):
            return NotImplemented
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        return Quaternion(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        )

    def dot(self, other: "Quaternion") -> float:
        return float(self.as_array() @ other.as_array())

    def canonical(self) -> "Quaternion":
        """
        Make the quaternion unique by forcing w >= 0 (optional but useful).
        Note: q and -q represent the same rotation.
        """
        return Quaternion(-self.w, -self.x, -self.y, -self.z) if self.w < 0 else self

    # ---------- Rotation operator ----------
    def rotate(self, v) -> np.ndarray:
        """
        Rotate 3D vector v by this quaternion (assumes this quaternion is a rotation).
        """
        v = _as_vec3(v)
        q = self.normalized()
        # Efficient formula: v' = v + 2*cross(q_vec, cross(q_vec, v) + w*v)
        qv = np.array([q.x, q.y, q.z], dtype=float)
        t = 2.0 * np.cross(qv, v)
        v_prime = v + q.w * t + np.cross(qv, t)
        return v_prime

    def to_rotation_matrix(self) -> np.ndarray:
        """
        3x3 rotation matrix corresponding to this quaternion (w,x,y,z).
        """
        q = self.normalized()
        w, x, y, z = q.w, q.x, q.y, q.z
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ], dtype=float)

    @staticmethod
    def from_rotation_matrix(R: np.ndarray) -> "Quaternion":
        """
        Create quaternion (w,x,y,z) from rotation matrix.
        """
        R = np.asarray(R, dtype=float)
        if R.shape != (3, 3):
            raise ValueError("R must be 3x3.")

        tr = float(np.trace(R))
        if tr > 0.0:
            S = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        else:
            # Find the largest diagonal element
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                w = (R[2, 1] - R[1, 2]) / S
                x = 0.25 * S
                y = (R[0, 1] + R[1, 0]) / S
                z = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                w = (R[0, 2] - R[2, 0]) / S
                x = (R[0, 1] + R[1, 0]) / S
                y = 0.25 * S
                z = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                w = (R[1, 0] - R[0, 1]) / S
                x = (R[0, 2] + R[2, 0]) / S
                y = (R[1, 2] + R[2, 1]) / S
                z = 0.25 * S

        return Quaternion(w, x, y, z).normalized().canonical()

    # ---------- Useful helpers ----------
    def __repr__(self) -> str:
        return f"Quaternion(w={self.w:.6g}, x={self.x:.6g}, y={self.y:.6g}, z={self.z:.6g})"
