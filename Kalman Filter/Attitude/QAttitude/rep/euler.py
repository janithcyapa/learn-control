import numpy as np
from .quaternion import Quaternion

_EPS = 1e-12


def euler_to_quat(roll: float, pitch: float, yaw: float) -> Quaternion:
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (w,x,y,z).
    Convention: ZYX intrinsic (yaw->pitch->roll), aka R = Rz(yaw) Ry(pitch) Rx(roll)
    """
    cr = np.cos(roll * 0.5);  sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5); sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5);   sy = np.sin(yaw * 0.5)

    w = cy*cp*cr + sy*sp*sr
    x = cy*cp*sr - sy*sp*cr
    y = sy*cp*sr + cy*sp*cr
    z = sy*cp*cr - cy*sp*sr

    return Quaternion(w, x, y, z).normalized().canonical()


def quat_to_euler(q: Quaternion) -> tuple[float, float, float]:
    """
    Convert quaternion (w,x,y,z) to Euler angles (roll, pitch, yaw).
    Convention: ZYX intrinsic.
    Returns (roll, pitch, yaw) in radians.
    """
    q = q.normalized()
    w, x, y, z = q.w, q.x, q.y, q.z

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w*y - z*x)
    # clamp to handle numeric errors
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return float(roll), float(pitch), float(yaw)
