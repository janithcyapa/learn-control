"""
attitude
========
A Python package for attitude representation and estimation.
"""

__version__ = "0.1.0"

from .rep import (
    Quaternion,
    euler_to_quat,
    quat_to_euler,
)

from .plot import (
    plot_attitude,
)

__all__ = [
    "__version__",
    "Quaternion",
    "euler_to_quat",
    "quat_to_euler",
    "plot_attitude", 
]
