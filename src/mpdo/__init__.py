"""
Matrix Product Density Operator (MPDO) Simulation Package.

core algorithms for running noisy, open-system quantum circuit simulations using Tensor Networks.
"""

from .simulator import MPDOState
from .gates import apply1_qubit_gate, apply2_qubit_gate
from .noise import KrausChannel, AmplitudeDamping, Dephasing, Depolarizing, get_kraus_ops
from .canonical import qr_sweep_left, svd_sweep_right

# Define the explicit public API for this subpackage
__all__ = [
    "MPDOState",
    "apply1_qubit_gate",
    "apply2_qubit_gate",
    "KrausChannel",
    "AmplitudeDamping",
    "Dephasing",
    "Depolarizing",
    "get_kraus_ops",
    "qr_sweep_left",
    "svd_sweep_right",
]