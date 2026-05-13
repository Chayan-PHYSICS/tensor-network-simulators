"""
Advanced Quantum Tensor Network Simulation Suite.

A unified framework for classical simulations of quantum systems, spanning closed 
many-body dynamics (MPS/MPO) and open, noisy quantum computational circuits (MPDO).
"""

# Expose the subpackages cleanly to the absolute top layer namespace
from . import mps_mpo
from . import mpdo

__all__ = [
    "mps_mpo",
    "mpdo",
    "MPDOState",
    "Dephasing",
    "AmplitudeDamping",
    "get_kraus_ops"

]