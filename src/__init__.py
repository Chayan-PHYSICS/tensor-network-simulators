"""
tebd-spin-chain
===============
A Python implementation of the Time-Evolving Block Decimation (TEBD) algorithm
for real-time evolution of Matrix Product States (MPS) in Vidal (Gamma-Lambda) form.

Modules
-------
mps         : MPS initialization and state representation
tebd        : TEBD gates and sweep logic
hamiltonian : Hamiltonian construction and Trotter gate generation
observables : Physical observables (energy, entropy, bond dimensions)
utils       : Convergence checks and helper utilities
"""
# MPS
from .mps import init_mps
# TEBD
from .tebd import apply_2site_gate, tebd_sweep
# Hamiltonian
from .hamiltonian import build_xxz_two_site, build_full_hamiltonian, build_trotter_gates
# Observables
from .observables import (
    compute_energy,
    bond_dimensions,
    max_bond_dimension,
    entanglement_entropy_bond,
    entanglement_entropy_profile,
)
# Utils
from .utils import is_converged

__all__ = [
    # MPS
    "init_mps",
    # TEBD
    "apply_2site_gate",
    "tebd_sweep",
    # Hamiltonian
    "build_xxz_two_site",
    "build_full_hamiltonian",
    "build_trotter_gates",
    # Observables
    "compute_energy",
    "bond_dimensions",
    "max_bond_dimension",
    "entanglement_entropy_bond",
    "entanglement_entropy_profile",
    # Utils
    "is_converged",
]