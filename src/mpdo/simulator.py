"""
MPDO Initialization and Full Density Matrix Reconstruction
=================================================================================================
Provides the foundational state initialization and full tensor network contraction routines for the MPDO noisy circuit simulator.

Tensor axis convention (all modules)
-------------------------------------
    T[k] : (chi_l, chi_r,   d,      kappa)
            ------ ------   -       -----
            left   right  physical  inner
            bond   bond   dim       dim (noise)

Qubit ordering note
-------------------
The MPDO chain is ordered left-to-right as qubit 0, 1, …, N−1. When contracting to a full density matrix, the row/column multi-index
follows big-endian convention:

    MPDO index  =  s_0 · 2^{N−1} + s_1 · 2^{N−2} + … + s_{N−1} · 2^0
                   ^^^
                   qubit 0 is the most significant bit

Qiskit's AerSimulator stores density matrices in little-endian order:

    Qiskit index  =  s_{N−1} · 2^{N−1} + … + s_0 · 2^0
                     ^^^^^^^
                     qubit 0 is the least significant bit

``mpdo_to_density_matrix`` applies a final axis permutation to convert from MPDO's big-endian layout to Qiskit's little-endian layout so that
fidelity comparisons are valid.

"""

from __future__ import annotations
 
import numpy as np
from scipy.linalg import svd as la_svd

from .gates     import apply1_qubit_gate, apply2_qubit_gate
from .canonical import qr_sweep_left, svd_sweep_right
from .noise     import KrausChannel

__all__ = ["MPDOState"]

class MPDOState:
    """
    Matrix Product Density Operator (MPDO) state for noisy quantum circuits.
 
    This is the primary object a user interacts with. It owns the list of site tensors and knows its own truncation bounds,
    so callers never have to pass ``chi_max`` or ``kappa_max`` around manually.
    Each site tensor has shape ``(chi_l, chi_r, d, kappa)`` where:
 
    - ``chi_l``, ``chi_r`` — left / right virtual bond dimensions (≤ chi_max)
    - ``d``                — physical (qubit) dimension, 2 for qubits
    - ``kappa``            — inner (environment / Kraus) dimension (≤ kappa_max)
 
    Construction
    ------------
    Always use the class method, not the raw constructor::
 
        state = MPDOState.init_product_state(N=6, chi_max=8, kappa_max=4)
 
    The raw ``__init__`` is reserved for internal use (e.g. copying a state).
 
    Parameters
    ----------
    tensors   : list of ndarray
        MPDO site tensors, each shape (chi_l, chi_r, d, kappa).
    chi_max   : int
        Maximum virtual bond dimension.  Applied during :meth:`canonicalize`.
    kappa_max : int
        Maximum inner (Kraus) bond dimension.  Applied inside
        :meth:`apply_noise` via :meth:`KrausChannel.apply`.
 
    Examples
    --------
    >>> state = MPDOState.init_product_state(N=4, chi_max=8, kappa_max=4)
    >>> state
    MPDOState(N=4, chi_max=8, kappa_max=4)
 
    >>> from src.mpdo.noise import Dephasing
    >>> channel = Dephasing(epsilon=0.01)
    >>> state.apply_single_qubit_gate(0, H)
    >>> state.apply_two_qubit_gate(0, 1, CNOT)
    >>> state.apply_noise(0, channel)
    >>> state.apply_noise(1, channel)
    >>> state.canonicalize()
    >>> rho = state.to_density_matrix()
    """
    def __init__(
        self,
        tensors:   list[np.ndarray],
        chi_max:   int,
        kappa_max: int,
    ) -> None:
        self.tensors   = tensors
        self.chi_max   = chi_max
        self.kappa_max = kappa_max

    # Constructor
    @classmethod
    def init_product_state(
        cls,
        N:         int,
        chi_max:   int,
        kappa_max: int,
        d:         int = 2,
    ) -> MPDOState:
        """
        Construct an N-qubit |00…0⟩⟨00…0| product state.
 
        This is the canonical starting point for any circuit simulation. Every site is initialised with bond dimension χ = 1 and inner
        dimension κ = 1 — the minimal exact representation of a pure product state.
 
        Parameters
        ----------
        N         : int   Number of qubits / sites.
        chi_max   : int   Maximum virtual bond dimension for this state.
        kappa_max : int   Maximum inner (Kraus) bond dimension for this state.
        d         : int   Physical dimension per site (default 2 for qubits).
 
        Returns
        -------
        MPDOState
            A fully configured state ready for gate and noise operations.
 
        Examples
        --------
        >>> state = MPDOState.init_product_state(N=6, chi_max=8, kappa_max=4)
        """
        return cls(init_mpdo(N, d), chi_max, kappa_max)
    
    #Properties 
    @property
    def N(self) -> int:
        """Number of qubits / sites in the chain."""
        return len(self.tensors)
    @property
    def d(self) -> int:
        """Physical dimension per site (2 for qubits)."""
        return self.tensors[0].shape[2]
    @property
    def bond_dims(self) -> list[int]:
        """
        Current virtual bond dimensions along the chain.
 
        Returns a list of length N+1 where entry k is the bond between sites k-1 and k. 
        The first and last entries are always 1 (OBC).
        
        Examples
        --------
        >>> state.bond_dims
        [1, 4, 8, 8, 4, 1]
        """
        dims = [self.tensors[0].shape[0]]          # left boundary
        for T in self.tensors:
            dims.append(T.shape[1])                # right bond of each site
        return dims
    
    # --------------------------------------- Gate methods --------------------------------------------------------
     
    def apply_single_qubit_gate(self, i: int, U: np.ndarray) -> None:
        """
        Apply a single-qubit unitary in-place at site ``i``.
 
        Parameters
        ----------
        i : int
            Site index, 0 ≤ i < N.
        U : ndarray, shape (2, 2)
            Unitary matrix (H, X, Rz, …).
 
        Examples
        --------
        >>> state.apply_single_qubit_gate(0, H_matrix)
        """
        self.tensors[i] = apply1_qubit_gate(self.tensors[i], U)
    
    def apply_two_qubit_gate(self, i: int, j: int, U: np.ndarray) -> None:
        """
        Apply a two-qubit unitary in-place across sites ``i`` and ``j = i+1``.
 
        The virtual bond between the two sites is truncated to ``chi_max``
        immediately via SVD as part of the gate application.
 
        Parameters
        ----------
        i : int
            Left site index.
        j : int
            Right site index.  Must satisfy ``j == i + 1``.
        U : ndarray, shape (4, 4)
            Two-qubit unitary (CNOT, CZ, …).
 
        Examples
        --------
        >>> state.apply_two_qubit_gate(0, 1, CNOT_matrix)
        """
        self.tensors[i], self.tensors[j], _ = apply2_qubit_gate(
            self.tensors[i], self.tensors[j], U, self.chi_max
        )
    
    # -------------------------------------- Noise method ----------------------------------------------------------------
 
    def apply_noise(self, i: int, channel: KrausChannel) -> None:
        """
        Apply a noise channel to site ``i`` and compress the environment bond.
 
        Internally calls :meth:`KrausChannel.apply`, which runs ``apply_local_noise`` then ``truncate_inner`` in sequence.
        The ``kappa_max`` bound is read from ``self.kappa_max`` automatically.
        
        Parameters
        ----------
        i       : int
            Site index, 0 ≤ i < N.
        channel : KrausChannel
            Any concrete channel: :class:`Dephasing`, :class:`AmplitudeDamping`, :class:`Depolarizing`.

        Examples
        --------
        >>> from src.mpdo.noise import Dephasing
        >>> channel = Dephasing(epsilon=0.01)
        >>> state.apply_noise(0, channel)
        >>> state.apply_noise(1, channel)
        """
        self.tensors[i] = channel.apply(self.tensors[i], self.kappa_max)
 
    # ------------------------------------- Canonicalization ------------------------------------------

    def canonicalize(self) -> None:
        """
        Perform a global gauge-fixing sweep to restore canonical form.
 
        Executes a left QR sweep followed by a right SVD sweep with truncation to ``chi_max``. Should be called once per circuit
        layer after all gates and noise operations are complete.

        """
        self.tensors = qr_sweep_left(self.tensors)
        self.tensors = svd_sweep_right(self.tensors, self.chi_max)

    # ---------------------------------------- Output -----------------------------------------------------
    def to_density_matrix(self) -> np.ndarray:
        """
        Contract the full MPDO chain into an explicit density matrix.
 
        Delegates to :func:`mpdo_to_density_matrix`.
 
        Returns
        -------
        ndarray, shape (2^N, 2^N), dtype complex
            Normalised density matrix in Qiskit's little-endian qubit ordering.
 
        Notes
        -----
        Cost is O(4^N · χ²).  Practical for N ≤ 12.
 
        Examples
        --------
        >>> rho = state.to_density_matrix()
        >>> rho.shape
        (1024, 1024)     # for N=10
        """
        return mpdo_to_density_matrix(self.tensors)
    
    # ---------------------------------- Dunder methods ------------------------------------------------------
 
    def __len__(self) -> int:
        """Return the number of sites (qubits)."""
        return self.N
 
    def __repr__(self) -> str:
        return (
            f"MPDOState("
            f"N={self.N}, "
            f"chi_max={self.chi_max}, "
            f"kappa_max={self.kappa_max})"
        )

# -------------------------------------- Private - MPDOState above ----------------------------------------------------------------
def init_mpdo(N: int, d: int = 2) -> list[np.ndarray]:
    """
    Initialize an MPDO chain in the product ground state |00…0⟩⟨00…0|.
 
    Each site is a rank-1 tensor with bond dimension χ = 1 and inner dimension κ = 1 — the minimal representation of a pure product state.
    The physical index is set to |0⟩, giving T[0, 0, 0, 0] = 1 and all other entries zero.
    
    Parameters
    ----------
    N : int
        Number of qubits / sites.
    d : int, optional
        Physical dimension per site (default 2 for qubits).
 
    Returns
    -------
    list of ndarray
        N tensors, each shape (chi_l=1, chi_r=1, d, kappa=1), dtype complex.
 
    Notes
    -----
    Not part of the public API.  Use :meth:`MPDOState.init_product_state`.
    """
    mpdo = []
    for _ in range(N):
        T = np.zeros((1, 1, d, 1), dtype=complex)
        T[0, 0, 0, 0] = 1.0          # |0⟩⟨0| at this site
        mpdo.append(T)
    return mpdo

def mpdo_to_density_matrix(mpdo: list[np.ndarray]) -> np.ndarray:
    """
    Contract the full MPDO chain into an explicit (2^N, 2^N) density matrix.
 
    For each site k the local M tensor is assembled as:
 
        M_k[s, s', l, l', r, r'] = Σ_a  T_k[l, r, s, a] · T_k*[l', r', s', a]
 
    The chain is contracted left-to-right, accumulating physical indices into a running boundary tensor C of shape (d^k, d^k, χ_r, χ_r).
    A final axis permutation converts the result from MPDO's big-endian qubit ordering to Qiskit's little-endian convention.
    
    Computational cost: O(4^N · χ²).  Practical for N ≤ 12.
 
    Parameters
    ----------
    mpdo : list of ndarray, each shape (chi_l, chi_r, d, kappa)
        MPDO chain of N site tensors.
 
    Returns
    -------
    ndarray, shape (2^N, 2^N), dtype complex
        Normalised density matrix in Qiskit's little-endian qubit ordering.
 
    Notes
    -----
    Not part of the public API.  Use :meth:`MPDOState.to_density_matrix`.
    """
    N = len(mpdo)
    d = mpdo[0].shape[2]
    T0 = mpdo[0][0]                                       # (chi_r, d, kappa)
 
    # M0[s, s', r, r'] = Σ_a T0[r, s, a] · T0*[r', s', a]
    C = np.tensordot(T0, T0.conj(), axes=([2], [2]))      # (chi_r, d, chi_r', d')
    C = C.transpose(1, 3, 0, 2)                           # (d, d', chi_r, chi_r')
 
    for i in range(1, N):
        T         = mpdo[i]          # (chi_l, chi_r, d, kap)
        chi_r_new = T.shape[1]
        D_left    = C.shape[0]       # accumulated physical dimension d^i
 
        # M_k[s, s', l, l', r, r'] = Σ_a T[l,r,s,a] · T*[l',r',s',a]
        Mk = np.tensordot(T, T.conj(), axes=([3], [3]))   # (chi_l,chi_r,d,chi_l',chi_r',d')
        Mk = Mk.transpose(2, 5, 0, 3, 1, 4)               # (d, d', chi_l, chi_l', chi_r, chi_r')
 
        # Contract C's open right bonds (axes 2, 3) with Mk's left bonds (axes 2, 3)
        C = np.tensordot(C, Mk, axes=([2, 3], [2, 3]))   # (D_left, D_left, d, d', chi_r_new, chi_r_new')
 
        # Interleave old and new physical indices, then merge
        C = C.transpose(0, 2, 1, 3, 4, 5)
        C = C.reshape(D_left * d, D_left * d, chi_r_new, chi_r_new)
 
    # -------- Finalise: squeeze the boundary right bond (dimension 1) ------------------------
    rho = C[:, :, 0, 0]
    rho /= np.trace(rho) # enforce Tr(ρ) = 1
 
    # -------- Qubit ordering correction: big-endian (MPDO) → little-endian (Qiskit) -
    rho       = rho.reshape([2] * (2 * N))
    row_axes  = list(range(N))[::-1]
    col_axes  = list(range(N, 2 * N))[::-1]
    rho       = rho.transpose(row_axes + col_axes).reshape(2**N, 2**N)
 
    return rho


