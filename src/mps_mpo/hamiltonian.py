# hamiltonian.py
"""
Hamiltonian construction and Trotter gate generation for 1-D spin-chain TEBD.

Conventions
-----------
- Local Hilbert space dimension  : d = 2  (spin-1/2).
- Two-site Hamiltonian index order: (physical_left, physical_right).
- Four-index gate shape           : (d, d, d, d)  →  (i, j, i', j').
- All energies / fields in units where ℏ = 1.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from scipy.linalg import expm
from typing import Dict, Tuple
from typing import Dict, Tuple, List

#: Local Hilbert-space dimension for spin-1/2.
D_SPIN: int = 2

# ── Pauli matrices and identity ───────────────────────────────────────────── #

SIGMA_X: npt.NDArray[np.complexfloating] = np.array([[0.0, 1.0], 
                                                     [1.0, 0.0]], dtype=complex)
    

SIGMA_Y: npt.NDArray[np.complexfloating] = np.array([[0.0, -1.0j], 
                                                     [1.0j, 0.0]], dtype=complex)
    

SIGMA_Z: npt.NDArray[np.complexfloating] = np.array([[1.0, 0.0], 
                                                     [0.0, -1.0]], dtype=complex)
    

IDENTITY: npt.NDArray[np.complexfloating] = np.eye(2, dtype=complex)

# ---------------------------------------------------------------------------
# Two-site Hamiltonian
# ---------------------------------------------------------------------------

def build_xxz_two_site(
    Jx: float,
    Jy: float,
    Jz: float,
    hz: float,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Build the three distinct two-site bond Hamiltonians for a finite chain.

    For an open-boundary chain of L sites, the on-site magnetic field
    ``-hz · σ_z`` must be distributed across bonds so that every site
    receives exactly its full field contribution. Interior sites are
    shared by two bonds and therefore receive half the field from each.
    Edge sites belong to only one bond and must receive the full field
    from that bond.

    This gives three distinct bond types:

    - **Bulk bond** ``(l, l+1)`` for ``1 ≤ l ≤ L-3``:
      site l gets  ``-hz/2``  (other half comes from the left bond),
      site l+1 gets ``-hz/2`` (other half comes from the right bond).

    - **Left edge bond** ``(0, 1)``:
      site 0 has no left neighbour → receives full ``-hz``,
      site 1 gets ``-hz/2`` (other half from its right bond).

    - **Right edge bond** ``(L-2, L-1)``:
      site L-2 gets ``-hz/2`` (other half from its left bond),
      site L-1 has no right neighbour → receives full ``-hz``.

    Parameters
    ----------
    Jx : float
        Exchange coupling along x: coefficient of σ_x ⊗ σ_x.
    Jy : float
        Exchange coupling along y: coefficient of σ_y ⊗ σ_y.
    Jz : float
        Exchange coupling along z: coefficient of σ_z ⊗ σ_z.
    hz : float
        Uniform longitudinal field strength (Pauli convention).
        Negative hz favours spin-up alignment.

    Returns
    -------
    h_left : ndarray, shape (4, 4)
        Bond Hamiltonian for the left edge bond (sites 0 and 1).
    h_bulk : ndarray, shape (4, 4)
        Bond Hamiltonian for all interior bonds.
    h_right : ndarray, shape (4, 4)
        Bond Hamiltonian for the right edge bond (sites L-2 and L-1).

    Notes
    -----
    For a chain with L = 2 only one bond exists. In that case
    ``h_left`` already carries the full field on both sites and
    ``h_bulk`` / ``h_right`` are not used.

    Examples
    --------
    >>> h_left, h_bulk, h_right = build_xxz_two_site(-1., -1., -1., -1.)
    >>> h_left.shape
    (4, 4)
    """
    # ── Exchange part: identical for all bond types ───────────────────────── #
    exchange = (
        Jx * np.kron(SIGMA_X, SIGMA_X)
        + Jy * np.kron(SIGMA_Y, SIGMA_Y)
        + Jz * np.kron(SIGMA_Z, SIGMA_Z)
    )

    # Convenience shorthands
    ZI = np.kron(SIGMA_Z, IDENTITY)
    IZ = np.kron(IDENTITY, SIGMA_Z)

    # ── Bulk bond: half field on each site ────────────────────────────────── #
    h_bulk = exchange - 0.5 * hz * (ZI + IZ)

    # ── Left edge bond: full field on site 0, half on site 1 ─────────────── #
    h_left = exchange - hz * ZI - 0.5 * hz * IZ

    # ── Right edge bond: half field on site L-2, full on site L-1 ────────── #
    h_right = exchange - 0.5 * hz * ZI - hz * IZ

    return h_left, h_bulk, h_right

#------------------------------------------------------------------------------
#  Full Hamiltonian (for exact diagonalisation reference) 
#------------------------------------------------------------------------------

def build_full_hamiltonian(
    Jx: float,
    Jy: float,
    Jz: float,
    hz: float,
    L: int,
) -> npt.NDArray[np.complexfloating]:
    """Build the full L-site XXZ Hamiltonian as a dense (2^L, 2^L) matrix.

    Intended for small systems only (L ≲ 14) where exact diagonalisation
    is feasible. Used primarily to validate TEBD results.

    Parameters
    ----------
    Jx, Jy, Jz : float
        Exchange couplings (Pauli convention).
    hz : float
        Longitudinal field strength (Pauli convention).
    L : int
        Number of sites.

    Returns
    -------
    H : ndarray, shape (2**L, 2**L)
        Full Hamiltonian matrix, Hermitian by construction.

    Examples
    --------
    >>> H = build_full_hamiltonian(-1., -1., -1., -1., L=6)
    >>> H.shape
    (64, 64)
    >>> np.allclose(H, H.conj().T)
    True
    """
    dim = 2 ** L
    H = np.zeros((dim, dim), dtype=complex)

    for l in range(L - 1):
        # Two-site operator: I^{⊗l} ⊗ h_bond ⊗ I^{⊗(L-l-2)}
        h_bond = (
            Jx * np.kron(SIGMA_X, SIGMA_X)
            + Jy * np.kron(SIGMA_Y, SIGMA_Y)
            + Jz * np.kron(SIGMA_Z, SIGMA_Z)
        )
        left_eye  = np.eye(2 ** l,       dtype=complex)
        right_eye = np.eye(2 ** (L-l-2), dtype=complex)
        H += np.kron(np.kron(left_eye, h_bond), right_eye)

    # On-site field: -hz · σ_z at every site
    for l in range(L):
        left_eye  = np.eye(2 ** l,       dtype=complex)
        right_eye = np.eye(2 ** (L-l-1), dtype=complex)
        H -= hz * np.kron(np.kron(left_eye, SIGMA_Z), right_eye)

    return H

#----------------------------------------------------------------------------------------
#  Trotter gates 
#----------------------------------------------------------------------------------------

@dataclass
class TrotterGates:
    """Half-step and full-step Suzuki–Trotter gates for one bond Hamiltonian.

    Attributes
    ----------
    U_half : ndarray, shape (2, 2, 2, 2)
        Gate exp(-i · h_bond · dt/2).
    U_full : ndarray, shape (2, 2, 2, 2)
        Gate exp(-i · h_bond · dt).
    """
    U_half: npt.NDArray[np.complexfloating]
    U_full: npt.NDArray[np.complexfloating]

def build_trotter_gates(
    h_left:  npt.NDArray[np.complexfloating],
    h_bulk:  npt.NDArray[np.complexfloating],
    h_right: npt.NDArray[np.complexfloating],
    dt: float,
) -> Tuple[
    Dict[str, npt.NDArray[np.complexfloating]],
    Dict[str, npt.NDArray[np.complexfloating]],
]:
    """Build second-order Suzuki–Trotter gate dictionaries for the XXZ chain.

    Computes the matrix exponentials:

        U_half = exp(-i · h_bond · dt/2)
        U_full = exp(-i · h_bond · dt)

    for each of the three bond types (left edge, bulk, right edge), and
    packages them into two ready-to-use gate dictionaries — one for the
    even-bond half-step sweeps and one for the odd-bond full-step sweep.

    A complete second-order Suzuki–Trotter step is then::

        tebd_sweep(gammas, lambdas, even_gates, chi_max, parity=0)
        tebd_sweep(gammas, lambdas, odd_gates,  chi_max, parity=1)
        tebd_sweep(gammas, lambdas, even_gates, chi_max, parity=0)

    Parameters
    ----------
    h_left : ndarray, shape (4, 4)
        Bond Hamiltonian for the left edge bond (sites 0 and 1).
        Returned by :func:`build_xxz_two_site`.
    h_bulk : ndarray, shape (4, 4)
        Bond Hamiltonian for all interior bonds.
        Returned by :func:`build_xxz_two_site`.
    h_right : ndarray, shape (4, 4)
        Bond Hamiltonian for the right edge bond (sites L-2 and L-1).
        Returned by :func:`build_xxz_two_site`.
    dt : float
        Time step size. For imaginary time evolution pass ``-1j * tau``.

    Returns
    -------
    even_gates : dict with keys ``"left"``, ``"bulk"``, ``"right"``
        Half-step gates (dt/2) for the even-bond sweeps (parity=0).
    odd_gates : dict with keys ``"left"``, ``"bulk"``, ``"right"``
        Full-step gates (dt) for the odd-bond sweep (parity=1).

    Raises
    ------
    ValueError
        If any bond Hamiltonian does not have shape (4, 4), is not
        Hermitian, or dt is zero.

    Examples
    --------
    >>> h_left, h_bulk, h_right = build_xxz_two_site(-1., -1., -1., -1.)
    >>> even_gates, odd_gates = build_trotter_gates(h_left, h_bulk, h_right, dt=0.005)
    >>> even_gates["bulk"].shape
    (2, 2, 2, 2)
    >>> # Verify unitarity of bulk odd gate
    >>> U = odd_gates["bulk"].reshape(4, 4)
    >>> np.allclose(U @ U.conj().T, np.eye(4))
    True
    """
    # ── Validation ────────────────────────────────────────────────────────── #
    for name, h in [("h_left", h_left), ("h_bulk", h_bulk), ("h_right", h_right)]:
        if h.shape != (4, 4):
            raise ValueError(
                f"{name} must have shape (4, 4), got {h.shape}."
            )
        if not np.allclose(h, h.conj().T, atol=1e-12):
            raise ValueError(
                f"{name} must be Hermitian. "
                f"Max asymmetry: {np.max(np.abs(h - h.conj().T)):.2e}."
            )
    if dt == 0:
        raise ValueError("dt must be non-zero.")

    # ── Internal gate builder ─────────────────────────────────────────────── #
    def _gate(h: npt.NDArray, tau: float) -> npt.NDArray:
        """Compute exp(-i · h · tau) reshaped to (d, d, d, d)."""
        return expm(-1j * h * tau).reshape(2, 2, 2, 2)

    # ── Assemble gate dicts ───────────────────────────────────────────────── #
    even_gates = {
        "left":  _gate(h_left,  dt / 2),
        "bulk":  _gate(h_bulk,  dt / 2),
        "right": _gate(h_right, dt / 2),
    }
    odd_gates = {
        "left":  _gate(h_left,  dt),
        "bulk":  _gate(h_bulk,  dt),
        "right": _gate(h_right, dt),
    }

    return even_gates, odd_gates


#----------------------------------------------NEW ADD-----------------------------------------------
def build_xxz_mpo(
    L: int,
    Jx: float,
    Jy: float,
    Jz: float,
    hx: float,
    hz: float,
    S: float = 0.5,
) -> List[np.ndarray]:
    """Construct the MPO representation of the generalized XXZ Hamiltonian.

    Implements the Hamiltonian

        H = Σ_i [ Jx·Sx_i·Sx_{i+1} + Jy·Sy_i·Sy_{i+1} + Jz·Sz_i·Sz_{i+1} ]
              - Σ_i [ hx·Sx_i + hz·Sz_i ]

    as a list of rank-4 MPO tensors with index ordering ``(D_L, D_R, d, d)``,
    where ``D`` is the MPO bond dimension and ``d = 2S + 1`` is the local
    physical dimension.

    The finite-state-machine (FSM) structure uses bond dimension ``D = 5``
    with the following auxiliary states:

    =========  =============================================
    Index      Role
    =========  =============================================
    0          Vacuum / rightward propagation ("start")
    1          Carrying Sx (awaiting right partner)
    2          Carrying Sy (awaiting right partner)
    3          Carrying Sz (awaiting right partner)
    4          Energy accumulator ("done")
    =========  =============================================

    The bulk W tensor as a block matrix (each block is a (d x d) operator):

               to:  0     1     2     3        4
        from  0  [  I     Sx    Sy    Sz    H_local ]
              1  [  0     0     0     0     Jx·Sx   ]
              2  [  0     0     0     0     Jy·Sy   ]
              3  [  0     0     0     0     Jz·Sz   ]
              4  [  0     0     0     0       I     ]

    W[α, β] is the local operator applied at this site when the MPO bond
    transitions from auxiliary state α (incoming, left bond) to state β
    (outgoing, right bond).

    Parameters
    ----------
    L : int
        Number of lattice sites.
    S : float, optional
        Spin quantum number (default ``0.5``).  Currently only ``S=0.5``
        is implemented; other values raise ``NotImplementedError``.
    Jx, Jy, Jz : float, optional
        Exchange couplings along x, y, z (default ``1.0`` each).
    hx, hz : float, optional
        Transverse and longitudinal magnetic field strengths (default ``0.0``).

    Returns
    -------
    mpo_list : list of np.ndarray
        Length-``L`` list of MPO tensors.  Boundary tensors have shapes
        ``(1, D, d, d)`` (left edge) and ``(D, 1, d, d)`` (right edge);
        bulk tensors have shape ``(D, D, d, d)``.

    Raises
    ------
    ValueError
        If ``L < 2``.
    NotImplementedError
        If ``S != 0.5``.
    """
    if L < 2:
        raise ValueError(f"Chain length must be at least 2, got L={L}.")
    if S != 0.5:
        raise NotImplementedError(
            f"Spin-S MPO is not yet implemented for S={S}; only S=0.5 is supported."
        )

    # ------------------------------------------------------------------ #
    # Spin-1/2 operators                                                   #
    # ------------------------------------------------------------------ #
    # I  = np.eye(2, dtype=complex)
    # Sx = 0.5 * np.array([[0,  1 ], [1,  0 ]], dtype=complex)
    # Sy = 0.5 * np.array([[0, -1j], [1j, 0 ]], dtype=complex)
    # Sz = 0.5 * np.array([[1,  0 ], [0, -1 ]], dtype=complex)
    # d  = I.shape[0]

    Sx = 0.5 * SIGMA_X
    Sy = 0.5 * SIGMA_Y
    Sz = 0.5 * SIGMA_Z
    d  = IDENTITY.shape[0]

    # ------------------------------------------------------------------ #
    # Bulk MPO tensor  W[D_L, D_R, d, d]                                  #
    # ------------------------------------------------------------------ #
    D = 5
    W = np.zeros((D, D, d, d), dtype=complex)

    H_local = -hx * Sx - hz * Sz   # one-body term accumulated at each site

    # Row 0 — "start" state: fan out to all interaction channels
    W[0, 0] = IDENTITY
    W[0, 1] = Sx
    W[0, 2] = Sy
    W[0, 3] = Sz
    W[0, 4] = H_local

    # Rows 1–3 — "carrying" states: complete the two-site interaction
    W[1, 4] = Jx * Sx
    W[2, 4] = Jy * Sy
    W[3, 4] = Jz * Sz

    # Row 4 — "done" state: propagate accumulated energy to the right
    W[4, 4] = IDENTITY

    # ------------------------------------------------------------------ #
    # Boundary projections                                                 #
    # ------------------------------------------------------------------ #
    W_left  = W[0:1, :, :, :]   # shape (1, D, d, d) — only "start" row
    W_right = W[:, 4:5, :, :]   # shape (D, 1, d, d) — only "done" column

    mpo_list = (
        [W_left]
        + [W.copy() for _ in range(L - 2)]
        + [W_right]
    )
    return mpo_list
