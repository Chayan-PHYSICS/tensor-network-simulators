"""
tebd.py
-------
Two-site gate application and single-parity TEBD sweep for Matrix Product
States in Vidal (Γ–Λ) canonical form.

The sweep function applies gates to either even or odd bonds only,
allowing the caller to compose a second-order Suzuki–Trotter step as:

    sweep(gammas, lambdas, U_half, chi_max, parity=EVEN)   # even, dt/2
    sweep(gammas, lambdas, U_full, chi_max, parity=ODD)    # odd,  dt
    sweep(gammas, lambdas, U_half, chi_max, parity=EVEN)   # even, dt/2

References
----------
Vidal, G. (2004). Efficient Simulation of One-Dimensional Quantum
Many-Body Systems. Physical Review Letters, 93(4), 040502.
https://doi.org/10.1103/PhysRevLett.93.040502
 
Vidal, G. (2003). Efficient Classical Simulation of Slightly Entangled
Quantum Computations. Physical Review Letters, 91(14), 147902.
https://doi.org/10.1103/PhysRevLett.91.147902
"""

from __future__ import annotations

import logging
import numpy as np
import numpy.typing as npt
from scipy.linalg import svd as la_svd
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Sequence

__all__ = ["apply_2site_gate", "tebd_sweep", "SweepParity"]

logger = logging.getLogger(__name__)

# Singular values below this threshold are treated as zero.
_SVD_ZERO_THRESHOLD: float = 1e-12


# ------------------------ Helper--------------------------------------------------- #

def safe_inv(
    lam: npt.NDArray[np.floating],
    eps: float = 1e-12,
) -> npt.NDArray[np.floating]:
    """Pseudo-inverse of a 1-D Schmidt vector.

    Inverts values above ``eps``; zeros out the rest to avoid
    division by numerically negligible singular values.
    """
    inv = np.zeros_like(lam)
    mask = lam > eps
    inv[mask] = 1.0 / lam[mask]
    return inv


# ----------------------------- Sweep result container----------------------------------- #

@dataclass
class FullSweepResult:
    """Container for the results of a single-parity TEBD sweep.

    Attributes
    ----------
    total_truncation_error : float
        Accumulated truncation error summed over all gate applications
        in this sweep.
    updated_bonds : list[int]
        Left-site indices of each updated bond, in processing order.
        The singular-value array updated in ``lambdas`` lives at
        index ``bond + 1``.
    per_bond_errors : list[float]
        Individual truncation error for each gate application,
        in the same order as ``updated_bonds``.
    parity_used : int
        The parity that was swept — 0 for even bonds, 1 for odd bonds.
    """
    total_truncation_error: float = 0.0
    updated_bonds: List[int] = field(default_factory=list)
    per_bond_errors: List[float] = field(default_factory=list)
    parity_used: int = 0

    def __repr__(self) -> str:
        parity_str = "even" if self.parity_used == 0 else "odd"
        return (
            f"FullSweepResult("
            f"parity={parity_str!r}, "
            f"total_error={self.total_truncation_error:.6e}, "
            f"n_bonds={len(self.updated_bonds)})"
        )


class SweepParity:
    """Symbolic constants for sweep parity selection.

    Use ``SweepParity.EVEN`` and ``SweepParity.ODD`` instead of raw
    integers to make call sites self-documenting.
    """
    EVEN: int = 0
    ODD:  int = 1


# ------------------------------- Gate application ------------------------------------------------- #
def apply_2site_gate(
    A: npt.NDArray[np.complexfloating],
    B: npt.NDArray[np.complexfloating],
    lam_l: npt.NDArray[np.floating],
    lam_m: npt.NDArray[np.floating],
    lam_r: npt.NDArray[np.floating],
    U: npt.NDArray[np.complexfloating],
    chi_max: int,
) -> Tuple[
    npt.NDArray[np.complexfloating],
    npt.NDArray[np.floating],
    npt.NDArray[np.complexfloating],
    float,
]:
    """Apply a two-site unitary gate to neighbouring sites in Vidal form.
 
    The update follows the standard TEBD procedure:
 
    1. Build the two-site tensor  ``M = Γ[l] · Λ[l+1] · Γ[l+1]``.
    2. Apply the gate *U* to the physical indices of *M* only.
    3. Absorb the outer Schmidt vectors to form the full two-site tensor
       ``Θ = Λ[l] · M_U · Λ[l+2]``.
    4. Reshape *Θ* and compute its SVD.
    5. Truncate to at most *chi_max* singular values.
    6. Reconstruct updated Gamma tensors by stripping the outer lambdas.
 
    Tensor index conventions
    ------------------------
    - Gamma tensors: ``Γ[α, σ, β]``  with  ``α`` left bond, ``σ`` physical,
      ``β`` right bond.
    - Lambda vectors: one-dimensional Schmidt spectra on each bond.
    - Gate *U*: ``U[σ', τ', σ, τ]``  (output physical indices first).
 
    Parameters
    ----------
    A:
        Left Gamma tensor ``Γ^[l]``, shape ``(χ_L, d, χ_M)``.
    B:
        Right Gamma tensor ``Γ^[l+1]``, shape ``(χ_M, d, χ_R)``.
    lam_l:
        Schmidt vector ``Λ[l]`` on the bond to the left of site *l*,
        shape ``(χ_L,)``.
    lam_m:
        Schmidt vector ``Λ[l+1]`` on the bond between sites *l* and
        *l+1*, shape ``(χ_M,)``.
    lam_r:
        Schmidt vector ``Λ[l+2]`` on the bond to the right of site *l+1*,
        shape ``(χ_R,)``.
    U:
        Two-site gate, shape ``(d, d, d, d)`` with index order
        ``(σ', τ', σ, τ)``.
    chi_max:
        Maximum bond dimension kept after truncation.  Must be ≥ 1.
 
    Returns
    -------
    A_new:
        Updated Gamma tensor ``Γ^[l]``, shape ``(χ_L, d, χ_new)``.
    lam_new:
        Updated, normalised Schmidt vector ``Λ[l+1]``,
        shape ``(χ_new,)``.  Satisfies ``‖lam_new‖₂ = 1``.
    B_new:
        Updated Gamma tensor ``Γ^[l+1]``, shape ``(χ_new, d, χ_R)``.
    trunc_error:
        Discarded weight  ``Σ_{k > χ_new} λ_k²``  computed on the
        *unnormalised* singular values before truncation.
 
    Raises
    ------
    ValueError
        If any tensor shapes are inconsistent or *chi_max* < 1.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("A and B must each be rank-3 tensors.")
    if U.ndim != 4:
        raise ValueError("Gate U must be rank-4 with shape (d, d, d, d).")
    if chi_max < 1:
        raise ValueError(f"chi_max must be a positive integer, got {chi_max}.")
 
    chi_L, d, chi_M = A.shape
    chi_M_B, d_B, chi_R = B.shape
 
    if chi_M != chi_M_B:
        raise ValueError(
            f"Bond-dimension mismatch: A has χ_M={chi_M}, "
            f"B has χ_M={chi_M_B}."
        )
    if d != d_B or U.shape != (d, d, d, d):
        raise ValueError(
            f"Physical-dimension mismatch: A (d={d}), B (d={d_B}), "
            f"U shape={U.shape}."
        )
    if lam_l.shape != (chi_L,):
        raise ValueError(f"lam_l must have shape ({chi_L},), got {lam_l.shape}.")
    if lam_m.shape != (chi_M,):
        raise ValueError(f"lam_m must have shape ({chi_M},), got {lam_m.shape}.")
    if lam_r.shape != (chi_R,):
        raise ValueError(f"lam_r must have shape ({chi_R},), got {lam_r.shape}.")
 
    # ------------------------------------------------------------------
    # Step 1 — Build the two-site tensor  M = Γ[l] · Λ[l+1] · Γ[l+1]
    #
    #   Absorb Λ[l+1] into B before contracting:
    #     B_scaled[i, σ, j] = lam_m[i] · B[i, σ, j]
    #   Then contract over the shared bond index χ_M:
    #     M[α, σ, τ, γ] = Σ_i  A[α, σ, i] · B_scaled[i, τ, γ]
    # ------------------------------------------------------------------
    B_scaled: npt.NDArray = lam_m[:, np.newaxis, np.newaxis] * B           # (χ_M, d, χ_R)
    M: npt.NDArray = np.tensordot(A, B_scaled, axes=([2], [0]))             # (χ_L, d, d, χ_R)
 
    # ------------------------------------------------------------------
    # Step 2 — Apply gate U to the physical indices of M
    #
    #   Reshape M → (d², χ_L·χ_R) and U → (d², d²), then:
    #     M_U[σ'τ', αγ] = Σ_{στ} U[σ'τ', στ] · M[στ, αγ]
    #   Restore shape → (χ_L, d, d, χ_R).
    # ------------------------------------------------------------------
    M_mat = M.transpose(1, 2, 0, 3).reshape(d * d, chi_L * chi_R)
    U_mat = U.reshape(d * d, d * d)
    M = (U_mat @ M_mat).reshape(d, d, chi_L, chi_R).transpose(2, 0, 1, 3)
    # M shape after gate: (χ_L, d, d, χ_R)
 
    # ------------------------------------------------------------------
    # Step 3 — Absorb outer Schmidt vectors to form Θ
    #
    #   Θ[α, σ', τ', γ] = Λ[l][α] · M_U[α, σ', τ', γ] · Λ[l+2][γ]
    # ------------------------------------------------------------------
    Theta: npt.NDArray = lam_l[:, np.newaxis, np.newaxis, np.newaxis] * M
    Theta: npt.NDArray = Theta * lam_r[np.newaxis, np.newaxis, np.newaxis, :]
    # Theta shape: (χ_L, d, d, χ_R)
 
    # ------------------------------------------------------------------
    # Step 4 — SVD
    #
    #   Reshape Θ[α, σ', τ', γ] → (χ_L·d,  d·χ_R)  for the SVD.
    #   scipy returns S in descending order; no argsort is needed.
    # ------------------------------------------------------------------
    Theta_mat = Theta.reshape(chi_L * d, d * chi_R)
    U_svd, S, Vh_svd = la_svd(Theta_mat, full_matrices=False, lapack_driver="gesdd")
 
    # ------------------------------------------------------------------
    # Step 5 — Truncation
    #
    #   Keep at most chi_max singular values that are above the zero
    #   threshold.  The guard max(1, ...) prevents a zero-dimensional
    #   bond when all values happen to be numerically zero (e.g. a
    #   product state on one bond after a projective-like gate).
    # ------------------------------------------------------------------
    n_significant = int(np.sum(S > _SVD_ZERO_THRESHOLD))
    chi_new = max(1, min(n_significant, chi_max))
 
    trunc_error = float(np.sum(S[chi_new:] ** 2))   # discarded weight (unnormalised)
 
    S      = S[:chi_new]
    U_svd  = U_svd[:, :chi_new]
    Vh_svd = Vh_svd[:chi_new, :]
 
    # Normalise so that Σ_k λ_k² = 1  (Vidal convention).
    norm = np.linalg.norm(S)
    lam_new = S / norm if norm > 0.0 else S
 
    # ------------------------------------------------------------------
    # Step 6 — Reconstruct Gamma tensors
    #
    #   Strip the outer lambdas to restore Γ-form:
    #     A_new[α, σ', k] = (Λ_l^{-1})[α] · U_svd[α·d, k]
    #     B_new[k, τ', γ] = Vh_svd[k, τ'·γ] · (Λ_r^{-1})[γ]
    # ------------------------------------------------------------------
    U_svd  = U_svd.reshape(chi_L, d, chi_new)
    Vh_svd = Vh_svd.reshape(chi_new, d, chi_R)
 
    A_new: npt.NDArray = safe_inv(lam_l)[:, np.newaxis, np.newaxis] * U_svd   # (χ_L, d, χ_new)
    B_new: npt.NDArray = Vh_svd * safe_inv(lam_r)[np.newaxis, np.newaxis, :]  # (χ_new, d, χ_R)
 
    return A_new, lam_new, B_new, trunc_error
 

# ---------------------------------- Single-parity sweep ------------------------------------------ #

def tebd_sweep(
    gammas: List[npt.NDArray[np.complexfloating]],
    lambdas: List[npt.NDArray[np.floating]],
    gates: Dict[str, npt.NDArray[np.complexfloating]],
    chi_max: int,
    parity: int = SweepParity.EVEN,
) -> FullSweepResult:
    """Perform a single-parity TEBD sweep with position-dependent gates.

    Applies two-site gates to either all even bonds ``(0,1), (2,3), …``
    or all odd bonds ``(1,2), (3,4), …`` depending on ``parity``.
    Because even and odd bonds do not share sites, all gates within one
    parity commute and can be applied independently.

    The gate applied to each bond is selected by position:

    - Bond 0       → ``gates["left"]``   (left edge bond)
    - Bond L-2     → ``gates["right"]``  (right edge bond)
    - All others   → ``gates["bulk"]``   (interior bonds)

    This correctly distributes the on-site magnetic field across bonds
    for open-boundary chains (see :func:`build_xxz_two_site`).

    To compose a full second-order Suzuki–Trotter time step of size dt,
    call this function three times::

        tebd_sweep(gammas, lambdas, half_gates, chi_max, parity=EVEN)
        tebd_sweep(gammas, lambdas, full_gates, chi_max, parity=ODD)
        tebd_sweep(gammas, lambdas, half_gates, chi_max, parity=EVEN)

    where ``half_gates`` and ``full_gates`` are built via
    :func:`build_gate_dict`.

    The site tensors and singular-value arrays are updated **in place**.

    Parameters
    ----------
    gammas : list of ndarray, length L
        Gamma tensors Γ^[i] in Vidal canonical form,
        each of shape ``(chi_left, d, chi_right)``.
    lambdas : list of ndarray, length L+1
        Schmidt vectors. ``lambdas[i]`` lives on the bond to the left
        of site i. Boundary vectors ``lambdas[0]`` and ``lambdas[L]``
        are trivial (value ``[1.0]``).
    gates : dict with keys ``"left"``, ``"bulk"``, ``"right"``
        Position-dependent two-site gates, each of shape ``(d, d, d, d)``.
        Build this dict with :func:`build_gate_dict`.
    chi_max : int
        Maximum bond dimension retained after SVD truncation.
    parity : int, optional
        ``SweepParity.EVEN`` (0, default) to update bonds 0, 2, 4, …
        ``SweepParity.ODD``  (1)          to update bonds 1, 3, 5, …

    Returns
    -------
    FullSweepResult
        Dataclass holding total truncation error, updated bond indices,
        per-bond errors, and the parity used.

    Raises
    ------
    ValueError
        If lengths are inconsistent, gates dict is missing required keys,
        any gate is not rank-4, parity is not 0 or 1, or chi_max < 1.

    Notes
    -----
    Mutations to *gammas* and *lambdas* are made in place so that the
    caller's references remain valid without copying the full MPS.
    """
    L = len(gammas)

    # -------------------------------- Validation ------------------------------------- #
    if len(lambdas) != L + 1:
        raise ValueError(
            f"Expected {L + 1} lambda vectors for {L} sites, got {len(lambdas)}."
        )

    required_keys = {"left", "bulk", "right"}
    missing = required_keys - gates.keys()
    if missing:
        raise ValueError(
            f"gates dict is missing required keys: {sorted(missing)}. "
            f"Expected all of {sorted(required_keys)}."
        )
    for key in required_keys:
        if gates[key].ndim != 4:
            raise ValueError(
                f"gates['{key}'] must be a rank-4 tensor with shape (d, d, d, d), "
                f"got shape {gates[key].shape}."
            )

    if parity not in (SweepParity.EVEN, SweepParity.ODD):
        raise ValueError(
            f"parity must be SweepParity.EVEN (0) or SweepParity.ODD (1), "
            f"got {parity!r}."
        )
    if chi_max < 1:
        raise ValueError(f"chi_max must be a positive integer, got {chi_max}.")

    # ---------------------------------- Sweep ------------------------------------------------- #
    result = FullSweepResult(parity_used=parity)
    parity_str = "even" if parity == SweepParity.EVEN else "odd"

    logger.debug(
        "Starting %s-bond sweep over %d sites (chi_max=%d).",
        parity_str, L, chi_max,
    )

    for bond in range(parity, L - 1, 2):
        left, right = bond, bond + 1

        # Select gate by bond position
        if bond == 0:
            current_gate = gates["left"]
        elif bond == L - 2:
            current_gate = gates["right"]
        else:
            current_gate = gates["bulk"]

        a_new, lam_new, b_new, trunc_err = apply_2site_gate(
            gammas[left],
            gammas[right],
            lambdas[left],       # Λ[bond]   — left of site l
            lambdas[right],      # Λ[bond+1] — middle bond
            lambdas[right + 1],  # Λ[bond+2] — right of site l+1
            current_gate,
            chi_max,
        )

        gammas[left]   = a_new
        gammas[right]  = b_new
        lambdas[right] = lam_new

        result.total_truncation_error += trunc_err
        result.updated_bonds.append(bond)
        result.per_bond_errors.append(trunc_err)
        logger.debug("  %s bond %d  trunc_err=%.4e", parity_str, bond, trunc_err)

    logger.debug("Sweep complete. %s", result)
    return result




