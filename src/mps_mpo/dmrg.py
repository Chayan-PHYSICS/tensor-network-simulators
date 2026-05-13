"""
dmrg.py — Two-Site Finite DMRG
================================
Implements the finite-size, two-site Density Matrix Renormalization Group
(DMRG) algorithm for finding the ground state of a 1-D quantum lattice model
whose Hamiltonian is expressed as a Matrix Product Operator (MPO).

Algorithm outline
-----------------
1. Start from an MPS in (right-)canonical form with bond dimension χ = 1.
2. Pre-compute all right-environment blocks from the right boundary inward.
3. Alternate left-to-right and right-to-left half-sweeps:
   a. Merge adjacent MPS tensors into a two-site tensor Θ.
   b. Solve  H_eff |Θ⟩ = E |Θ⟩  via the Lanczos algorithm.
   c. Split the optimised Θ with SVD, truncating to at most χ_max singular
      values.
   d. Update the environment block that was just "consumed" by the gauge move.
4. Stop when the energy change between consecutive sweeps falls below a
   user-supplied tolerance.

Conventions
-----------
* MPS tensors   : shape  (χ_L, d, χ_R)   — bond, physical, bond
* MPO tensors   : shape  (w_L, w_R, d, d) — MPO-bond, MPO-bond, bra, ket
* Environments  : shape  (χ_bra, w, χ_ket)
* Site indexing : 1-based (site 1 … L); index 0 and L+1 hold boundary scalars.

References
----------
* Schollwöck, U. (2011). "The density-matrix renormalization group in the age
  of matrix product states." Annals of Physics, 326(1), 96-192.
* Hubig, C., McCulloch, I. P., & Schollwöck, U. (2015). "Strictly single-site
  DMRG algorithm with subspace expansion." Physical Review B, 91(15), 155115.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from scipy.linalg import svd as la_svd
from .solver import lanczos_solver

__all__ = [
    "apply_heff",
    "update_left_env",
    "update_right_env",
    "initialize_environments",
    "run_dmrg",
]

# -- constants --------------------------------------------------------------
#: Singular values smaller than this are treated as numerically zero during SVD
#: truncation.  Overridable at call-site via the ``svd_thresh`` parameter of
#: :func:`run_dmrg`.
_SVD_ZERO_THRESHOLD: float = 1e-12


# ------------------------
# Effective Hamiltonian
# ------------------------

def apply_heff(
    left_env:  np.ndarray,
    W1:        np.ndarray,
    W2:        np.ndarray,
    right_env: np.ndarray,
    theta:     np.ndarray,
) -> np.ndarray:
    """Apply the two-site effective Hamiltonian to a two-site wavefunction Θ.

    Contracts the left environment, two MPO tensors, and the right environment
    around Θ to compute H_eff |Θ⟩ without explicitly building the full matrix.
    This is the core ``matvec`` operation passed to the Lanczos eigensolver.

    Tensor index conventions
    ------------------------
    All MPS bond indices are written as χ; MPO bond indices as w.
    ```
    +-----------+------------------------------+
    | Tensor    |           Shape              |
    +===========+==============================+
    | left_env  | (χ_L_bra,  w_l,  χ_L_ket)    |
    | W1        | (w_l,  w_m,  d,  d)          |
    | W2        | (w_m,  w_r,  d,  d)          |
    | right_env | (χ_R_bra,  w_r,  χ_R_ket)    |
    | theta     | (χ_L,  d,  d,  χ_R)          |
    +-----------+------------------------------+
    ```
    Contraction diagram:

        (χ_L_bra)        (d_bra_i)     (d_bra_{i+1})       (χ_R_bra)
              ↑               ↑               ↑                ↑
         ┌────┴────┐     ┌────┴────┐     ┌────┴────┐      ┌────┴────┐
         │  L_env  │─w_l─│   W1    │─w_m─│   W2    │─w_r─ │   R_env │
         └────┬────┘     └────┬────┘     └────┬────┘      └────┬────┘
              ↓               ↓               ↓                ↓
        (χ_L_ket)        (d_ket_i)     (d_ket_{i+1})       (χ_R_ket)
              └───────────────┴────[Θ]────────┴─────────────────┘

    The contraction is performed in four sequential steps to avoid ever forming
    the full (χ d²)² effective-Hamiltonian matrix.  The dominant cost is
    O(d² w² χ²) per application.

    Parameters
    ----------
    left_env : ndarray, shape (χ_L_bra, w_l, χ_L_ket)
        Left environment block accumulated up to (but not including) site i.
    W1 : ndarray, shape (w_l, w_m, d, d)
        MPO tensor at site i.  Axis ordering: (w_left, w_right, d_bra, d_ket).
    W2 : ndarray, shape (w_m, w_r, d, d)
        MPO tensor at site i+1.
    right_env : ndarray, shape (χ_R_bra, w_r, χ_R_ket)
        Right environment block accumulated from (but not including) site i+2.
    theta : ndarray, shape (χ_L, d, d, χ_R)
        Two-site wavefunction tensor Θ to act on.

    Returns
    -------
    ndarray, shape (χ_L, d, d, χ_R)
        Result of H_eff |Θ⟩, in the same index order as *theta*.
    """
    # Step 1 — absorb left environment into Θ
    # (χ_L_bra, w_l, χ_L_ket) ⊗ (χ_L, d, d, χ_R) → (χ_L_bra, w_l, d, d, χ_R)
    T1 = np.tensordot(left_env, theta, axes=([2], [0]))

    # Step 2 — contract with W1 over (w_l, d_ket_i)
    # (w_l, w_m, d_bra, d_ket) ⊗ (χ_L_bra, w_l, d_ket, d, χ_R)
    #   → (w_m, d_bra_i, χ_L_bra, d_{i+1}, χ_R)
    T2 = np.tensordot(W1, T1, axes=([0, 3], [1, 2]))

    # Step 3 — contract with W2 over (w_m, d_ket_{i+1})
    # (w_m, w_r, d_bra, d_ket) ⊗ (w_m, d_bra_i, χ_L_bra, d_ket, χ_R)
    #   → (w_r, d_bra_{i+1}, d_bra_i, χ_L_bra, χ_R)
    T3 = np.tensordot(W2, T2, axes=([0, 3], [0, 3]))

    # Step 4 — absorb right environment, closing χ_R and w_r
    # (χ_R_bra, w_r, χ_R_ket) ⊗ (w_r, d_bra_{i+1}, d_bra_i, χ_L_bra, χ_R_ket)
    #   → (χ_R_bra, d_bra_{i+1}, d_bra_i, χ_L_bra)
    result = np.tensordot(right_env, T3, axes=([1, 2], [0, 4]))

    # Step 5 — restore standard (χ_L, d_i, d_{i+1}, χ_R) index ordering
    return result.transpose(3, 2, 1, 0)


# ----------------------------------------------
# Environment updates
# ----------------------------------------------

def update_left_env(
    left_prev: np.ndarray,
    A:         np.ndarray,
    W:         np.ndarray,
) -> np.ndarray:
    """Push the left environment one site to the right.

    Contracts the existing left-environment block with the (left-canonical) MPS
    site tensor *A* and the corresponding MPO tensor *W* to produce the left
    environment for the next site.

    Diagram::

        (χ_R_bra) ←──[ A* ]──── (χ_L_bra)
                         │
        (w_r)     ←──[ W  ]──── (w_l)
                         │
        (χ_R_ket) ←──[ A  ]──── (χ_L_ket)
                         ↑
                   (left_prev bra, mpo, ket)

    Parameters
    ----------
    left_prev : ndarray, shape (χ_L_bra, w_l, χ_L_ket)
        Current left environment, with the index order (bra, MPO, ket).
    A : ndarray, shape (χ_L, d, χ_R)
        Left-canonical MPS tensor at the current site.
        Axis ordering: (left bond, physical, right bond).
    W : ndarray, shape (w_l, w_r, d_bra, d_ket)
        MPO tensor at the current site.
        Axis ordering: (left MPO bond, right MPO bond, bra physical, ket physical).

    Returns
    -------
    ndarray, shape (χ_R_bra, w_r, χ_R_ket)
        Updated left environment for the next site, in (bra, MPO, ket) order.
    """
    # Step 1 — absorb ket layer A into the left environment
    # (χ_L_bra, w_L, χ_L_ket) ⊗ (χ_L, d, χ_R) → (χ_L_bra, w_L, d, χ_R)
    T1 = np.tensordot(left_prev, A, axes=([2], [0]))

    # Step 2 — contract with W over the shared w_L and d_ket indices
    # (χ_L_bra, w_L, d_ket, χ_R) ⊗ (w_L, w_R, d_bra, d_ket) → (χ_L_bra, χ_R, w_R, d_bra)
    T2 = np.tensordot(T1, W, axes=([1, 2], [0, 3]))

    # Step 3 — contract with bra layer A* over χ_L_bra and d_bra
    # (χ_L_bra, χ_R_ket, w_R, d_bra) ⊗ (χ_L, d, χ_R) → (χ_R_ket, w_R, χ_R_bra)
    L_new = np.tensordot(T2, A.conj(), axes=([0, 3], [0, 1]))

    # Step 4 — reorder to (bra, MPO, ket) convention
    return L_new.transpose(2, 1, 0)


def update_right_env(
    right_next: np.ndarray,
    B:          np.ndarray,
    W:          np.ndarray,
) -> np.ndarray:
    """Push the right environment one site to the left.

    Contracts the existing right-environment block with the (right-canonical) MPS
    site tensor *B* and the corresponding MPO tensor *W* to produce the right
    environment for the previous site.

    Diagram::

        (χ_L_bra) ──[ B* ]──→ (χ_R_bra)
                        │
        (w_l)     ──[ W  ]──→ (w_r)
                        │
        (χ_L_ket) ──[ B  ]──→ (χ_R_ket)
                               ↑
                       (right_next bra, mpo, ket)

    Parameters
    ----------
    right_next : ndarray, shape (χ_R_bra, w_r, χ_R_ket)
        Current right environment, with the index order (bra, MPO, ket).
    B : ndarray, shape (χ_L, d, χ_R)
        Right-canonical MPS tensor at the current site.
        Axis ordering: (left bond, physical, right bond).
    W : ndarray, shape (w_l, w_r, d_bra, d_ket)
        MPO tensor at the current site.
        Axis ordering: (left MPO bond, right MPO bond, bra physical, ket physical).

    Returns
    -------
    ndarray, shape (χ_L_bra, w_l, χ_L_ket)
        Updated right environment for the previous site, in (bra, MPO, ket) order.
    """
    # Step 1 — absorb ket layer B over the shared right virtual bond
    # (χ_R_bra, w_R, χ_R_ket) ⊗ (χ_L, d, χ_R) → (χ_R_bra, w_R, χ_L, d)
    T1 = np.tensordot(right_next, B, axes=([2], [2]))

    # Step 2 — contract with W over the shared w_R and d_ket indices
    # (χ_R_bra, w_R, χ_L, d_ket) ⊗ (w_L, w_R, d_bra, d_ket) → (χ_R_bra, χ_L, w_L, d_bra)
    T2 = np.tensordot(T1, W, axes=([1, 3], [1, 3]))

    # Step 3 — contract with bra layer B* over χ_R_bra and d_bra
    # (χ_R_bra, χ_L_ket, w_L, d_bra) ⊗ (χ_L, d, χ_R) → (χ_L_ket, w_L, χ_L_bra)
    R_new = np.tensordot(T2, B.conj(), axes=([0, 3], [2, 1]))

    # Step 4 — reorder to (bra, MPO, ket) convention
    return R_new.transpose(2, 1, 0)


# -------------------------------------------
# Initialisation
# -------------------------------------------

def initialize_environments(
    mps: List[np.ndarray],
    mpo: List[np.ndarray],
    L:   int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Build boundary environments and pre-compute all right-environment blocks.

    For open boundary conditions (OBC) the trivial boundaries are rank-3
    tensors of shape (1, 1, 1) with value 1.  Starting from the right boundary,
    this function sweeps leftward and caches every right-environment block so
    that the main DMRG loop can retrieve them in O(1) time.

    Parameters
    ----------
    mps : list of ndarray
        1-indexed MPS tensor list.  ``mps[0]`` is unused (``None``);
        ``mps[1]`` … ``mps[L]`` hold the physical site tensors, each with
        shape (χ_L, d, χ_R).
    mpo : list of ndarray
        1-indexed MPO tensor list with the same indexing convention as *mps*.
    L : int
        Number of physical sites.

    Returns
    -------
    left_envs : list of ndarray, length L+2
        ``left_envs[0]`` is the trivial left boundary ``(1,1,1)``.
        All other entries are initialised to ``None`` and filled in by the
        main sweep loop.
    right_envs : list of ndarray, length L+2
        ``right_envs[L+1]`` is the trivial right boundary ``(1,1,1)``.
        ``right_envs[i]`` for i = L down to 1 contains the right environment
        block for everything to the right of site i-1.

    Notes
    -----
    Both returned lists use 1-based site indexing to match *mps* and *mpo*.
    """
    # Trivial boundary scalars (shape (1,1,1)) for open boundary conditions
    left_envs  = [None] * (L + 2)
    right_envs = [None] * (L + 2)

    left_envs[0]    = np.ones((1, 1, 1), dtype=complex)
    right_envs[L+1] = np.ones((1, 1, 1), dtype=complex)

    # Sweep right-to-left, accumulating right environments
    for i in range(L, 0, -1):
        right_envs[i] = update_right_env(right_envs[i + 1], mps[i], mpo[i])

    return left_envs, right_envs


# -----------------------------------------
# Main DMRG driver
# -----------------------------------------
def run_dmrg(
    mps:        List[np.ndarray],
    mpo:        List[np.ndarray],
    left_envs:  List[np.ndarray],
    right_envs: List[np.ndarray],
    n_sweeps:   int,
    chi_max:    int,
    conv_tol:   float = 1e-12,
    svd_thresh: float = _SVD_ZERO_THRESHOLD,
    verbose:    bool  = True,
) -> float:
    """Run the two-site finite-DMRG optimisation loop.

    Performs alternating left-to-right and right-to-left half-sweeps.  At each
    step the two-site wavefunction Θ is optimised by solving the local
    eigenvalue problem  H_eff |Θ⟩ = E |Θ⟩  via :func:`lanczos_solver`, then
    split back into two site tensors using a truncated SVD.

    Sweep structure
    ---------------
    Left-to-right (sites i = 1 … L-1, optimising pair (i, i+1)):
      1. Contract mps[i] and mps[i+1] into Θ.
      2. Solve H_eff |Θ⟩ = E |Θ⟩  (Lanczos).
      3. SVD: Θ ≈ U · S · Vᴴ,  keep at most *chi_max* singular values.
      4. mps[i]   ← reshape(U)           (left-canonical)
         mps[i+1] ← reshape(diag(S) · Vᴴ) (carries singular values rightward)
      5. Update left_envs[i].

    Right-to-left (sites i = L-1 … 1, optimising pair (i, i+1)):
      Steps 1-3 identical.
      4. mps[i]   ← reshape(U · diag(S))  (carries singular values leftward)
         mps[i+1] ← reshape(Vᴴ)           (right-canonical)
      5. Update right_envs[i+1].

    Parameters
    ----------
    mps : list of ndarray
        1-indexed MPS tensors.  Modified **in-place** during the sweep.
    mpo : list of ndarray
        1-indexed MPO tensors (read-only).
    left_envs : list of ndarray
        Environment list returned by :func:`initialize_environments`.
        Modified **in-place**.
    right_envs : list of ndarray
        Environment list returned by :func:`initialize_environments`.
        Modified **in-place**.
    n_sweeps : int
        Maximum number of full (left + right) sweeps.
    chi_max : int
        Maximum bond dimension; singular values beyond this rank are discarded.
    conv_tol : float, optional
        Stop early when  |E_sweep - E_prev| < conv_tol.  Default 1e-12.
    svd_thresh : float, optional
        Singular values below this threshold are treated as zero before
        applying the *chi_max* cutoff.  Default :data:`_SVD_ZERO_THRESHOLD`.
    verbose : bool, optional
        Print per-sweep energy and convergence messages.  Default ``True``.

    Returns
    -------
    float
        Ground-state energy estimate after the final sweep.

    Notes
    -----
    The energy reported (and returned) is the Lanczos eigenvalue from the
    **last** two-site window optimised in the final half-sweep.  For a
    converged run this equals the full expectation value ⟨ψ|H|ψ⟩ to within
    *conv_tol*.
    """
    L      = len(mps) - 1  # sites are indexed 1 … L
    E_prev = np.inf
    energy = 0.0

    for sweep in range(n_sweeps):

        # -- Left-to-right half-sweep --------------------------------
        for i in range(1, L):
            # 1. Two-site wavefunction
            theta = np.tensordot(mps[i], mps[i + 1], axes=([2], [0]))

            # 2. Local eigenvalue problem
            energy, theta_opt = lanczos_solver(
                apply_heff,
                left_envs[i - 1], mpo[i], mpo[i + 1], right_envs[i + 2],
                theta, k_max=20, tol=1e-12,
            )

            # 3. Truncated SVD — gauge moves right
            chi_L, d1, d2, chi_R = theta_opt.shape
            M         = np.reshape(theta_opt, (chi_L * d1, d2 * chi_R))
            U, S, Vh  = la_svd(M, full_matrices=False, lapack_driver="gesdd")

            n_sig   = int(np.sum(S > svd_thresh))
            chi_new = max(1, min(n_sig, chi_max))
            U, S, Vh = U[:, :chi_new], S[:chi_new], Vh[:chi_new, :]

            # 4. Update MPS tensors
            mps[i]     = np.reshape(U, (chi_L, d1, chi_new))           # left-canonical
            mps[i + 1] = np.reshape(np.diag(S) @ Vh, (chi_new, d2, chi_R))

            # 5. Push left environment one site to the right
            left_envs[i] = update_left_env(left_envs[i - 1], mps[i], mpo[i])

        # -- Right-to-left half-sweep ----------------------------------------------
        for i in range(L - 1, 0, -1):
            # 1. Two-site wavefunction
            theta = np.tensordot(mps[i], mps[i + 1], axes=([2], [0]))

            # 2. Local eigenvalue problem
            energy, theta_opt = lanczos_solver(
                apply_heff,
                left_envs[i - 1], mpo[i], mpo[i + 1], right_envs[i + 2],
                theta, k_max=20, tol=1e-12,
            )

            # 3. Truncated SVD — gauge moves left
            chi_L, d1, d2, chi_R = theta_opt.shape
            M         = np.reshape(theta_opt, (chi_L * d1, d2 * chi_R))
            U, S, Vh  = la_svd(M, full_matrices=False, lapack_driver="gesdd")

            n_sig   = int(np.sum(S > svd_thresh))
            chi_new = max(1, min(n_sig, chi_max))
            U, S, Vh = U[:, :chi_new], S[:chi_new], Vh[:chi_new, :]

            # 4. Update MPS tensors
            mps[i]     = np.reshape(U @ np.diag(S), (chi_L, d1, chi_new))
            mps[i + 1] = np.reshape(Vh, (chi_new, d2, chi_R))           # right-canonical

            # 5. Push right environment one site to the left
            right_envs[i + 1] = update_right_env(
                right_envs[i + 2], mps[i + 1], mpo[i + 1]
            )

        # -- Convergence check and reporting ---------------------------------------
        if verbose:
            print(f"Sweep {sweep:3d} │ E = {energy:+.12f} │ ΔE = {abs(energy - E_prev):.3e}")

        if abs(energy - E_prev) < conv_tol:
            if verbose:
                print(f"Converged after {sweep + 1} sweep(s)  (ΔE < {conv_tol:.1e})")
            break

        E_prev = energy

    return energy