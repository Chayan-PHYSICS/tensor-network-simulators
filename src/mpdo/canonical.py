"""
Gauge Fixing and Canonical Form Sweeps for MPDO Simulation
==========================================================================
Implements left-to-right QR and right-to-left SVD sweeps across an MPDO
chain to enforce canonical form and enable globally optimal bond truncation.

Background
----------
A tensor network does not have a unique representation — any invertible
matrix X inserted between two sites leaves the physical state unchanged:

    T[k] ── T[k+1]   =   T[k]·X ── X⁻¹·T[k+1]

This gauge freedom means that a naive local SVD truncation (e.g. directly
after each two-qubit gate) is suboptimal: the singular values reflect the
current arbitrary gauge, not the true entanglement spectrum.

The canonical form removes this ambiguity in two passes:

    Pass 1 — QR sweep (left → right):
        Each tensor T[k] is QR-decomposed.  Q is left-orthogonal
        (Q†Q = I) and stays at site k; the triangular R is absorbed
        into site k+1.  After the sweep all sites except the last
        satisfy the left-orthogonality condition.

        and all "weight" is concentrated in the rightmost tensor.

    Pass 2 — SVD sweep (right → left):
        Starting from the rightmost tensor, each bond is SVD-truncated.
        Because the QR sweep has fixed the gauge, the singular values
        at each bond are now gauge-invariant Schmidt coefficients —
        discarding the smallest ones gives the globally minimal
        approximation error for a fixed bond dimension χ_max.

Both passes together cost O(N) in the number of sites — compared to
O(N²) for site-by-site canonicalization — without any loss of accuracy.

"""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr  as la_qr
from scipy.linalg import svd as la_svd

__all__ = ["qr_sweep_left", "svd_sweep_right"]


# Left-to-right QR sweep
def qr_sweep_left(mpdo: list[np.ndarray]) -> list[np.ndarray]:
    """
    Left-to-right QR sweep: bring all sites except the last into left-canonical form.

    After the sweep, every T[k] for k < N−1 satisfies: T[k]* · T[k]  =  I
    This is a prerequisite for the SVD sweep to be globally optimal.

    Parameters
    ----------
    mpdo : list of ndarray, each shape (chi_l, chi_r, d, kappa)
        MPDO chain of N site tensors.

    Returns
    -------
    list of ndarray
        In-place modified MPDO with left-orthogonal sites 0 … N−2.
    """
    N = len(mpdo)

    for k in range(N - 1):
        chi_l, chi_r, d, kap = mpdo[k].shape

        # Group (chi_l, d, kappa) as rows, chi_r as columns
        # Shape: (chi_l, chi_r, d, kap) → (chi_l·d·kap, chi_r)
        T_mat   = mpdo[k].transpose(0, 2, 3, 1).reshape(chi_l * d * kap, chi_r)

        # Q is left-orthogonal, R carries forward
        Q, R    = la_qr(T_mat, mode='economic')
        chi_new = Q.shape[1]

        # Restore left-orthogonal tensor to standard layout (chi_l, chi_new, d, kap)
        mpdo[k] = Q.reshape(chi_l, d, kap, chi_new).transpose(0, 3, 1, 2)

        # Absorb R into the left bond of the next site
        # R: (chi_new, chi_r),  mpdo[k+1]: (chi_r, chi_r_next, d, kap) → (chi_new, chi_r_next, d, kap)  
        mpdo[k + 1] = np.tensordot(R, mpdo[k + 1], axes=([1], [0]))

    return mpdo


# Right-to-left SVD sweep
def svd_sweep_right(mpdo: list[np.ndarray], chi_max: int) -> list[np.ndarray]:
    """
    Right-to-left SVD sweep: truncate every bond to chi_max with globally optimal accuracy.

    Must be called after ``qr_sweep_left``.  Because all left-side tensors are
    left-orthogonal, the singular values at each bond are gauge-independent
    Schmidt coefficients; discarding the smallest gives minimum global error.

    Parameters
    ----------
    mpdo : list of ndarray, each shape (chi_l, chi_r, d, kappa)
        MPDO chain — should be left-canonical from ``qr_sweep_left``.
    chi_max : int
        Maximum bond dimension χ after truncation.

    Returns
    -------
    list of ndarray
        In-place modified MPDO with bonds truncated to chi_max.
    """
    N = len(mpdo)

    for k in range(N - 1, 0, -1):
        chi_l, chi_r, d, kap = mpdo[k].shape

        # Group chi_l as rows, (chi_r, d, kappa) as columns
        # Shape: (chi_l, chi_r, d, kap) → (chi_l, chi_r·d·kap)
        T_mat = mpdo[k].reshape(chi_l, chi_r * d * kap)

        U_svd, S, Vh = la_svd(T_mat, full_matrices=False, lapack_driver='gesdd')

        chi_new = min(chi_max, len(S))
        U_svd   = U_svd[:, :chi_new]
        S       = S[:chi_new]
        Vh      = Vh[:chi_new, :]

        # Right tensor: restore standard layout (chi_new, chi_r, d, kap)
        mpdo[k] = Vh.reshape(chi_new, chi_r, d, kap)
        u_scaled    = U_svd * S[None, :]                              # broadcast S along rows
        absorbed    = np.tensordot(mpdo[k - 1], u_scaled, axes=([1], [0]))
        mpdo[k - 1] = absorbed.transpose(0, 3, 1, 2) # Restored layout: (chi_l_prev, chi_new, d_prev, kap_prev)  
        

    return mpdo
