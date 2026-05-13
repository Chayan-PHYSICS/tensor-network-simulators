"""
Local Quantum Gate Operations for MPDO Simulation
=============================================================
Implements single-qubit and two-qubit unitary gate contractions on
Matrix Product Density Operator (MPDO) site tensors.

All tensors follow the axis convention:
    T[k] : (chi_l,  chi_r,  d,        kappa)
            -----   -----   -         -----
            left    right   physical  inner
            bond    bond    dim       dim (noise)

Two-qubit gate matrices are accepted in flat (4, 4) or tensor (d, d, d, d)
form and are internally reshaped to (out_q1, out_q2, in_q1, in_q2).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import svd as la_svd

__all__ = ["apply1_qubit_gate", "apply2_qubit_gate"]


# Single-qubit gate
def apply1_qubit_gate(T: np.ndarray, G: np.ndarray) -> np.ndarray:
    """
    Apply a single-qubit unitary gate to one MPDO site tensor.

    Contracts G over the physical index of T and restores the standard axis layout. No bond or inner index is affected.
    
    Parameters
    ----------
    T : ndarray, shape (chi_l, chi_r, d, kappa)
        Target site tensor.
    G : ndarray, shape (d, d)
        Unitary gate matrix.

    Returns
    -------
    ndarray, Gate-evolved site tensor with shape (chi_l, chi_r, d, kappa) 
        
    """
    # Contract G[d_out, d_in] with T[chi_l, chi_r, d_in, kappa] → (d_out, chi_l, chi_r, kappa)
    result = np.tensordot(G, T, axes=([1], [2]))

    # Restore standard layout: (chi_l, chi_r, d_out, kappa)
    return result.transpose(1, 2, 0, 3)

# Two-qubit gate
def apply2_qubit_gate(
    Tk:      np.ndarray,
    Tk1:     np.ndarray,
    U:       np.ndarray,
    chi_max: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Apply a two-qubit unitary gate across adjacent MPDO site tensors k and k+1.

    Procedure
    ---------
    1. Contract the shared virtual bond between T[k] and T[k+1] to form the two-site tensor Θ.
    2. Apply gate U by contracting its input legs with Θ's physical indices.
    3. Rearrange indices into a bipartite (left | right) layout.
    4. Reshape to a matrix and compute SVD & truncate to chi_max singular values(rough local cut; globally optimal
       truncation is done by canonical sweep after each full layer).
    6. Reconstruct the two site tensors T'[k] and T'[k+1].

    Parameters
    ----------
    Tk : ndarray, shape (chi_l, chi_mid, d, kappa_k)
        Site tensor at position k.
    Tk1 : ndarray, shape (chi_mid, chi_r, d, kappa_k1)
        Site tensor at position k+1.
    U : ndarray, shape (4, 4) or (d, d, d, d)
        Two-qubit unitary.
    chi_max : int
        Maximum bond dimension after truncation.

    Returns
    -------
    Tk_new : ndarray, shape (chi_l, chi_new, d, kappa_k)
    Tk1_new : ndarray, shape (chi_new, chi_r, d, kappa_k1)
    trunc_err : float
        Relative truncation error = Σ_{i>chi_new} s_i² / Σ_i s_i².
    """
    chi_l  = Tk.shape[0]
    chi_r  = Tk1.shape[1]
    d      = Tk.shape[2]
    kap_k  = Tk.shape[3]
    kap_k1 = Tk1.shape[3]

    # Reshape gate to 4-index tensor (out_q1, out_q2, in_q1, in_q2)
    U_tensor = np.asarray(U, dtype=complex).reshape(d, d, d, d)

    # --------- Step 1: Contract shared bond (chi_mid) --------------------------
    # Tk  : (chi_l, chi_mid, d_k,  kappa_k)
    # Tk1 : (chi_mid, chi_r, d_k1, kappa_k1)
    # Theta axes: (chi_l, d_k, kappa_k, chi_r, d_k1, kappa_k1)
    Theta = np.tensordot(Tk, Tk1, axes=([1], [0]))

    # -------- Step 2: Apply gate ----------------------------------------------
    # Contract  U[*, *, in_q1, in_q2] & Theta[*, d_k, *, *, d_k1, *] → (out_q1, out_q2, chi_l, kappa_k, chi_r, kappa_k1)
    W = np.tensordot(U_tensor, Theta, axes=([2, 3], [4, 1]))

    # ------- Step 3: Bipartite reordering --------------------------------------
    # source axes: (out_q1, out_q2, chi_l, kappa_k, chi_r, kappa_k1)
    W = W.transpose(2, 1, 3, 4, 0, 5)
    # W axes now: (chi_l, d_k, kappa_k, chi_r, d_k1, kappa_k1)

    # ------- Step 4: SVD + truncation ------------------------------------------
    W_mat = W.reshape(chi_l * d * kap_k,
                      chi_r * d * kap_k1)
    # U_svd, S, Vh = la_svd(W_mat, full_matrices=False, lapack_driver='gesdd')
    try:
        U_svd, S, Vh = la_svd(W_mat, full_matrices=False, lapack_driver='gesdd')
    except np.linalg.LinAlgError:
        U_svd, S, Vh = la_svd(W_mat, full_matrices=False, lapack_driver='gesvd')

    chi_new   = min(chi_max, len(S))
    s_sq      = S ** 2
    trunc_err = (float(np.sum(s_sq[chi_new:]) / np.sum(s_sq))
                 if chi_new < len(S) else 0.0)

    U_svd = U_svd[:, :chi_new]
    S     = S[:chi_new]
    Vh    = Vh[:chi_new, :]

    # ------- Step 5: Reconstruct site tensors ---------------------------------
    # Absorb singular values into the left tensor → (chi_l, d_k, kappa_k, chi_new)
    Tk_new  = (U_svd * S[None, :]).reshape(chi_l, d, kap_k, chi_new).transpose(0, 3, 1, 2)                      

    # Vh: (chi_new, chi_r, d_k1, kappa_k1) → (chi_new, chi_r, d, kappa_k1)
    Tk1_new = Vh.reshape(chi_new, chi_r, d, kap_k1) 
    return Tk_new, Tk1_new, trunc_err

