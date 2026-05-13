"""
DMRG Iterative Solvers Module
=============================
This module contains eigenvalue solvers tailored for Density Matrix Renormalization 
Group (DMRG) applications. The primary focus is on Krylov subspace methods like 
the Lanczos algorithm, used to find the ground state of the effective Hamiltonian.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Callable

__all__ = ["lanczos_solver"]

def lanczos_solver(
    heff_func: Callable,
    left_env: np.ndarray, 
    W_1: np.ndarray, 
    W_2: np.ndarray, 
    right_env: np.ndarray, 
    theta_guess: np.ndarray, 
    k_max: int = 20, 
    tol: float = 1e-12
) -> Tuple[float, np.ndarray]:
    """
    Finds the ground state of the effective Hamiltonian using the Lanczos algorithm.
    
    Args:
        heff_func: The function that applies the Hamiltonian (apply_heff).
        left_env: Left environment tensor (chi_L, w_L, chi_L).
        W_1: First MPO tensor (w_L, w_m, d, d).
        W_2: Second MPO tensor (w_m, w_R, d, d).
        right_env: Right environment tensor (chi_R, w_R, chi_R).
        theta_guess: Initial guess for the two-site tensor (chi_L, d, d, chi_R).
        k_max: Maximum number of Lanczos iterations.
        tol: Tolerance for convergence based on the residual norm.
    """
    # NO IMPORTS FROM DMRG HERE
    
    theta_shape = theta_guess.shape
    v_flat = theta_guess.flatten() / np.linalg.norm(theta_guess)

    # Storage for Krylov basis and tridiagonal elements
    V = [v_flat]
    alphas = []  # Diagonal elements
    betas = []   # Off-diagonal elements
    v_prev = None

    for j in range(k_max):
        # 1. Apply Effective Hamiltonian
        # We call the function passed as an argument
        H_v = heff_func(
            left_env, W_1, W_2, right_env, 
            v_flat.reshape(theta_shape)
        )
        H_v_flat = H_v.flatten()

        # 2. Calculate alpha (Rayleigh quotient)
        alpha = np.vdot(v_flat, H_v_flat)
        alphas.append(alpha.real)

        # 3. Calculate residual vector
        w = H_v_flat - alpha * v_flat
        if v_prev is not None:
            w -= betas[-1] * v_prev
        
        # 4. Full Re-orthogonalization (Gram-Schmidt)
        for previous_v in V:
            w -= np.vdot(previous_v, w) * previous_v
        
        # 5. Calculate beta and check for early convergence
        beta = np.linalg.norm(w) # Calculate beta: b_j = ||w||
        if beta < tol or j == k_max - 1:
            # If we hit the end or converge, DON'T add this beta 
            # because we don't have a next vector for it.
            break
        # if beta < tol:
        #     break

        betas.append(beta)
        v_prev = v_flat
        v_flat = w / beta
        V.append(v_flat)

    # 6. Construct the Tridiagonal Matrix T
    m = len(alphas)
    T = np.diag(alphas)
    if m > 1:
        T += np.diag(betas, k=1) + np.diag(betas, k=-1)

    # 7. Diagonalize the reduced Hamiltonian
    evals, evecs = np.linalg.eigh(T)
    E0 = evals[0]

    # 8. Reconstruct the optimized tensor
    optimal_coeffs = evecs[:, 0]
    ground_state_flat = np.dot(optimal_coeffs, np.array(V))

    theta_optimized = ground_state_flat.reshape(theta_shape)
    
    return E0, theta_optimized





# """
# DMRG Iterative Solvers Module
# =============================
# This module contains eigenvalue solvers tailored for Density Matrix Renormalization 
# Group (DMRG) applications. The primary focus is on Krylov subspace methods like 
# the Lanczos algorithm, used to find the ground state of the effective Hamiltonian.
# """
# from __future__ import annotations
# import numpy as np
# from typing import Tuple


# __all__ = ["lanczos_solver"]

# def lanczos_solver(
#     left_env: np.ndarray, 
#     W_1: np.ndarray, 
#     W_2: np.ndarray, 
#     right_env: np.ndarray, 
#     theta_guess: np.ndarray, 
#     k_max: int = 20, 
#     tol: float = 1e-12
# ) -> Tuple[float, np.ndarray]:
#     """
#     Finds the ground state of the effective Hamiltonian using the Lanczos algorithm.
    
#     This function iteratively builds a Krylov subspace and solves the eigenvalue 
#     problem within that reduced basis to approximate the global ground state.

#     Args:
#         left_env: Left environment tensor (chi_L, w_L, chi_L).
#         W_1: First MPO tensor (w_L, w_m, d, d).
#         W_2: Second MPO tensor (w_m, w_R, d, d).
#         right_env: Right environment tensor (chi_R, w_R, chi_R).
#         theta_guess: Initial guess for the two-site tensor (chi_L, d, d, chi_R).
#         k_max: Maximum number of Lanczos iterations.
#         tol: Tolerance for convergence based on the residual norm.

#     Returns:
#         E0: The approximate ground state energy (lowest eigenvalue).
#         theta_optimized: The optimized two-site tensor in its original shape.
#     """
#     # Import inside to maintain file independence if needed, 
#     # though usually assumed to be in global scope.
#     from dmrg import apply_Heff 

#     theta_shape = theta_guess.shape
#     v = theta_guess / np.linalg.norm(theta_guess)
#     v_flat = v.flatten()

#     # Storage for Krylov basis and tridiagonal elements
#     V = [v_flat]
#     alphas = []  # Diagonal elements
#     betas = []   # Off-diagonal elements
#     v_prev = None

#     for j in range(k_max):
#         # 1. Apply Effective Hamiltonian
#         # Reshape v_flat back to 4D for the contraction logic
#         H_v = apply_Heff(left_env, W_1, W_2, right_env, v_flat.reshape(theta_shape))
#         H_v_flat = H_v.flatten()

#         # 2. Calculate alpha (Rayleigh quotient)
#         alpha = np.vdot(v_flat, H_v_flat)
#         alphas.append(alpha.real)

#         # 3. Calculate residual vector
#         # w = H|v_j> - alpha_j|v_j> - beta_{j-1}|v_{j-1}>
#         w = H_v_flat - alpha * v_flat
#         if v_prev is not None:
#             w -= betas[-1] * v_prev
        
#         # 4. Full Re-orthogonalization (Gram-Schmidt)
#         # Essential to prevent loss of orthogonality due to finite precision
#         for previous_v in V:
#             w -= np.vdot(previous_v, w) * previous_v
        
#         # 5. Calculate beta and check for early convergence
#         beta = np.linalg.norm(w)
#         if beta < tol:
#             break

#         betas.append(beta)
#         v_prev = v_flat
#         v_flat = w / beta
#         V.append(v_flat)

#     # 6. Construct the Tridiagonal Matrix T
#     m = len(alphas)
#     T = np.diag(alphas)
#     if m > 1:
#         T += np.diag(betas, k=1) + np.diag(betas, k=-1)

#     # 7. Diagonalize the reduced Hamiltonian
#     evals, evecs = np.linalg.eigh(T)
    
#     # Lowest energy is the first eigenvalue
#     E0 = evals[0]

#     # 8. Reconstruct the optimized tensor in the original basis
#     # ground_state = sum(optimal_coeffs_i * V_i)
#     optimal_coeffs = evecs[:, 0]
#     ground_state_flat = np.dot(optimal_coeffs, np.array(V))

#     theta_optimized = ground_state_flat.reshape(theta_shape)
    
#     return E0, theta_optimized
