# observables.py
"""
Observables and diagnostics for 1-D MPS in Vidal (Γ–Λ) canonical form.

Conventions
-----------
- Site tensors  A_list[i]  have shape  (χ_L, d, χ_R).
- Bond tensors  lam_list[i] are 1-D arrays of Schmidt values on bond i,
  where bond i sits to the LEFT of site i  (i.e. between sites i-1 and i).
- Two-site reduced density matrices are formed using lam_list[i] as the
  left boundary weight, giving the correct norm in Vidal gauge.
- h_bond is expected in the reshaped rank-4 form  (d, d, d, d)
  with index order  (s', t', s, t).
"""

from __future__ import annotations
from typing import List
import numpy as np

# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------
def compute_energy(
    A_list: List[np.ndarray],
    lam_list: List[np.ndarray],
    h_left: np.ndarray,
    h_bulk: np.ndarray,
    h_right: np.ndarray,
) -> float:
    """Compute the total energy expectation value ⟨H⟩ of the MPS.

    Evaluates the sum of local two-site expectation values:

        E = Σ_i ⟨Θ_i | h_i | Θ_i⟩

    where the normalised two-site tensor at bond (i, i+1) is:

        Θ[i] = Λ[i] · Γ[i] · Λ[i+1] · Γ[i+1] · Λ[i+2]

    The operator h_i acts on the physical indices of Θ[i] only.
    The outer Schmidt vectors Λ[i] and Λ[i+2] live on virtual indices
    and are needed for correct normalisation — they do not affect how
    h_i is applied.

    Parameters
    ----------
    A_list : list of ndarray, length L
        Gamma tensors Γ^[i] in Vidal canonical form,
        each of shape ``(χ_L, d, χ_R)``.
    lam_list : list of ndarray, length L+1
        Schmidt vectors. ``lam_list[i]`` lives on the bond to the
        left of site i.
    h_left : ndarray, shape (d, d, d, d)
        Bond Hamiltonian for the left edge bond (sites 0 and 1),
        index order ``(s', t', s, t)``.
    h_bulk : ndarray, shape (d, d, d, d)
        Bond Hamiltonian for all interior bonds,
        index order ``(s', t', s, t)``.
    h_right : ndarray, shape (d, d, d, d)
        Bond Hamiltonian for the right edge bond (sites L-2 and L-1),
        index order ``(s', t', s, t)``.

    Returns
    -------
    float
        Real part of ⟨H⟩. A large imaginary residual indicates loss
        of canonical form.

    Examples
    --------
    >>> h_left, h_bulk, h_right = build_xxz_two_site(Jx, Jy, Jz, hz)
    >>> E = compute_energy(
    ...     A_list, lam_list,
    ...     h_left.reshape(d,d,d,d),
    ...     h_bulk.reshape(d,d,d,d),
    ...     h_right.reshape(d,d,d,d),
    ... )
    >>> print(f"Energy per site: {E / n_sites:.6f}")
    """
    L      = len(A_list)
    energy = 0.0

    for i in range(L - 1):

        # ── Select bond Hamiltonian by position ───────────────────────────── #
        if i == 0:
            h_bond = h_left
        elif i == L - 2:
            h_bond = h_right
        else:
            h_bond = h_bulk

        # ── Step 1 — Build M = Γ[i] · Λ[i+1] · Γ[i+1] ───────────────────── #
        #   Absorb the middle Schmidt vector into the right site tensor,
        #   then contract over the shared bond index.
        A_lam = np.tensordot(
            A_list[i], np.diag(lam_list[i + 1]), axes=([2], [0])
        )                                                                  # (χ_L, d, χ_M)
        M = np.tensordot(A_lam, A_list[i + 1], axes=([2], [0]))          # (χ_L, d, d, χ_R)

        # ── Step 2 — Absorb outer lambdas to form normalised Θ ────────────── #
        #   Θ = Λ[i] · M · Λ[i+2]
        #   Λ[i] and Λ[i+2] act on virtual indices only — they do not
        #   interfere with the physical-index operator h_bond.
        Theta = lam_list[i][:, np.newaxis, np.newaxis, np.newaxis] * M
        Theta = Theta * lam_list[i + 2][np.newaxis, np.newaxis, np.newaxis, :]
        # Theta shape: (χ_L, d, d, χ_R)

        # ── Step 3 — Apply h_bond on physical indices of Θ ───────────────── #
        #   h[s', t', s, t] · Θ[α, s, t, γ] → (d, d, χ_L, χ_R)
        h_Theta = np.tensordot(h_bond, Theta, axes=([2, 3], [1, 2]))     # (d, d, χ_L, χ_R)
        h_Theta = h_Theta.transpose(2, 0, 1, 3)                          # (χ_L, d, d, χ_R)

        # ── Step 4 — Local expectation value ⟨Θ | h | Θ⟩ ─────────────────── #
        local_energy = np.tensordot(
            Theta.conj(), h_Theta,
            axes=([0, 1, 2, 3], [0, 1, 2, 3]),
        )
        energy += float(np.real(local_energy))

    return energy



def compute_energy_old(
    A_list: List[np.ndarray],
    lam_list: List[np.ndarray],
    h_bond: np.ndarray,
) -> float:
    """Compute the ground-state energy expectation value <H>.

    Parameters
    ----------
    A_list:
        Length-N list of site tensors, each of shape (χ_L, d, χ_R).
    lam_list:
        Length-(N+1) list of 1-D Schmidt-value arrays.
        lam_list[i] lives on the bond to the LEFT of site i.
    h_bond:
        Two-site Hamiltonian of shape (d, d, d, d) with index order (s', t', s, t).

    Returns
    -------
    float
        Real part of <H>. A large imaginary residual indicates loss of canonical form.

    Parameters
    ----------
    A_list:
        Length-*N* list of site tensors, each of shape ``(χ_L, d, χ_R)``.
    lam_list:
        Length-*(N+1)* list of 1-D Schmidt-value arrays.
        ``lam_list[i]`` lives on the bond to the **left** of site *i*.
    h_bond:
        Two-site Hamiltonian of shape ``(d, d, d, d)`` with index order
        ``(s', t', s, t)``.

    Returns
    -------
    float
        Real part of ⟨H⟩.  The imaginary part is discarded; a large
        imaginary residual indicates a loss of canonical form.

    Notes
    -----
    The function assumes the MPS is in Vidal canonical form so that
    contracting with ``lam_list[i]`` on the left boundary is sufficient
    to produce a properly normalised two-site reduced state.

    Examples
    --------
    >>> E = compute_energy(A_list, lam_list, h_bond)
    >>> print(f"Energy per site: {E / n_sites:.6f}")
    """
    n_sites = len(A_list)
    energy = 0.0

    for i in range(n_sites - 1):
        # ------------------------------------------------------------------
        # Step 1 – Build the two-site tensor Θ[α, s, t, γ]
        #
        #   First contract neighbouring site tensors along their shared bond:
        #     A[i]  (χ_L, d, χ_mid)  ⊗  A[i+1]  (χ_mid, d, χ_R)
        #   → C  (χ_L, d, d, χ_R)
        #
        #   Then weight the left bond with Λ[i] to enter the Vidal gauge:
        #     Λ[i] (χ_L,)  ·  C  (χ_L, d, d, χ_R)
        #   → Θ  (χ_L, d, d, χ_R)
        # ------------------------------------------------------------------
        C = np.tensordot(A_list[i], A_list[i + 1], axes=(2, 0))          # (χ_L, d, d, χ_R)
        Theta = np.tensordot(np.diag(lam_list[i]), C, axes=(1, 0))       # (χ_L, d, d, χ_R)

        # ------------------------------------------------------------------
        # Step 2 – Apply h_bond on physical indices
        #
        #   h4[s', t', s, t]  ·  Θ[α, s, t, γ]
        #   → (s', t', α, γ)  → transpose →  (α, s', t', γ)
        # ------------------------------------------------------------------
        h_Theta = np.tensordot(h_bond, Theta, axes=([2, 3], [1, 2]))     # (d, d, χ_L, χ_R)
        h_Theta = h_Theta.transpose(2, 0, 1, 3)                          # (χ_L, d, d, χ_R)

        # ------------------------------------------------------------------
        # Step 3 – Evaluate the local expectation value
        #
        #   ⟨h⟩_i = Θ*[α, s, t, γ]  ·  (h·Θ)[α, s, t, γ]
        # ------------------------------------------------------------------
        local_energy = np.tensordot(
            Theta.conj(), h_Theta,
            axes=([0, 1, 2, 3], [0, 1, 2, 3]),
        )
        energy += float(np.real(local_energy))

    return energy



# ---------------------------------------------------------------------------
# Bond dimension
# ---------------------------------------------------------------------------

def bond_dimensions(lam_list: List[np.ndarray]) -> List[int]:
    """Return the bond dimension at each bond of the chain.

    The bond dimension at bond *i* is the number of non-negligible Schmidt
    values retained in ``lam_list[i]``.

    Parameters
    ----------
    lam_list:
        Length-*(N+1)* list of 1-D Schmidt-value arrays.
        Boundary bonds (i=0 and i=N) are trivial and return ``1``.

    Returns
    -------
    list[int]
        Length-*(N+1)* list of bond dimensions ``χ[i] = len(lam_list[i])``.

    Examples
    --------
    >>> chi = bond_dimensions(lam_list)
    >>> print("Max bond dim:", max(chi))
    """
    return [len(lam) for lam in lam_list]


def max_bond_dimension(lam_list: List[np.ndarray]) -> int:
    """Return the maximum bond dimension across the chain.

    Parameters
    ----------
    lam_list:
        Length-*(N+1)* list of 1-D Schmidt-value arrays.

    Returns
    -------
    int
        Largest bond dimension present in the MPS.
    """
    return max(bond_dimensions(lam_list))


# ---------------------------------------------------------------------------
# Entanglement entropy

def entanglement_entropy_bond(
    lam: np.ndarray,
    renyi_index: float = 1.0,
) -> float:
    """Compute the Rényi entanglement entropy for a single bond.

    S_n = (1 / 1-n) · ln Σ λ_α^(2n)

    Special cases:
    - n = 1   (von Neumann) : S = -Σ λ_α² · ln λ_α²
    - n = inf (single-copy) : S = -ln λ_max²

    Parameters
    ----------
    lam:
        1-D array of Schmidt values, normalised so Σ λ_α² = 1.
    renyi_index:
        Rényi index n. Use 1.0 for von Neumann, np.inf for single-copy.

    Returns
    -------
    float
        Entanglement entropy in nats.
    """
    p = lam ** 2
    p = p[p > 1e-15]   # suppress numerical zeros before log

    if renyi_index == 1.0:
        return float(-np.sum(p * np.log(p)))
    elif renyi_index == np.inf:
        return float(-np.log(np.max(p)))
    else:
        return float(np.log(np.sum(p ** renyi_index)) / (1.0 - renyi_index))


def entanglement_entropy_profile(
    lam_list: List[np.ndarray],
    renyi_index: float = 1.0,
) -> List[float]:
    """Compute the Rényi entanglement entropy at every bond of the chain.

    This is a convenience wrapper that calls :func:`entanglement_entropy_bond`
    on each entry of *lam_list*, giving the full spatial entropy profile in
    one call.  Useful for visualising the entanglement structure of the MPS
    at a fixed point in time.

    Parameters
    ----------
    lam_list:
        Length-*(N+1)* list of 1-D Schmidt-value arrays.
    renyi_index:
        Rényi index passed through to :func:`entanglement_entropy_bond`.

    Returns
    -------
    list[float]
        Length-*(N+1)* list of entanglement entropies, one per bond.
    """
    return [entanglement_entropy_bond(lam, renyi_index) for lam in lam_list]


