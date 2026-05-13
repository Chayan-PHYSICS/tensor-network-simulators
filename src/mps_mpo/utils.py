# def is_converged(history: list, tol: float = 1e-10, window: int = 5) -> bool:
#     """
#     Return True if the last `window` values of `history`
#     have all changed by less than `tol`.
#     """
#     if len(history) < window:
#         return False
#     recent = history[-window:]
#     return max(abs(recent[i] - recent[i-1]) for i in range(1, window)) < tol

"""
utils.py
========
Convergence checks and general helper utilities for the TEBD simulation.
"""
from __future__ import annotations
import numpy as np
from typing import List

__all__ = ["is_converged"]


def is_converged(
    history: List[float],
    tol: float = 1e-10,
    window: int = 5,
) -> bool:
    """Check whether a scalar observable has converged.

    Returns ``True`` if the last ``window`` consecutive values of
    ``history`` have all changed by less than ``tol``. Useful for
    detecting convergence in imaginary-time evolution or iterative
    ground-state searches.

    Parameters
    ----------
    history : list of float
        Sequence of scalar values recorded at each step
        (e.g. energy, entropy, truncation error).
    tol : float, optional
        Absolute convergence threshold (default 1e-10).
    window : int, optional
        Number of recent steps to examine (default 5).

    Returns
    -------
    bool
        ``True`` if converged, ``False`` otherwise.

    Examples
    --------
    >>> is_converged([1.0, 1.0, 1.0, 1.0, 1.0])
    True
    >>> is_converged([1.0, 0.9, 0.8])          # too short
    False
    >>> is_converged([1.0, 1.0, 1.0, 1.0, 0.5], tol=1e-10)
    False
    """
    if len(history) < window:
        return False
    recent = history[-window:]
    return max(abs(recent[i] - recent[i - 1]) for i in range(1, window)) < tol


def normalize_lam(lam: "np.ndarray") -> "np.ndarray":
    """Renormalise a Schmidt vector so that Σ λ² = 1.

    Useful as a cheap sanity check after a long evolution to correct
    any small drift in normalisation caused by floating-point errors.

    Parameters
    ----------
    lam : ndarray
        1-D Schmidt vector.

    Returns
    -------
    ndarray
        Normalised copy of ``lam``.
    """
    import numpy as np
    norm = np.linalg.norm(lam)
    if norm < 1e-15:
        raise ValueError("Schmidt vector has near-zero norm — MPS may be corrupted.")
    return lam / norm


def check_normalisation(
    lam_list: List["np.ndarray"],
    tol: float = 1e-10,
) -> bool:
    """Check that every Schmidt vector satisfies Σ λ² = 1.

    A quick sanity check to confirm the MPS has not drifted out of
    Vidal canonical form during a long evolution.

    Parameters
    ----------
    lam_list : list of ndarray
        Schmidt vectors from the MPS.
    tol : float
        Tolerance for deviation from unit norm.

    Returns
    -------
    bool
        ``True`` if all bonds are normalised within ``tol``.
    """
    import numpy as np
    for i, lam in enumerate(lam_list):
        norm_sq = float(np.sum(lam ** 2))
        if abs(norm_sq - 1.0) > tol:
            return False
    return True


def format_time(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted string, e.g. ``'1h 23m 45s'`` or ``'3m 02s'``.

    Examples
    --------
    >>> format_time(5025.0)
    '1h 23m 45s'
    >>> format_time(182.0)
    '3m 02s'
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"