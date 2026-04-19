"""
mps.py
======
MPS initialization in Vidal (Gamma-Lambda) canonical form.

A product state |config⟩ is represented as a list of site tensors B[i]
of shape (chi_l, d, chi_r) and bond singular-value vectors lam[i],
where all bond dimensions chi = 1 for an unentangled state.
"""

import numpy as np
from typing import Union, List, Tuple

# -- Module-level constants --------------------------------------------------- #

# Maps spin labels to local basis indices.
# "up"   → index d-1  (e.g. index 1 for spin-1/2)
# "down" → index 0

_SPIN_LABELS = {
    "up":   None,   # resolved dynamically as d-1 at call time
    "down": 0,
}

_VALID_LABELS = {"up", "down"}

# -- Internal helpers --------------------------------------------------- #

def _resolve_config(config, d: int = 2) -> List[int]:
    """
    Resolve any supported config format into a flat list of basis indices.

    Parameters
    ----------
    config : list[int] | list[tuple]
        See ``init_mps`` docstring for all supported formats.
    d : int
        Local Hilbert-space dimension.

    Returns
    -------
    List[int]
        Flat list of basis indices, one per site.

    Supported formats
    -----------------
    1. Explicit flat list of ints    : [1, 0, 1, 0]
    2. Tuple list with string labels : [("up", 5), ("down", 4)]
    3. Tuple list with raw indices   : [(1, 5), (0, 4)]
    4. Mixed tuple list              : [("up", 5), (0, 4), ("down", 6)]
    """
    label_to_index = {"up": d - 1, "down": 0}

    # Format 2, 3, 4 — list of (state, length) tuples
    if isinstance(config, list) and len(config) > 0 and isinstance(config[0], tuple):
        resolved = []
        for item in config:
            if len(item) != 2:
                raise ValueError(
                    f"Each tuple must be (state, length), got {item!r}."
                )
            state, length = item

            if not isinstance(length, int) or length < 1:
                raise ValueError(
                    f"Length must be a positive integer, got {length!r}."
                )

            if isinstance(state, str):
                key = state.lower()
                if key not in _VALID_LABELS:
                    raise ValueError(
                        f"Unknown label {state!r}. Valid options: {sorted(_VALID_LABELS)}."
                    )
                resolved += [label_to_index[key]] * length

            elif isinstance(state, int):
                if not (0 <= state < d):
                    raise ValueError(
                        f"Index {state} out of range for d={d} "
                        f"(must satisfy 0 ≤ index < d)."
                    )
                resolved += [state] * length

            else:
                raise TypeError(
                    f"Tuple state must be str or int, got {type(state).__name__!r}."
                )
        return resolved

    # Format 1 — explicit flat list of ints
    elif isinstance(config, list):
        return config

    # Bare string — helpful error message
    elif isinstance(config, str):
        raise ValueError(
            f"A bare string {config!r} is not a valid config. "
            "Use [(\"up\", L)] or [(\"down\", L)] instead."
        )

    else:
        raise TypeError(
            f"Unsupported config type {type(config).__name__!r}. "
            "Expected a list of ints or a list of (state, length) tuples."
        )


# -- Public API --------------------------------------------------------- #

def init_mps(
    config: Union[List[int], List[Tuple]],
    d: int = 2,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Initialize an MPS in Vidal (Gamma-Lambda) canonical form representing
    an unentangled product state.

    In Vidal form the state is stored as:
    - B_list  : site tensors  B[i]   of shape (chi_l, d, chi_r)
    - lam_list: bond vectors  lam[i] of shape (chi,)

    For a product state every bond dimension chi = 1, so:
    - B[i]    has shape (1, d, 1)
    - lam[i]  = [1.0]

    The physical tensor at site i is recovered as:
        Gamma[i] = diag(lam[i])^{-1} · B[i] · diag(lam[i+1])^{-1}

    Parameters
    ----------
    config : list[int] | list[tuple]
        Specifies the local basis state at every site. Supported formats:

        1. **Explicit flat list** — direct basis indices, length = L::

               [1, 0, 1, 0]

        2. **Tuple list with string labels** — (label, length) pairs::

               [("up", 5), ("down", 5)]

           Valid labels: ``"up"`` → index d-1,  ``"down"`` → index 0.

        3. **Tuple list with raw indices** — (index, length) pairs::

               [(1, 5), (0, 5)]

        4. **Mixed** — freely combine labels and raw indices::

               [("up", 5), (0, 5), ("down", 5)]

    d : int, optional
        Local Hilbert-space dimension (default 2 for spin-1/2).

    Returns
    -------
    B_list : list of np.ndarray, length L
        Site tensors B[i] of shape (1, d, 1), dtype complex128.
    lam_list : list of np.ndarray, length L+1
        Bond singular-value vectors. Every entry equals [1.0].

    Raises
    ------
    ValueError
        If any index is outside [0, d), an unknown label is given,
        or a tuple has invalid length.
    TypeError
        If config format is not supported.

    Examples
    --------
    >>> # Explicit flat list — Néel state, 4 sites
    >>> B, lam = init_mps([1, 0, 1, 0])

    >>> # All-up state, 30 sites
    >>> B, lam = init_mps([("up", 30)])

    >>> # Néel state, 30 sites
    >>> B, lam = init_mps([("up", 1), ("down", 1)] * 15)

    >>> # Domain wall — 15 up, 15 down
    >>> B, lam = init_mps([("up", 15), ("down", 15)])

    >>> # Custom blocks via raw indices
    >>> B, lam = init_mps([(1, 10), (0, 5), (1, 10), (0, 5)])
    """
    # 1. Resolve config to a flat index list
    resolved = _resolve_config(config, d)

    # 2. Validate
    L = len(resolved)
    if L == 0:
        raise ValueError("Resolved config is empty — at least 1 site required.")

    invalid = [(i, s) for i, s in enumerate(resolved) if not (0 <= s < d)]
    if invalid:
        raise ValueError(
            f"Basis indices out of range for d={d}: "
            f"(site, index) pairs {invalid}."
        )

    # 3. Build MPS tensors
    B_list = []
    lam_list = [np.array([1.0]) for _ in range(L + 1)]

    for spin in resolved:
        B = np.zeros((1, d, 1), dtype=complex)
        B[0, spin, 0] = 1.0
        B_list.append(B)

    return (B_list, lam_list)