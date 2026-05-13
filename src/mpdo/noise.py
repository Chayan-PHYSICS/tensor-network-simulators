"""
Kraus Noise Channels and Inner Dimension Compression for MPDO Simulation
===========================================================================================
Implements local noise injection via Completely Positive Trace-Preserving (CPTP)
Kraus maps on Matrix Product Density Operator (MPDO) site tensors, alongside
auxiliary-leg truncation to keep memory scaling polynomial.

Noise is injected site-locally after each two-qubit gate:

    ρ  →  U ∘ E^{⊗2}(ρ)                  

where E is a single-qubit CPTP map applied independently to each qubit.
The Kraus representation of E is:

    E(ρ) = Σ_k  A_k ρ A_k†              

In the MPDO formalism the Kraus branches are stacked along the inner index κ
via direct sum, keeping them orthogonal and avoiding spurious cross-terms.

Noise models
------------
Implemented
    amplitude_damping   : energy decay into a zero-temperature bath 
    dephasing           : phase randomisation channel               
    depolarizing        : global white-noise channel                 

"""
from __future__ import annotations
 
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import svd as la_svd

__all__ = [
    # primary  API
    "KrausChannel",
    "AmplitudeDamping",
    "Dephasing",
    "kraus_depolarizing",   
    # internal engine, still public
    "kraus_amplitude_damping",
    "kraus_dephasing",
    "get_kraus_ops",
    "apply_local_noise",
    "truncate_inner",
]

class KrausChannel(ABC):
    """
    Abstract base class for all single-qubit CPTP noise channels.
 
    Every concrete subclass must implement :meth:`kraus_ops`, which returns the list of Kraus matrices for that channel. 
    The shared :meth:`apply` method is inherited for free and handles the full noise-application + environment-compression pipeline.
 
    Design notes
    ------------
    - **Subclassing**: only override ``kraus_ops``.  Do not override ``apply`` unless the channel requires a non-standard contraction.
    - **Immutability**: channel parameters (``gamma``, ``epsilon``, …) are set at construction and should not be mutated afterwards.
    - The ``kappa_max`` truncation bound belongs to the *state*, not the channel, which is why it is passed as an argument to ``apply`` rather
      than stored here.
 
    Examples
    --------
    >>> ch = Dephasing(epsilon=0.01)
    >>> ops = ch.kraus_ops()          # [A0, A1] each (2,2)
    >>> evolved = ch.apply(tensor, kappa_max=4)
    """
    @abstractmethod
    def kraus_ops(self) -> list[np.ndarray]:
        """
        Return the Kraus operator list for this channel.
 
        Returns
        -------
        list of ndarray
            [K0, K1, …], each shape (d, d), dtype complex.
            Must satisfy the completeness relation: Σ_k K_k† K_k = I.
        """
        ...
    def apply(self, tensor: np.ndarray, kappa_max: int) -> np.ndarray:
        """
        Apply this channel to one MPDO site tensor and compress.
 
        Calls :func:`apply_local_noise` then :func:`truncate_inner` in sequence.
        Subclasses rarely need to override this.
 
        Parameters
        ----------
        tensor : ndarray, shape (chi_l, chi_r, d, kappa)
            Target site tensor.
        kappa_max : int
            Maximum inner (environment) dimension after compression. This bound is a property of the *state*, not the channel.
             
        Returns
        -------
        ndarray, shape (chi_l, chi_r, d, kappa_new)  where kappa_new <= kappa_max
        """
        tensor = apply_local_noise(tensor, self.kraus_ops())
        return truncate_inner(tensor, kappa_max)
 
    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v}" for k, v in self.__dict__.items()
            if not k.startswith("_")
        )
        return f"{type(self).__name__}({params})"
    

# ============================================================================
# Concrete channel classes
# ============================================================================

class AmplitudeDamping(KrausChannel):
    """
    Amplitude damping channel — models energy loss to a zero-temperature bath.

        E_AD(ρ) = A0 ρ A0† + A1 ρ A1†
    
    Parameters
    ----------
    gamma : float
        Decay probability per gate application, γ ∈ [0, 1].
 
    Examples
    --------
    >>> ch = AmplitudeDamping(gamma=0.05)
    >>> ch.kraus_ops()          # returns [A0, A1]
    >>> ch.apply(tensor, kappa_max=4)
    """
    def __init__(self, gamma: float) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}.")
        self.gamma = gamma

    def kraus_ops(self) -> list[np.ndarray]:
        """Kraus operators for amplitude damping with decay rate ``self.gamma``."""
        return kraus_amplitude_damping(self.gamma)

class Dephasing(KrausChannel):
    """
    Dephasing channel — destroys off-diagonal coherences without energy loss.
 
        E_DF(ρ) = (1−ε) ρ + ε Z ρ Z†
    
    Parameters
    ----------
    epsilon : float
        Dephasing probability per gate application, ε ∈ [0, 1].
 
    Examples
    --------
    >>> ch = Dephasing(epsilon=0.01)
    >>> ch.kraus_ops()          # returns [A0, A1]
    >>> ch.apply(tensor, kappa_max=4)
    """
    def __init__(self, epsilon: float) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}.")
        self.epsilon = epsilon
 
    def kraus_ops(self) -> list[np.ndarray]:
        """Kraus operators for dephasing with rate ``self.epsilon``."""
        return kraus_dephasing(self.epsilon)
    

class Depolarizing(KrausChannel):
    """
    Depolarizing channel — mixes the state toward the maximally mixed state.

        E_DP(ρ) = (1−ε) ρ + ε I/2

    Applied locally per qubit after each gate, using four Kraus operators built from I, X, Y, Z. This is a single-site
    operation, identical in structure to Dephasing and AmplitudeDamping.

    Parameters
    ----------
    epsilon : float
        Depolarizing probability per gate, ε ∈ [0, 1].

    Examples
    --------
    >>> ch = Depolarizing(epsilon=0.01)
    >>> ch.kraus_ops()       # [K0, K1, K2, K3], each (2, 2)
    >>> ch.apply(tensor, kappa_max=4)
    """

    def __init__(self, epsilon: float) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}.")
        self.epsilon = epsilon

    def kraus_ops(self) -> list[np.ndarray]:
        """Kraus operators for depolarizing with rate ``self.epsilon``."""
        return kraus_depolarizing(self.epsilon)

# ============================================================================    
# Kraus operator generators 
# ============================================================================
def kraus_amplitude_damping(gamma: float) -> list[np.ndarray]:
    """
    Single-qubit Kraus operators for the amplitude damping channel.
 
    Models energy dissipation from a two-level system into a zero-temperature thermal bath with decay probability γ per gate application.
    
    Parameters
    ----------
    gamma : float
        Decay probability per gate, γ ∈ [0, 1].
 
    Returns
    -------
    list of ndarray
        [A0, A1], each shape (2, 2), dtype complex.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}.")
 
    A0 = np.array([[1.0, 0.0],
                   [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
 
    A1 = np.array([[0.0, np.sqrt(gamma)],
                   [0.0, 0.0]], dtype=complex)
 
    return [A0, A1]

def kraus_dephasing(epsilon: float) -> list[np.ndarray]:
    """
    Single-qubit Kraus operators for the dephasing channel
    Parameters
    ----------
    epsilon : float
        Dephasing probability per gate, ε ∈ [0, 1].
 
    Returns
    -------
    list of ndarray
        [A0, A1], each shape (2, 2), dtype complex.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}.")
 
    A0 = np.sqrt(1.0 - epsilon) * np.array([[1.0, 0.0],
                                             [0.0, 1.0]], dtype=complex)
 
    A1 = np.sqrt(epsilon) * np.array([[1.0,  0.0],
                                      [0.0, -1.0]], dtype=complex)
    return [A0, A1]

def kraus_depolarizing(epsilon: float) -> list[np.ndarray]:
    """
    Single-qubit Kraus operators for the depolarizing channel.

    Models the local per-gate noise. Since the unitary U is applied separately before noise injection, 
    only the noise part is implemented here:

        E_DPol(ρ) = (1−ε) ρ + ε I/2

    Expanding I/2 = (IρI + XρX + YρY + ZρZ)/4 gives four Kraus operators:

    Parameters
    ----------
    epsilon : float
        Depolarizing probability per gate, ε ∈ [0, 1].
 
    Returns
    -------
    list of ndarray
        [K0, K1, K2, K3], each shape (2, 2), dtype complex.
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}.")

    K0 = np.sqrt(1.0 - 3.0 * epsilon / 4.0) * np.array([[1.0,  0.0],
                                                        [0.0,  1.0]], dtype=complex)

    K1 = np.sqrt(epsilon / 4.0) * np.array([[0.0,  1.0],
                                            [1.0,  0.0]], dtype=complex)  

    K2 = np.sqrt(epsilon / 4.0) * np.array([[0.0, -1.0j],
                                            [1.0j, 0.0 ]], dtype=complex)  

    K3 = np.sqrt(epsilon / 4.0) * np.array([[1.0,  0.0],
                                            [0.0, -1.0]], dtype=complex)  
    return [K0, K1, K2, K3]


# Registry: maps model name → KrausChannel subclass.
_CHANNEL_REGISTRY: dict[str, type[KrausChannel]] = {
    "amplitude_damping": AmplitudeDamping,
    "dephasing": Dephasing,
    "depolarizing": Depolarizing
}

# ------------------------- get kraus operator — internal engine -------------------------------------------

def get_kraus_ops(model_name: str, rate: float) -> KrausChannel:
    """
    Factory: construct and return the named :class:`KrausChannel`.
 
    This is the string-based entry point for examples and notebooks that select a noise model at runtime (e.g. from a config dict or CLI flag).
    For code that knows the channel type at write-time, prefer constructing directly:
        
        channel = Dephasing(epsilon=0.01)   # explicit — preferred in src/
        channel = get_kraus_ops("dephasing", 0.01)  # dynamic — ok in examples
 
    Parameters
    ----------
    model_name : str
        One of ``'amplitude_damping'``, ``'dephasing'``, ``'depolarizing'``
    rate : float
        Noise rate / decay probability for the chosen channel.
 
    Returns
    -------
    KrausChannel
        A configured channel instance.  Call ``.kraus_ops()`` to get the
        raw Kraus matrices, or pass directly to :meth:`MPDOState.apply_noise`.
 
    Raises
    ------
    ValueError
        If ``model_name`` is not registered.
 
    Examples
    --------
    >>> ch = get_kraus_ops("dephasing", 0.03)
    >>> type(ch)
    <class 'Dephasing'>
    >>> len(ch.kraus_ops())
    2
    """
    if model_name not in _CHANNEL_REGISTRY:
        available = list(_CHANNEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown noise model '{model_name}'. "
            f"Available: {available}"
        )
    return _CHANNEL_REGISTRY[model_name](rate)


# ------------------------ Noise application — internal engine -------------------------------------------------------------------------

def apply_local_noise(T: np.ndarray, kraus_ops: list[np.ndarray],) -> np.ndarray:
    """
    Apply a local CPTP noise channel to one MPDO site tensor.
 
    Each Kraus operator is contracted with the physical index of T to produce one noise branch. 
    If the channel has m operators, the inner dimension grows κ → m·κ.
    Call :func:`truncate_inner` immediately after to bound κ ≤ κ_max.
 
    Parameters
    ----------
    T : ndarray, shape (chi_l, chi_r, d, kappa)
        Target site tensor.
    kraus_ops : list of ndarray, each shape (d, d)
        Kraus operator set {A_k} for the chosen channel.
 
    Returns
    -------
    ndarray, shape (chi_l, chi_r, d, m * kappa)
        Noise-evolved site tensor with expanded inner dimension.
 
    Notes
    -----
    Prefer calling :meth:`KrausChannel.apply` from application code; this function is the low-level engine invoked by that method.
    """
    branches = []
 
    for E in kraus_ops:
        # Contract E[d_out, d_in] with T[chi_l, chi_r, d_in, kappa]
        # Result shape: (d_out, chi_l, chi_r, kappa)
        branch = np.tensordot(E, T, axes=([1], [2]))
 
        # Restore standard layout: (chi_l, chi_r, d_out, kappa)
        branch = branch.transpose(1, 2, 0, 3)
        branches.append(branch)
 
    # Stack branches along inner axis (axis 3): kappa → m·kappa, Each branch occupies an orthogonal block — no cross-terms in M = T T†
    return np.concatenate(branches, axis=3)

# ----------------------------- Inner dimension compression — internal engine --------------------------------------------------------
 
def truncate_inner(T: np.ndarray, kappa_max: int) -> np.ndarray:
    """
    Compress the inner (κ) dimension of an MPDO site tensor via local SVD.
 
    This operation is purely local — no coupling between sites — and is applied immediately after :func:`apply_local_noise` on the affected sites.

    Parameters
    ----------
    T : ndarray, shape (chi_l, chi_r, d, kappa_current)
        Noise-expanded site tensor.
    kappa_max : int
        Maximum inner dimension after compression.
 
    Returns
    -------
    ndarray, shape (chi_l, chi_r, d, kappa_new)  where kappa_new <= kappa_max
        Compressed site tensor.
 
    Notes
    -----
    Prefer calling :meth:`KrausChannel.apply` from application code; this function is the low-level engine invoked by that method.
    """
    chi_l, chi_r, d, kap = T.shape
 
    # Reshape T : (chi_l, chi_r, d, kappa) → (chi_l·chi_r·d, kappa)
    T_mat = T.reshape(chi_l * chi_r * d, kap)
 
    try:
        U_svd, S, _ = la_svd(T_mat, full_matrices=False, lapack_driver='gesdd')
    except np.linalg.LinAlgError:
        U_svd, S, _ = la_svd(T_mat, full_matrices=False, lapack_driver='gesvd')
 
    kap_new = min(kappa_max, len(S))
 
    # Absorb singular values: T_new = U[:, :κ] · diag(S[:κ])
    T_compressed = U_svd[:, :kap_new] * S[None, :kap_new]
 
    # Restore full tensor layout: (chi_l·chi_r·d, κ_new) → (chi_l, chi_r, d, κ_new)
    return T_compressed.reshape(chi_l, chi_r, d, kap_new)
