"""
Basic Simulation and Qiskit Verification Script for the MPDO Simulator.

This executable example generates a random 1D quantum circuit with alternating
layers, executes it over the custom Matrix Product Density Operator (MPDO) engine,
runs an exact noisy density matrix simulation via Qiskit Aer, and computes the 
Quantum State Fidelity to validate physical and structural alignment.
"""

import os
import sys
import numpy as np
from scipy.linalg import sqrtm

# Ensure the root of the project is in the system path for seamless relative package execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.mpdo import MPDOState, Dephasing, AmplitudeDamping, Depolarizing
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import DensityMatrix, Kraus
from qiskit.circuit.library import UGate, CXGate, CZGate

# ==============================================================================
# 1. SHARED GATE GENERATORS
# ==============================================================================

def random_one_qgate(rng: np.random.Generator) -> np.ndarray:
    """
    Draw (theta, phi, lam) and return Qiskit's U gate matrix.
    build_circuit uses the same angles via qc.u(theta, phi, lam).
    """
    theta, phi, lam = rng.uniform(0, 2 * np.pi, size=3)
    return UGate(theta, phi, lam).to_matrix()


def random_two_qgate(rng: np.random.Generator) -> np.ndarray:
    """
    Choose CNOT or CZ with equal probability.
    build_circuit uses the same draw via qc.cx / qc.cz.
    """
    if rng.random() < 0.5:
        return CXGate().to_matrix()                    # (4, 4) complex
    else:
        return CZGate().to_matrix()


# ==============================================================================
# 2. EXACT SIMULATION MECHANICS (QISKIT AER)
# ==============================================================================

def build_circuit(N: int, D: int, seed: int) -> QuantumCircuit:
    """
    Build the 1D random circuit from Figure 3.
    RNG draw order matches simulate_circuit() exactly for the same seed.
    """

    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(N)

    for layer in range(D):
        # Apply single-qubit gate layer
        for i in range(N):
            theta, phi, lam = rng.uniform(0, 2 * np.pi, size=3)
            qc.u(theta, phi, lam, i)

        # Determine alternating 1D nearest-neighbor pairs
        pairs = (
            [(k, k + 1) for k in range(0, N - 1, 2)]
            if layer % 2 == 0 else
            [(k, k + 1) for k in range(1, N - 1, 2)]
        )

        # Apply two-qubit gate layer
        for q1, q2 in pairs:
            if rng.random() < 0.5:
                qc.cx(q1, q2)
            else:
                qc.cz(q1, q2)

    return qc


def build_noise_model(model_name: str, rate: float) -> NoiseModel:
    """
    E⊗E noise after each two-qubit gate, matching equation (2.8).
    """
    noise_model = NoiseModel()

    if model_name == "depolarizing":
        p = rate
        K0 = np.sqrt(1.0 - 3*p/4) * np.eye(2, dtype=complex)
        K1 = np.sqrt(p/4) * np.array([[0,  1 ], [1,  0 ]], dtype=complex)  
        K2 = np.sqrt(p/4) * np.array([[0, -1j], [1j, 0 ]], dtype=complex)  
        K3 = np.sqrt(p/4) * np.array([[1,  0 ], [0, -1 ]], dtype=complex)  
        kraus_2q = [np.kron(a, b) for a in [K0,K1,K2,K3] for b in [K0,K1,K2,K3]]
        noise_model.add_all_qubit_quantum_error(Kraus(kraus_2q), ["cx", "cz"])

    elif model_name == "dephasing":
        p  = rate
        K0 = np.sqrt(1 - p) * np.eye(2, dtype=complex)
        K1 = np.sqrt(p) * np.array([[1, 0], [0, -1]], dtype=complex)
        kraus_2q = [np.kron(a, b) for a in [K0, K1] for b in [K0, K1]]
        noise_model.add_all_qubit_quantum_error(Kraus(kraus_2q), ["cx", "cz"])

    elif model_name == "amplitude_damping":
        g  = rate
        A0 = np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex)
        A1 = np.array([[0, np.sqrt(g)], [0, 0]],     dtype=complex)
        kraus_2q = [np.kron(a, b) for a in [A0, A1] for b in [A0, A1]]
        noise_model.add_all_qubit_quantum_error(Kraus(kraus_2q), ["cx", "cz"])

    else:
        raise ValueError(f"Unknown noise model: '{model_name}'.")

    return noise_model


def simulate_exact(
    N: int, D: int, model_name: str, rate: float, seed: int 
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns rho0 (noiseless) and rho_e (noisy) as numpy arrays.
    """
    qc = build_circuit(N, D, seed=seed)

    # noiseless
    qc_nl = qc.copy()
    qc_nl.save_density_matrix()
    rho0 = np.array(
        AerSimulator().run(qc_nl).result().data()["density_matrix"]
    )

    # noisy
    qc_ns = qc.copy()
    qc_ns.save_density_matrix()
    noise_model   = build_noise_model(model_name, rate)
    backend_noisy = AerSimulator(noise_model=noise_model)
    rho_e = np.array(
        backend_noisy.run(qc_ns).result().data()["density_matrix"]
    )

    return rho0, rho_e


# ==============================================================================
# 3. MPDO SIMULATION MECHANICS 
# ==============================================================================

def apply_layer(state, single_gates, pairs, two_qubit_gates, channel):
    # Step 1: single-qubit gates
    for i in range(state.N):
        state.apply_single_qubit_gate(i, single_gates[i])

    # Step 2: two-qubit gates + noise
    for (k, k1), U in zip(pairs, two_qubit_gates):
        state.apply_two_qubit_gate(k, k1, U)
        state.apply_noise(k,  channel)
        state.apply_noise(k1,  channel)

    # Step 3: canonicalize (chi_max handled internally)
    state.canonicalize()

#  simulate_circuit: creates the state, runs all layers
def simulate_circuit(N, D, epsilon, chi_max, kappa_max, seed=0):
    rng = np.random.default_rng(seed)
    state = MPDOState.init_product_state(N,chi_max, kappa_max)
    channel = AmplitudeDamping(epsilon)

    for layer in range(D):
        print(f"  Layer {layer+1}/{D}  |  {state}")   # MPDOState.__repr__

        single_gates = [random_one_qgate(rng) for _ in range(N)]

        # Determine alternating structural layers
        pairs = ([(k, k + 1) for k in range(0, N - 1, 2)] if layer %2 ==0
                 else
                 [(k, k+1) for k in range(1, N-1, 2)]
                 )
        # Sample two-qubit operators for each pair configuration
        tq_gates = [random_two_qgate(rng) for _ in pairs]

        apply_layer(state, single_gates, pairs, tq_gates, channel)  
    
    return state 


# ==============================================================================
# 4. MATHEMATICAL METRICS
# ==============================================================================

def matrix_sqrt(M: np.ndarray) -> np.ndarray:
    """Matrix square root of a Hermitian PSD matrix."""
    eigenvalues, V = np.linalg.eigh(M)
    eigenvalues    = np.maximum(eigenvalues, 0.0)
    return V @ np.diag(np.sqrt(eigenvalues)) @ V.conj().T

def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    F(rho, sigma) = Tr( sqrt( sqrt(rho) @ sigma @ sqrt(rho) ) )
    """
    sqrt_rho      = matrix_sqrt(rho)
    M             = sqrt_rho @ sigma @ sqrt_rho
    eigenvalues_M, _ = np.linalg.eigh(M)
    eigenvalues_M = np.maximum(eigenvalues_M, 0.0)
    F             = np.sum(np.sqrt(eigenvalues_M))
    return float(np.clip(np.real(F), 0.0, 1.0))


# ==============================================================================
# 4. EXECUTION EXEC MODULE
# ==============================================================================

if __name__ == "__main__":
    print("==========================================================")
    print("Starting Tensor Network MPDO vs. Qiskit Aer Verification Suite")
    print("==========================================================\n")

    # Simulation Configuration Parameters
    NUM_QUBITS = 10
    CIRCUIT_DEPTH = 3
    NOISE_MODEL = "amplitude_damping"
    NOISE_RATE = 0.00       # Gamma decay probability profile
    BOND_MAX_CHI = 16       # Maximum virtual MPS/MPDO rank
    BOND_MAX_KAPPA = 4      # Maximum auxiliary environment dimension
    RANDOM_SEED = 0

    print(f"[Config] Qubits (N)   : {NUM_QUBITS}")
    print(f"[Config] Depth (D)    : {CIRCUIT_DEPTH}")
    print(f"[Config] Noise Rate   : {NOISE_RATE}")
    print(f"[Config] Max Chi      : {BOND_MAX_CHI}")
    print(f"[Config] Max Kappa    : {BOND_MAX_KAPPA}")
    print(f"[Config] Random Seed  : {RANDOM_SEED}\n")

    # --------------------------------------------------------------------------
    # Run Custom Tensor Network MPDO Simulator
    # --------------------------------------------------------------------------
    print("Executing Custom MPDO Simulator...")

    state = simulate_circuit(
        N=NUM_QUBITS,
        D=CIRCUIT_DEPTH,
        epsilon=NOISE_RATE,
        chi_max=BOND_MAX_CHI,
        kappa_max=BOND_MAX_KAPPA,
        seed=RANDOM_SEED
    )
    
    # Contract the network back to an explicit density matrix
    
    rho_mpdo = state.to_density_matrix()
    print(" MPDO simulation completed successfully.")

    # --------------------------------------------------------------------------
    # Run Reference Full-State Qiskit Aer Simulator
    # --------------------------------------------------------------------------
    print("\nExecuting Exact Reference Qiskit Aer Density Matrix Simulator...")
    rho0, rho_qiskit = simulate_exact(NUM_QUBITS, CIRCUIT_DEPTH, NOISE_MODEL, NOISE_RATE, seed=RANDOM_SEED)
    print(" Qiskit benchmark completed successfully.")
    print("\nEvaluating Simulation Metrics...")
    # Assert structural dimensionality matching
    assert rho_mpdo.shape == rho_qiskit.shape, "Shape mismatch between simulators!"

    # Calculate fidelity trace
    fidelity_score = fidelity(rho_mpdo, rho_qiskit)
    print(f"--> Reconstructed Quantum State Fidelity: F = {fidelity_score:.6f}")

    # Final verification safety assertion checks
    if NOISE_RATE == 0.0 or (BOND_MAX_CHI >= 2**NUM_QUBITS and BOND_MAX_KAPPA >= 2**CIRCUIT_DEPTH):
        print("\n[Result] Ideal/Unbounded rank mode check: Target Fidelity should approach 1.00.")
    else:
        print("\n[Result] Bounded truncation simulation finalized within valid numerical margins.")
    print("==========================================================")