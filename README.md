# TEBD Spin Chain

**Real-time evolution of Matrix Product States using the Time-Evolving Block Decimation algorithm in Vidal (Γ–Λ) canonical form.**

---

## Overview

This project implements the **Time-Evolving Block Decimation (TEBD)** algorithm for simulating real-time quantum dynamics of one-dimensional spin-1/2 chains. The state is represented as a **Matrix Product State (MPS)** in **Vidal canonical form**, where the entanglement structure is made explicit through the Γ–Λ decomposition.

The implementation follows the original formulation of Vidal (2004) and supports:

- Arbitrary initial product states (Néel, domain wall, uniform, custom)
- Anisotropic XXZ Hamiltonian with a longitudinal magnetic field
- Second-order Suzuki–Trotter time evolution
- Correct open-boundary field distribution across edge and bulk bonds
- Entanglement entropy, bond dimensions, and energy as observables
- Validation against exact diagonalisation (ED) for small systems

---

## Physics

The Hamiltonian is defined using **Pauli matrices** (not spin-1/2 operators):

$$H = -h_z \sum_{l=1}^{L} \sigma_z^{[l]} + \sum_{l=1}^{L-1} \left( J_x\, \sigma_x^{[l]}\sigma_x^{[l+1]} + J_y\, \sigma_y^{[l]}\sigma_y^{[l+1]} + J_z\, \sigma_z^{[l]}\sigma_z^{[l+1]} \right)$$

> **Convention note:** TeNPy uses spin operators $S = \sigma/2$. To reproduce TeNPy results with couplings $(J_x^T, J_z^T, h_z^T)$, use $J = J^T/4$ and $h_z = h_z^T/2$.

### Vidal Canonical Form

The MPS is stored as a list of **Gamma tensors** $\Gamma^{[l]}$ (shape `(χ_L, d, χ_R)`) and **Schmidt vectors** $\Lambda^{[l]}$ (shape `(χ,)`), related by:

$$|\psi\rangle = \sum_{\{s\}} \Lambda^{[0]} \Gamma^{[1]} \Lambda^{[1]} \Gamma^{[2]} \Lambda^{[2]} \cdots \Gamma^{[L]} \Lambda^{[L]} |s_1 s_2 \cdots s_L\rangle$$

The two-site reduced state at bond $(l, l+1)$ is obtained by contracting:

$$\Theta^{[l,l+1]} = \Lambda^{[l]} \cdot \Gamma^{[l]} \cdot \Lambda^{[l+1]} \cdot \Gamma^{[l+1]} \cdot \Lambda^{[l+2]}$$

### TEBD Gate Application

Each time step applies a **second-order Suzuki–Trotter** decomposition:

```
even bonds (dt/2)  →  odd bonds (dt)  →  even bonds (dt/2)
```

The per-step Trotter error is $\mathcal{O}(dt^3)$, giving a total error of $\mathcal{O}(dt^2)$ at fixed time $t$.

### Open-Boundary Field Distribution

For a finite chain, the on-site field $-h_z \sigma_z^{[l]}$ is distributed across bonds so that every site receives its full contribution exactly once:

| Bond | Site $l$ gets | Site $l+1$ gets |
|---|---|---|
| Left edge $(0, 1)$ | $-h_z$ (full) | $-h_z/2$ |
| Bulk $(l, l+1)$ | $-h_z/2$ | $-h_z/2$ |
| Right edge $(L-2, L-1)$ | $-h_z/2$ | $-h_z$ (full) |

---

## Repository Structure

```
tebd-spin-chain/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── __init__.py              # Public API
│   ├── mps.py                   # MPS initialisation in Vidal form
│   ├── tebd.py                  # Gate application and sweep
│   ├── hamiltonian.py           # XXZ Hamiltonian and Trotter gates
│   ├── observables.py           # Energy, entropy, bond dimensions
│   └── utils.py                 # Convergence checks and helpers
│
├── examples/
│   └── run_tebd.ipynb           # All examples and demonstrations
│                                #   · Energy conservation check
│                                #   · Entanglement entropy growth
│                                #   · Exact diagonalisation comparison
│                                #   · Schmidt value validation plots
│
├── tests/
│   ├── 
│   └── 
│
└── figures/
```

---

## Installation

```bash
git clone https://github.com/Chayan-PHYSICS/TEBD-spin-chain.git
cd TEBD-spin-chain
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.8, NumPy, SciPy, Matplotlib.

---

## Quick Start
All examples are demonstrated interactively in `examples/run_tebd.ipynb`.
For a self-contained walkthrough, open the notebook directly:

```bash
jupyter notebook examples/run_tebd.ipynb
```

```python
from src.mps         import init_mps
from src.hamiltonian import build_xxz_two_site, build_trotter_gates
from src.tebd        import tebd_sweep
from src.observables import compute_energy, entanglement_entropy_bond

# Parameters
n_sites         = 20
chi_max         = 32
dt              = 0.005
Jx, Jy, Jz, hz = -1.0, -1.0, -1.0, 1.0

# Initial state: domain wall |↓↓↑↑↑...↑⟩
A_list, lam_list = init_mps([("down", 2), ("up", n_sites - 2)])

# Build Hamiltonians and Trotter gates
h_left, h_bulk, h_right = build_xxz_two_site(Jx, Jy, Jz, hz)
even_gates, odd_gates   = build_trotter_gates(h_left, h_bulk, h_right, dt)

# Time evolution
for step in range(1000):
    tebd_sweep(A_list, lam_list, even_gates, chi_max, parity=0)  # even, dt/2
    tebd_sweep(A_list, lam_list, odd_gates,  chi_max, parity=1)  # odd,  dt
    tebd_sweep(A_list, lam_list, even_gates, chi_max, parity=0)  # even, dt/2

# Observables
d = 2
E = compute_energy(
    A_list, lam_list,
    h_left.reshape(d,d,d,d),
    h_bulk.reshape(d,d,d,d),
    h_right.reshape(d,d,d,d),
)
S = entanglement_entropy_bond(lam_list[n_sites // 2])

print(f"Energy : {E:.6f}")
print(f"Entropy (central bond) : {S:.6f}")
```
---
## Key Features

### Flexible Initial States

```python
# Explicit flat list
A_list, lam_list = init_mps([1, 0, 1, 0, 1, 0])

# Néel state — 20 sites
A_list, lam_list = init_mps([("up", 1), ("down", 1)] * 10)

# Domain wall — 15 up, 15 down
A_list, lam_list = init_mps([("up", 15), ("down", 15)])
```

### Energy Conservation Check

Since unitary time evolution is energy-conserving, the drift in $\langle H \rangle$ over time quantifies the combined Trotter and truncation error:

```python
E0 = compute_energy(A_list, lam_list, h_left_4, h_bulk_4, h_right_4)

for step in range(n_steps):
    # ... evolve ...
    E = compute_energy(A_list, lam_list, h_left_4, h_bulk_4, h_right_4)
    print(f"|ΔE/E0| = {abs(E - E0) / abs(E0):.2e}")
```

### Exact Diagonalisation Validation

For small systems ($L \leq 12$), the Schmidt values from TEBD are compared against those obtained by exact SVD of the full state vector:

```python
from examples.compare_with_ed import run_comparison, print_summary

max_errors, times, psi_ed, lam_list = run_comparison(
    n_sites=6, chi_max=10, dt=0.005, n_steps=50
)
print_summary(max_errors)
```

A correct implementation shows errors growing as $\mathcal{O}(dt^3 \cdot \text{step})$, with no sudden jumps.

---

## Validation

The figure below shows the TEBD error against exact diagonalisation for a 6-site XXZ chain ($J_x = J_y = J_z = -1$, $h_z = 1$, $\chi_\text{max} = 10$, $dt = 0.005$):

- **Left:** Error growth tracks the expected $\mathcal{O}(dt^3 \cdot \text{step})$ Trotter scaling.
- **Right:** TEBD bond dimensions match ED exactly — truncation is negligible at $\chi_\text{max} = 10$ for this system size.

*(See `examples/run_tebd.ipynb` to reproduce all results.)*

---

## Algorithm Reference

| Step | Operation |
|---|---|
| 1 | Build $M = \Gamma^{[l]} \cdot \Lambda^{[l+1]} \cdot \Gamma^{[l+1]}$ |
| 2 | Apply gate $U$ on physical indices of $M$ only |
| 3 | Form $\Theta = \Lambda^{[l]} \cdot M \cdot \Lambda^{[l+2]}$ |
| 4 | SVD: $\Theta = U_\text{svd} \cdot S \cdot V^\dagger_\text{svd}$ |
| 5 | Truncate to $\chi_\text{max}$ largest singular values |
| 6 | Reconstruct $\Gamma^{[l]}_\text{new} = \Lambda^{[l]^{-1}} \cdot U_\text{svd}$, $\quad \Gamma^{[l+1]}_\text{new} = V^\dagger_\text{svd} \cdot \Lambda^{[l+2]^{-1}}$ |

---

## Reference

> Vidal, G. (2004). *Efficient Simulation of One-Dimensional Quantum Many-Body Systems.*
> Physical Review Letters, **93**(4), 040502.
> https://doi.org/10.1103/PhysRevLett.93.040502

---
