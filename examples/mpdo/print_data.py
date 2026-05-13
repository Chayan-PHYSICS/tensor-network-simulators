import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Path Setup
# file_pattern = "./examples/mpdo/inner_dim/fidelity_NEW_DP_N*_D*_noise*_seed*.txt"
file_pattern = "./examples/mpdo/chi_dim/fidelity_NEW_DPol_N*_D*_noise*_seed*.txt"
files = glob.glob(file_pattern)

if not files:
    print("No matching .txt files found! Check your path.")
else:
    plt.figure(figsize=(6, 5))
    files.sort()
    all_x_values = set()

    # --- Initialize metadata variables outside the loop ---
    n_qubits, depth, noise, seed = None, None, None, None

    for file_path in files:
        filename = os.path.basename(file_path)

        match = re.search(
            r"fidelity_NEW_DPol_N(\d+)_D(\d+)_noise([\d\.\-eE]+)_seed(\d+)\.txt",
            filename,
        )

        if match:
            # Assigning to the outer variables
            n_qubits = match.group(1)
            depth = match.group(2)
            noise = match.group(3)
            seed = match.group(4)
            label_text = f"N={n_qubits}, D={depth}, Noise={noise}, Seed={seed}"
        else:
            label_text = filename  

        try:
            data = np.loadtxt(file_path, delimiter="\t", skiprows=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)

            bond_max_x = data[:, 0]
            fidelity_scores = data[:, 1]
            all_x_values.update(bond_max_x)

            plt.plot(
                bond_max_x,
                fidelity_scores,
                marker="D",
                markersize=5,
                linestyle="-",
                linewidth=1.5,
                label=label_text,
            )
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Plot Layout
    plt.title("MPDO Quantum Circuit Simulation Comparison", fontsize=12, fontweight="bold", pad=15)
    # plt.xlabel("Maximum inner Bond Dimension (kappa)", fontsize=11)
    plt.xlabel("Maximum inner Bond Dimension (chi)", fontsize=11)
    plt.ylabel("Fidelity with Exact Qiskit State", fontsize=11)
    plt.xticks(sorted(list(all_x_values)))
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best", fontsize=9, framealpha=0.9, edgecolor="gray")
    plt.tight_layout()

    os.makedirs("./figures", exist_ok=True)
    
    # Check if the variables actually got populated during the loop
    if n_qubits is not None:
        fig_name = f"fidelity_plot_chi_DPol_N{n_qubits}_D{depth}_noise{noise}.png"
    else:
        fig_name = "fidelity_plot_unknown_params.png"
        
    
    output_plot_path = f"./figures/{fig_name}"
    
    plt.savefig(output_plot_path, dpi=150, bbox_inches="tight")
    print(f"🎉 Plot successfully saved to: {output_plot_path}")

    plt.show()
