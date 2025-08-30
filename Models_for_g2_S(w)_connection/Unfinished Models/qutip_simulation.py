# qutip is the Quantum Toolbox in Python
from qutip import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from matplotlib.animation import FuncAnimation
from numpy.fft import fft, fftshift, fftfreq

# --------------------------------------------------------------------------
# System Definition
# --------------------------------------------------------------------------

# The total Hamiltonian for a 3-mode optomechanical system is given by:
# H_total = H_1 + H_2 + H_3
#
# Where the Hamiltonian for each individual mode 'j' is:
# H_j = -Δ_j * a_j^† * a_j + g̃_j * A_p * (a_j^† * b_j^† + a_j * b_j)
#
# And the total field operator is the sum of the individual optical operators:
# a_total = a_1 + a_2 + a_3

# --------------------------------------------------------------------------
# Parameters to be defined by the user
# --------------------------------------------------------------------------

# N_opt:      Fock space size for the optical modes (e.g., 10)
# N_ph:       Fock space size for the phonon modes (e.g., 10)

# For each mode j = 1, 2, 3:
#
# Δ_j (Delta_j):    Detuning of the optical mode from the pump laser.
# g̃_j (gtilde_j):   Single-photon optomechanical coupling strength.
# A_p:              Amplitude of the pump laser (can be treated as a constant).
# α_s_j (alpha_s_j):  Decay rate (loss) of the optical mode.
# α_p_j (alpha_p_j):  Decay rate (loss) of the phonon mode.
# a0_j:             Initial amplitude for the optical coherent state.
# b0_j:             Initial amplitude for the phonon coherent state.

# --------------------------------------------------------------------------
# Main execution block
# --------------------------------------------------------------------------
if __name__ == "__main__":
    
    # --- User-defined parameters will go here ---
    
    # --- QuTiP simulation logic (steadystate, mesolve, etc.) will go here ---
    
    # --- Analysis (correlation functions, spectrum) will go here ---
    
    # --- Plotting will go here ---

    #I will only try to work with this more if I really need to. I think for now the equation that was solved is
    # more than enough.
    
    print("Script setup complete. Ready for simulation logic.")