from qutip import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from matplotlib.animation import FuncAnimation



'''
Schedule:
	1. Create a random thermal state
	2. Define a random hamiltonian. Use the Linband master equation
	3. Explore the heterodyne simulation and the photon statistics simulation
	4. Using that try to implement it on the system we have.
'''

def create_and_plot_thermal_state():
    """
    Creates a thermal state and plots its Wigner function and photon distribution.
    """
    # =========================================================================
    # 1. Define Parameters for the Thermal State
    # =========================================================================
    
    # Hilbert space dimension (size of the Fock space).
    # This must be large enough to accurately represent the state.
    # A good rule of thumb is N > 5 * n_avg.
    N = 30
    
    # Average number of thermal photons (phonons in your case).
    # This is directly related to the temperature of the bath.
    # Try changing this value (e.g., to 1, 5, 10) to see how the plots change!
    n_avg = 5.0
    
    # =========================================================================
    # 2. Create the Thermal State
    # =========================================================================
    
    # qutip.thermal_dm(N, n) creates the density matrix for a thermal state.
    # It represents a statistical mixture of number states, weighted by the
    # Bose-Einstein distribution.
    rho_thermal = thermal_dm(N, n_avg)
    
    print("Created a thermal state with the following properties:")
    print(f"  - Hilbert Space Dimension (N): {N}")
    print(f"  - Average Photon Number (n_avg): {n_avg}")
    print("\nDensity Matrix (first 5x5 elements):")
    print(rho_thermal.full()[:5, :5].round(4))
    
    # =========================================================================
    # 3. Visualize the State
    # =========================================================================
    
    # --- Plot 1: Wigner Function ---
    # The Wigner function is a quasi-probability distribution in phase space.
    # For a thermal state, it's a Gaussian centered at the origin.
    
    xvec = np.linspace(-6, 6, 200)  # Range for x and p quadratures
    W_thermal = wigner(rho_thermal, xvec, xvec)
    
    # --- Plot 2: Photon Number Distribution ---
    # The diagonal elements of the density matrix in the Fock basis give the
    # probability P(n) of finding 'n' photons in the state.
    
    photon_distribution = rho_thermal.diag()
    n_values = np.arange(N)

    # --- Create the plots ---
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Visualizing a Thermal State (n̄ = {n_avg})', fontsize=16)

    # Wigner function plot
    contour = axs[0].contourf(xvec, xvec, W_thermal, 100, cmap='RdBu_r')
    axs[0].set_title('Wigner Function in Phase Space')
    axs[0].set_xlabel('Position Quadrature (x)')
    axs[0].set_ylabel('Momentum Quadrature (p)')
    fig.colorbar(contour, ax=axs[0])
    
    # Photon number distribution plot
    axs[1].bar(n_values, photon_distribution, color='dodgerblue', alpha=0.8)
    axs[1].set_title('Photon Number Distribution P(n)')
    axs[1].set_xlabel('Photon Number (n)')
    axs[1].set_ylabel('Probability P(n)')
    axs[1].set_xlim(-0.5, 3 * n_avg + 2) # Adjust x-limit for better viewing
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def calculate_g2_for_thermal_mode():
    """
    Calculates and plots g^(2)(τ) for a single thermal mode.
    """
    # =========================================================================
    # 1. Define the System (using your provided parameters)
    # =========================================================================
    
    # Hilbert space dimension
    N = 30
    
    # System parameters
    omega_M = 1.0  # Mechanical frequency (frequency of the phonon mode)
    gamma = 0.1    # Damping rate (how quickly it loses energy to the bath)
    n_th = 5.0     # Mean thermal occupation of the bath (temperature)

    # Operators
    b = destroy(N)
    bd = b.dag()

    # Hamiltonian of the single harmonic oscillator
    H = omega_M * (bd * b) # The 0.5 is a constant energy shift, can be ignored

    # Collapse operators describing the interaction with the thermal bath
    # c1 describes energy loss (decay), proportional to n_th + 1
    c1 = np.sqrt(gamma * (n_th + 1)) * b 
    # c2 describes energy gain (excitation), proportional to n_th
    c2 = np.sqrt(gamma * n_th) * bd
    
    c_ops = [c1, c2]

    print("System Parameters:")
    print(f"  - Frequency (ω_M): {omega_M}")
    print(f"  - Damping Rate (γ): {gamma}")
    print(f"  - Thermal Occupation (n̄_th): {n_th}")

    # =========================================================================
    # 2. Find the Steady State and Calculate Correlations
    # =========================================================================
    
    # First, find the steady state density matrix of the system.
    # This is the state the oscillator settles into due to the thermal bath.
    print("\nCalculating the steady state of the system...")
    rho_ss = steadystate(H, c_ops)
    
    # Check the average number of phonons in the steady state
    n_ss = expect(bd * b, rho_ss)
    print(f"Calculated average phonon number in steady state: {n_ss:.4f}")
    print(f"(This should be very close to n_th={n_th})")

    # Define a list of time delays (τ) for the correlation function
    tlist = np.linspace(0, 50, 500)

    # Calculate the first-order correlation function g^(1)(τ) = <b†(τ)b(0)>
    # This is the most computationally intensive step.
    # CORRECTED: Used the correct function name, correlation_2op_1t.
    print("\nCalculating the first-order correlation function g^(1)(τ)...")
    g1_t = correlation_2op_1t(H, rho_ss, tlist, c_ops, bd, b)
    
    # For the normalized g1, we divide by the number of photons at t=0
    # g1_t[0] is equivalent to <b†(0)b(0)> = <n>
    g1_t_normalized = g1_t / g1_t[0]

    # Now, apply the Siegert Relation for chaotic light to get g^(2)(τ)
    g2_t = 1 + np.abs(g1_t_normalized)**2
    
    # =========================================================================
    # 3. Plot the Result
    # =========================================================================
    print("Plotting the results...")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.plot(tlist, g2_t.real, 'r-', linewidth=2)
    ax.set_title(r'Photon Statistics $g^{(2)}(\tau)$ for a Single Thermal Mode', fontsize=14)
    ax.set_xlabel(r'Delay Time $\tau$ (1/γ)', fontsize=12)
    ax.set_ylabel(r'$g^{(2)}(\tau)$', fontsize=12)
    
    # Add text to highlight the key features
    ax.text(0.05, 0.9, f'$g^{{(2)}}(0) = {g2_t[0]:.2f}$', transform=ax.transAxes)
    ax.text(0.6, 0.15, r'$g^{(2)}(\tau \to \infty) \to 1$', transform=ax.transAxes)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
def calculate_g2_for_multi_mode():
    """
    Calculates and plots g^(2)(τ) for a composite system of three thermal modes.
    """
    # =========================================================================
    # 1. Define the System for Three Competing Modes
    # =========================================================================
    
    # Hilbert space dimension for EACH oscillator.
    # The total space will be N*N*N.
    # CORRECTED: Reduced N from 8 to 5 to prevent MemoryError.
    N = 3
    
    # --- Mode 1 Parameters ---
    omega_1 = 10.0  # Frequency of mode 1
    gamma_1 = 0.1   # Damping rate of mode 1
    n_th_1 = 3.0    # Thermal occupation of mode 1 (proportional to its intensity)

    # --- Mode 2 Parameters ---
    omega_2 = 10.5  # A slightly different frequency
    gamma_2 = 0.1
    n_th_2 = 1

    # --- Mode 3 Parameters ---
    omega_3 = 13  # Another different frequency
    gamma_3 = 0.1
    n_th_3 = 0.1

    # =========================================================================
    # 2. Build the Composite System using Tensor Products
    # =========================================================================
    
    # Create destruction operators for each mode in the composite space.
    # qeye(N) is the identity operator (a placeholder).
    b1 = tensor(destroy(N), qeye(N), qeye(N))
    b2 = tensor(qeye(N), destroy(N), qeye(N))
    b3 = tensor(qeye(N), qeye(N), destroy(N))

    # The total Hamiltonian is the sum of the individual Hamiltonians.
    H = omega_1 * b1.dag() * b1 + \
        omega_2 * b2.dag() * b2 + \
        omega_3 * b3.dag() * b3

    # Create collapse operators for each mode's thermal bath.
    c_ops = [
        # Mode 1 bath interaction
        np.sqrt(gamma_1 * (n_th_1 + 1)) * b1,
        np.sqrt(gamma_1 * n_th_1) * b1.dag(),
        # Mode 2 bath interaction
        np.sqrt(gamma_2 * (n_th_2 + 1)) * b2,
        np.sqrt(gamma_2 * n_th_2) * b2.dag(),
        # Mode 3 bath interaction
        np.sqrt(gamma_3 * (n_th_3 + 1)) * b3,
        np.sqrt(gamma_3 * n_th_3) * b3.dag(),
    ]

    # =========================================================================
    # 3. Find Steady State and Calculate Correlations for the TOTAL Field
    # =========================================================================

    print("Calculating the steady state of the 3-mode system...")
    rho_ss = steadystate(H, c_ops)
    
    # The detector measures the total field, which is the sum of the individual fields.
    b_total = b1 + b2 + b3

    # Define a list of time delays (τ)
    tlist = np.linspace(0, 80, 2000) # Use more points to resolve the fast beats

    print("Calculating g^(1)(τ) for the total field...")
    g1_t = correlation_2op_1t(H, rho_ss, tlist, c_ops, b_total.dag(), b_total)
    
    # Normalize and apply the Siegert Relation
    g1_t_normalized = g1_t / g1_t[0]
    g2_t = 1 + np.abs(g1_t_normalized)**2

    # =========================================================================
    # 4. Plot the Result
    # =========================================================================
    print("Plotting the results...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(tlist, g2_t.real, 'b-', linewidth=2)
    ax.set_title(r'Photon Statistics $g^{(2)}(\tau)$ for Three Competing Modes', fontsize=14)
    ax.set_xlabel(r'Delay Time $\tau$', fontsize=12)
    ax.set_ylabel(r'$g^{(2)}(\tau)$', fontsize=12)
    ax.set_xlim(0, tlist[-1])
    
    ax.text(0.5, 0.9, 'Beating is caused by interference between the modes!', 
            transform=ax.transAxes, ha='center', style='italic',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
    
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()




# --- Run the main function ---
if __name__ == "__main__":
    #create_and_plot_thermal_state()
	#calculate_g2_for_thermal_mode()
	calculate_g2_for_multi_mode()