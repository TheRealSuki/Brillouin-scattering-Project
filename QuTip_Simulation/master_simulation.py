from qutip import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from matplotlib.animation import FuncAnimation
from numpy.fft import fft, fftshift, fftfreq


'''
Schedule:
	1. Create a random thermal state
	2. Define a random hamiltonian. Use the Linband master equation
	3. Explore the heterodyne simulation and the photon statistics simulation
	4. Using that try to implement it on the system we have.

    It explores the theory that was uploaded on the 3th August on the Teams. Siegert, Fourier transform to get the Power spectrum and also the Wiener–Khinchin theorem.

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
    N = 3
    
    # --- Mode 1 Parameters ---
    omega_1 = 10.0  # Frequency of mode 1
    gamma_1 = 0.1   # Damping rate of mode 1
    n_th_1 = 3.0    # Thermal occupation of mode 1 (proportional to its intensity)

    # --- Mode 2 Parameters ---
    omega_2 = 11  # A slightly different frequency
    gamma_2 = 0.1
    n_th_2 = 1

    # --- Mode 3 Parameters ---
    omega_3 = 12  # Another different frequency
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

def calculate_g2_and_spectrum():
    """
    Calculates and plots g^(2)(τ) and the power spectrum S(ω) for a
    composite system of three thermal modes.
    """
    # =========================================================================
    # 1. Define the System for Three Competing Modes
    # =========================================================================
    
    # Hilbert space dimension for EACH oscillator.
    N = 5
    
    # --- Mode 1 Parameters ---
    omega_1 = 10.0  # Frequency of mode 1
    gamma_1 = 0.1   # Damping rate of mode 1
    n_th_1 = 0.1    # Thermal occupation of mode 1 (proportional to its intensity)

    # --- Mode 2 Parameters ---
    omega_2 = 11  # A slightly different frequency
    gamma_2 = 0.1
    n_th_2 = 1

    # --- Mode 3 Parameters ---
    omega_3 = 12  # Another different frequency
    gamma_3 = 0.1
    n_th_3 = 3

    # =========================================================================
    # 2. Build the Composite System using Tensor Products
    # =========================================================================
    
    b1 = tensor(destroy(N), qeye(N), qeye(N))
    b2 = tensor(qeye(N), destroy(N), qeye(N))
    b3 = tensor(qeye(N), qeye(N), destroy(N))

    H = omega_1 * b1.dag() * b1 + \
        omega_2 * b2.dag() * b2 + \
        omega_3 * b3.dag() * b3

    c_ops = [
        np.sqrt(gamma_1 * (n_th_1 + 1)) * b1, np.sqrt(gamma_1 * n_th_1) * b1.dag(),
        np.sqrt(gamma_2 * (n_th_2 + 1)) * b2, np.sqrt(gamma_2 * n_th_2) * b2.dag(),
        np.sqrt(gamma_3 * (n_th_3 + 1)) * b3, np.sqrt(gamma_3 * n_th_3) * b3.dag(),
    ]

    # =========================================================================
    # 3. Find Steady State and Calculate Correlations
    # =========================================================================

    print("Calculating the steady state of the 3-mode system...")
    rho_ss = steadystate(H, c_ops)
    
    b_total = b1 + b2 + b3
    tlist = np.linspace(0, 80, 4000) # Use more points for better FFT resolution

    print("Calculating g^(1)(τ) for the total field...")
    g1_t = correlation_2op_1t(H, rho_ss, tlist, c_ops, b_total.dag(), b_total)
    
    # --- Calculate g^(2)(τ) for the time-domain plot ---
    g1_t_normalized = g1_t / g1_t[0]
    g2_t = 1 + np.abs(g1_t_normalized)**2

    # =========================================================================
    # 4. Calculate the Spectrum via Fourier Transform
    # =========================================================================
    print("Calculating the power spectrum via Fourier Transform of g^(1)(τ)...")
    
    N_t = len(tlist)
    dt = tlist[1] - tlist[0]
    
    # Perform the FFT on the first-order correlation function
    S_w = fft(g1_t)
    
    # Create the frequency axis for the plot. The frequencies are relative
    # to the original mode frequencies. 2*pi is to convert to angular frequency.
    wlist = 2 * np.pi * fftfreq(N_t, dt)

    # Shift the FFT so that zero frequency is at the center
    S_w_shifted = fftshift(S_w)
    wlist_shifted = fftshift(wlist)

    # =========================================================================
    # 5. Plot Both Results
    # =========================================================================
    print("Plotting the results...")
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Time and Frequency Domain Pictures of a 3-Mode Thermal Field', fontsize=16)

    # --- Plot 1: Photon Statistics g^(2)(τ) [Time Domain] ---
    axs[0].plot(tlist, g2_t.real, 'b-')
    axs[0].set_title(r'Time Domain: Photon Statistics $g^{(2)}(\tau)$')
    axs[0].set_xlabel(r'Delay Time $\tau$ (a.u.)')
    axs[0].set_ylabel(r'$g^{(2)}(\tau)$')
    axs[0].set_xlim(0, tlist[-1])
    axs[0].grid(True, linestyle='--')
    axs[0].text(0.5, 0.9, 'Beating from mode interference', 
                transform=axs[0].transAxes, ha='center', style='italic')

    # --- Plot 2: Heterodyne Spectrum S(ω) [Frequency Domain] ---
    axs[1].plot(wlist_shifted, np.abs(S_w_shifted), 'r-')
    axs[1].set_title(r'Frequency Domain: Spectrum $S(\omega) = \mathcal{F}[g^{(1)}(\tau)]$')
    axs[1].set_xlabel(r'Frequency $\omega$ (a.u.)')
    axs[1].set_ylabel(r'Power Spectral Density (a.u.)')
    axs[1].set_xlim(omega_1 - 2, omega_3 + 2) # Zoom in on the peaks
    
    # Add vertical lines to show the original mode frequencies
    for omega, n_th in zip([omega_1, omega_2, omega_3], [n_th_1, n_th_2, n_th_3]):
        axs[1].axvline(x=omega, color='k', linestyle='--', alpha=0.6, 
                       label=f'Mode at ω={omega}, n̄={n_th}')
    axs[1].legend()
    axs[1].grid(True, linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def calculate_g2_and_spectrum_differentHilbertSpaces():
    """
    Calculates and plots g^(2)(τ) and the power spectrum S(ω) for a
    composite system of three thermal modes using a memory-efficient method.
    """
    # =========================================================================
    # 1a. Define Simulation Parameters for Modes in Physical Units
    # =========================================================================
    
    # Define a base frequency unit for the simulation.
    base_freq_GHz = 1.0
    
    # --- Your desired parameters (now interpreted as GHz) ---
    params = [
        {'name': 'Mode 1', 'omega_GHz': 10.0, 'gamma_GHz': 0.1, 'n_th': 0.1, 'N': 4},
        {'name': 'Mode 2', 'omega_GHz': 11.0, 'gamma_GHz': 0.1, 'n_th': 1.0, 'N': 6},
        {'name': 'Mode 3', 'omega_GHz': 12.0, 'gamma_GHz': 0.1, 'n_th': 3.0, 'N': 10},
    ]

    # Time list for the correlation functions, now in seconds.
    tlist = np.linspace(0, 50e-9, 4000) # 0 to 50 ns

    # =========================================================================
    # 1b. Define Physical Conversion Parameters)
    # =========================================================================
    
    # Assume the TOTAL integrated power measured by the detector is 1 nanowatt.
    # This factor accounts for detector efficiency, coupling, etc.
    total_power_watts = 1.0e-9  # 1 nW

    # =========================================================================
    # 2. Solve for Each Mode INDIVIDUALLY
    # =========================================================================
    
    g1_total = np.zeros_like(tlist, dtype=np.complex128)

    print("Solving for each mode individually (memory-efficient method)...")
    for p in params:
        print(f"  Calculating for {p['name']} (ω={p['omega_GHz']} GHz, n_th={p['n_th']})...")
        
        omega_rad_s = p['omega_GHz'] * base_freq_GHz * 1e9 * (2 * np.pi)
        gamma_rad_s = p['gamma_GHz'] * base_freq_GHz * 1e9 * (2 * np.pi)
        
        b = destroy(p['N'])
        H = omega_rad_s * b.dag() * b
        c_ops = [
            np.sqrt(gamma_rad_s * (p['n_th'] + 1)) * b,
            np.sqrt(gamma_rad_s * p['n_th']) * b.dag()
        ]
        
        rho_ss = steadystate(H, c_ops)
        g1_mode = correlation_2op_1t(H, rho_ss, tlist, c_ops, b.dag(), b)
        g1_total += g1_mode

    print("All modes calculated and combined.")

    # =========================================================================
    # 3. Calculate Final g^(2)(τ) and Spectrum from the Combined g^(1)(τ)
    # =========================================================================
    
    g1_total_normalized = g1_total / g1_total[0]
    g2_t = 1 + np.abs(g1_total_normalized)**2

    print("Calculating the power spectrum via Fourier Transform of total g^(1)(τ)...")
    
    N_t = len(tlist)
    dt = tlist[1] - tlist[0]
    
    S_w_au = fft(g1_total) # Spectrum in arbitrary units
    wlist_Hz = fftfreq(N_t, dt)

    S_w_au_shifted = np.abs(fftshift(S_w_au))
    wlist_Hz_shifted = fftshift(wlist_Hz)

    # =========================================================================
    # 4. Convert Spectrum to Physical Power Units (Watts and dBm)
    # =========================================================================
    
    # Calculate the scaling factor to match the desired total power
    # Integrated power in a.u. is the area under the curve
    integrated_power_au = np.trapz(S_w_au_shifted, wlist_Hz_shifted)
    scaling_factor = total_power_watts / integrated_power_au
    
    # Power Spectral Density in Watts / Hz
    psd_watts_hz = S_w_au_shifted * scaling_factor

    # Convert Power Spectral Density to dBm / Hz
    # Add a small epsilon to avoid log(0) errors where the spectrum is zero
    epsilon = 1e-30 
    psd_dBm_hz = 10 * np.log10((psd_watts_hz + epsilon) / 1e-3) # 1mW = 1e-3 W

    # =========================================================================
    # 5. Plot All Results with Physical Units
    # =========================================================================
    print("Plotting the results...")
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Time and Frequency Domain Pictures of a 3-Mode Thermal Field', fontsize=16)

    # --- Plot 1: Photon Statistics g^(2)(τ) [Time Domain] ---
    axs[0].plot(tlist * 1e9, g2_t.real, 'b-')
    axs[0].set_title(r'Time Domain: Photon Statistics $g^{(2)}(\tau)$')
    axs[0].set_xlabel(r'Delay Time $\tau$ (ns)')
    axs[0].set_ylabel(r'$g^{(2)}(\tau)$')
    axs[0].set_xlim(0, tlist[-1] * 1e9)
    axs[0].grid(True, linestyle='--')

    # --- Plot 2: Spectrum in Watts [Frequency Domain] ---
    axs[1].plot(wlist_Hz_shifted / 1e9, psd_watts_hz, 'r-')
    axs[1].set_title(r'Frequency Domain: Spectrum $S(f)$')
    axs[1].set_xlabel(r'Frequency $f$ (GHz)')
    axs[1].set_ylabel(r'Power Spectral Density (W/Hz)')
    axs[1].set_xlim(params[0]['omega_GHz'] - 2, params[-1]['omega_GHz'] + 2)
    axs[1].grid(True, linestyle='--')

    # --- Plot 3: Spectrum in dBm [Frequency Domain] ---
    axs[2].plot(wlist_Hz_shifted / 1e9, psd_dBm_hz, 'g-')
    axs[2].set_title(r'Frequency Domain: Spectrum $S(f)$ in dBm')
    axs[2].set_xlabel(r'Frequency $f$ (GHz)')
    axs[2].set_ylabel(r'Power Spectral Density (dBm/Hz)')
    axs[2].set_xlim(params[0]['omega_GHz'] - 2, params[-1]['omega_GHz'] + 2)
    axs[2].grid(True, linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- Run the main function ---
if __name__ == "__main__":
    #create_and_plot_thermal_state()
	#calculate_g2_for_thermal_mode()
	#calculate_g2_for_multi_mode()
    #calculate_g2_and_spectrum()
    calculate_g2_and_spectrum_differentHilbertSpaces()