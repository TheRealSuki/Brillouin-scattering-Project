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


def run_simulation(p):
    """
    Runs the full simulation and analysis based on the provided parameters.
    """
    # --- 1. Calculate intermediate variables from parameters ---
    k_j = p['k_j_real'] + 1j * p['k_j_imag']
    chi_s = -p['alpha_s']/2 + 1j * p['Delta_s']
    chi_m = -p['alpha_m']/2

    A = (chi_s + chi_m) / 2
    # Use np.lib.scimath.sqrt for complex square roots
    # Note: The user's notes had a minus sign under the root, but standard solutions
    # for this type of system (Bogoliubov transformation) use a plus sign.
    # Using the standard form here.
    D = np.lib.scimath.sqrt((chi_s - chi_m)**2 + 4 * k_j * np.conj(k_j))

    lambda_p = A + D/2
    lambda_m = A - D/2

    # --- CORRECTED COEFFICIENT DEFINITIONS ---
    # Coefficients P, Q, L as per the user's notes.
    # Added a small epsilon to avoid division by zero.
    epsilon = 1e-15
    # P has (chi_s - A + D/2) in the denominator
    P_denom = chi_s - A - D/2
    # Q has (chi_s - A - D/2) in the denominator
    Q_denom = chi_s - A + D/2
    
    P = k_j / (P_denom + epsilon)
    Q = k_j / (Q_denom + epsilon)
    
    # The definition of L depends on the P and Q from the eigenvector matrix,
    # which can be different from the P and Q in the final solution coefficients.
    # For a standard solution v(z) = G(z)v(0), L is implicitly handled by the
    # matrix components of G(z). Here, we stick to the user's formula.
    L = 1 / (P - Q + epsilon)

    # --- 2. Simulate the evolution of a(z) ---
    z = np.linspace(0, p['z_max'], p['num_points'])

    # The formula from the user's notes for a_s_j(z).
    # Based on the structure, we assume the second term is multiplied by b0_dagger.
    term1 = (-P * np.exp(lambda_p * z) + Q * np.exp(lambda_m * z)) * p['a0']
    term2 = (P * Q * (-np.exp(lambda_p * z) + np.exp(lambda_m * z))) * p['b0_dagger']
    a_z = L * (term1 + term2)

    # --- 3. First-Order Coherence g^(1)(tau) and Power Spectrum ---
    # Calculate g^(1) via normalized autocorrelation
    autocorr = np.correlate(a_z, a_z, mode='full')
    g1 = autocorr / np.sum(np.abs(a_z)**2)
    tau = np.arange(-p['num_points'] + 1, p['num_points']) * (z[1] - z[0])

    # Power Spectrum S(omega) is the Fourier Transform of g1
    S_omega = fftshift(fft(fftshift(g1)))
    omega = fftshift(fftfreq(len(tau), d=(z[1] - z[0]))) * 2 * np.pi

     # --- 4. Calculate g^(2)(tau) from a(z) ---
    # This now explicitly calculates the time-averaged intensity correlation.
    I_z = np.abs(a_z)**2
    mean_I = np.mean(I_z)
    
    # Calculate the intensity autocorrelation for the numerator of g^(2)
    autocorr_I = np.correlate(I_z, I_z, mode='full')
     # Define the integer lags for normalization
    tau_indices = np.arange(-p['num_points'] + 1, p['num_points'])
    
    # Normalize the autocorrelation to get the time average <I(t)I(t+tau)>
    # The number of overlapping points for each lag tau is N - |tau|
    normalization = p['num_points'] - np.abs(tau_indices)
    numerator_g2 = autocorr_I / normalization
    
    # Calculate g^(2) using the definition
    denominator_g2 = mean_I**2
    g2 = numerator_g2 / denominator_g2

    # --- 5. Plotting ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(12, 16))
    fig.suptitle('Direct Simulation and Coherence Analysis', fontsize=16)

    # Plot a(z)
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(z, np.abs(a_z), 'r-', label='$|a(z)|$')
    ax1.set_title('Amplitude Evolution $|a(z)|$')
    ax1.set_xlabel('z')
    ax1.set_ylabel('Magnitude')
    ax1.legend()

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(z, np.real(a_z), 'b-', label='Re[$a(z)$]')
    ax2.plot(z, np.imag(a_z), 'g--', label='Im[$a(z)$]')
    ax2.set_title('Real and Imaginary Parts')
    ax2.set_xlabel('z')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    
    # Plot g^(1)(tau)
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(tau, np.abs(g1), 'm-')
    ax3.set_title('First-Order Coherence $|g^{(1)}(\\tau)|$')
    ax3.set_xlabel('$\\tau$ (Delay)')
    ax3.set_ylabel('Magnitude')
    ax3.set_xlim(-p['z_max']/2, p['z_max']/2)

    # Plot Power Spectrum
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(omega, S_omega, 'c-')
    ax4.set_title('Power Spectrum $S(\\omega)$')
    ax4.set_xlabel('$\\omega$ (Frequency)')
    ax4.set_ylabel('Power')
    ax4.set_xlim(-20, 20)

    # Plot g^(2)(tau)
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(tau, g2, 'k-')
    ax5.set_title('Second-Order Coherence $g^{(2)}(\\tau)$')
    ax5.set_xlabel('$\\tau$ (Delay)')
    ax5.set_ylabel('Value')
    ax5.set_xlim(-p['z_max']/2, p['z_max']/2)
    ax5.axhline(1, ls='--', color='gray', label='Coherent (g$^{(2)}$=1)')
    ax5.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def run_parameter_sweep(base_params, sweep_ranges):
    """
    Runs the simulation over a range of parameters and visualizes the results.
    """
    k_range = sweep_ranges['k_j_imag_range']
    delta_range = sweep_ranges['Delta_s_range']
    
    results_grid = np.zeros((len(delta_range), len(k_range)))

    print("Starting parameter sweep...")
    # --- Loop over all parameter combinations ---
    for i, delta_s in enumerate(delta_range):
        for j, k_imag in enumerate(k_range):
            # Create a copy of the params and update with current sweep values
            current_params = base_params.copy()
            current_params['Delta_s'] = delta_s
            current_params['k_j_imag'] = k_imag
            
            # Run the calculation and store the result
            results_grid[i, j] = calculate_g2_peak(current_params)
        
        # Progress indicator
        progress = (i + 1) / len(delta_range) * 100
        print(f"Progress: {progress:.1f}%")

    # --- Plotting the Heatmap ---
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        results_grid,
        extent=[k_range[0], k_range[-1], delta_range[0], delta_range[-1]],
        aspect='auto',
        origin='lower',
        interpolation='nearest',
        cmap='inferno' # 'hot', 'viridis', 'plasma' are also good
    )

    fig.colorbar(im, ax=ax, label='Peak $g^{(2)}(0)$ Value')
    ax.set_title('Parameter Sweep: Photon Bunching $g^{(2)}(0)$')
    ax.set_xlabel('Coupling Strength ($k_{j, imag}$)')
    ax.set_ylabel('Detuning ($\\Delta_s$)')
    
    plt.show()
    return results_grid
def calculate_g2_peak(p):
    """
    Core calculation function for a single set of parameters.
    This function calculates and returns the peak value of g^(2)(0).
    It does not generate any plots.
    """
    # --- 1. Calculate Deterministic Part ---
    k_j = p['k_j_real'] + 1j * p['k_j_imag']
    chi_s = -p['alpha_s']/2 + 1j * p['Delta_s']
    chi_m = -p['alpha_m']/2
    A = (chi_s + chi_m) / 2
    D = np.lib.scimath.sqrt((chi_s - chi_m)**2 + 4 * k_j * np.conj(k_j))
    lambda_p = A + D/2
    lambda_m = A - D/2
    epsilon = 1e-15
    P_denom = chi_s - A + D/2
    Q_denom = chi_s - A - D/2
    P = k_j / (P_denom + epsilon)
    Q = k_j / (Q_denom + epsilon)
    L = 1 / (Q - P + epsilon)

    z = np.linspace(0, p['z_max'], p['num_points'])
    term1 = (-P * np.exp(lambda_p * z) + Q * np.exp(lambda_m * z)) * p['a0']
    term2 = (P * Q * (-np.exp(lambda_p * z) + np.exp(lambda_m * z))) * p['b0_dagger']
    a_z_deterministic = L * (term1 + term2)

    # --- 2. Ensemble Averaging for Statistics ---
    ensemble_g2_numerator_at_zero = 0
    ensemble_mean_I_sq = 0

    for _ in range(p['num_ensemble']):
        real_noise = np.random.normal(0, p['noise_strength'], p['num_points'])
        imag_noise = np.random.normal(0, p['noise_strength'], p['num_points'])
        stochastic_noise = real_noise + 1j * imag_noise
        a_z_total = a_z_deterministic + stochastic_noise
        I_z = np.abs(a_z_total)**2
        
        # We only need the value at tau=0 for the peak
        ensemble_g2_numerator_at_zero += np.mean(I_z**2)
        ensemble_mean_I_sq += np.mean(I_z)**2

    mean_g2_numerator = ensemble_g2_numerator_at_zero / p['num_ensemble']
    mean_denominator_g2 = ensemble_mean_I_sq / p['num_ensemble']

    if mean_denominator_g2 < epsilon:
        return 1.0 # Avoid division by zero, return coherent value

    g2_peak = mean_g2_numerator / mean_denominator_g2
    return g2_peak






# --- Run the main function ---
if __name__ == "__main__":
    #create_and_plot_thermal_state()
	#calculate_g2_for_thermal_mode()
	#calculate_g2_for_multi_mode()
    #calculate_g2_and_spectrum()
    #calculate_g2_and_spectrum_differentHilbertSpaces()
    #simulation_multiple_mode()
    # --- Main Parameters You Can Change ---
    # All parameters are collected here for easy modification.
    # --- Main Parameters You Can Change ---
    # These parameters are adjusted to isolate the thermal statistics.
    '''
    params = {
        'alpha_s': 0.1, 'alpha_m': 0.5,
        'k_j_real': 0,
        # Set a0 to be very small to make the deterministic signal negligible.
        'a0': 1e-5 + 0.0j, 
        'b0_dagger': 0.0 + 0.0j,
        'z_max': 50.0, 'num_points': 1024,
        # Set noise strength to a clear value.
        'noise_strength': 1.0,
        'num_ensemble': 50  # Reduced for faster sweeps
    }
    '''
    params = {
    # Physical parameters
    'alpha_s': 0.1,    # Loss/gain for 's' mode
    'alpha_m': 0.5,    # Loss/gain for 'm' mode
    'Delta_s': 2.0,    # Detuning for 's' mode
    'k_j_real': 0,     # Real part of the coupling k_j
    'k_j_imag': 1.5,     # Imaginary part of the coupling k_j

    # Initial conditions
    'a0': 1.0 + 0.0j,  # Initial amplitude of a_s,j
    'b0_dagger': 0.0 + 0.0j, # Initial amplitude of b_j_dagger

    # Simulation parameters
    'z_max': 50.0,     # Maximum z to simulate
    'num_points': 4096 # Number of points (power of 2 is good for FFT)
    }

    # --- Parameters to Sweep ---
    # Define the ranges for the parameters you want to explore.
    sweep_params = {
        'k_j_imag_range': np.linspace(0.1, 2.5, 20),
        'Delta_s_range': np.linspace(-3.0, 3.0, 20)
    }

    run_simulation(params)
    #run_parameter_sweep(params, sweep_params)