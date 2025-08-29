import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.fft import fft, fftshift, fftfreq

# A small constant to prevent division by zero
eps = 1e-20

# --- Simulation & Calculation Functions ---

def generate_single_mode(p):
    """
    Calculates the evolution of a single mode based on its parameters using an
    analytical solution.
    """
    from numpy.lib import scimath

    k_j = -1j * p['gtilde'] * p['A_p']
    alpha_m = p['alpha_p']
    chi_s = -p['alpha_s']/2 + 1j * p['Delta_s']
    chi_m = -alpha_m/2

    A = (chi_s + chi_m) / 2
    D = scimath.sqrt((chi_s - chi_m)**2 + 4 * k_j * np.conj(k_j))
    lambda_p = A + D/2
    lambda_m = A - D/2

    P_denom = chi_s - lambda_p
    Q_denom = chi_s - lambda_m
    
    P = k_j / (P_denom + eps)
    Q = k_j / (Q_denom + eps)
    L = 1 / (Q - P + eps)

    z = np.linspace(0, p['z_max'], p['num_points'])
    term1 = (-P * np.exp(lambda_p * z) + Q * np.exp(lambda_m * z)) * p['a0']
    term2 = (P * Q * (-np.exp(lambda_p * z) + np.exp(lambda_m * z))) * p['b0_dagger']
    a_z = L * (term1 + term2)
    
    return z, a_z

def compute_power_spectrum(x, z, zero_pad_factor=4, window=True):
    """
    Computes the power spectrum of a signal x by taking the Fourier
    Transform of the signal itself (the direct method).
    """
    dt = z[1] - z[0]
    N = len(x)
    Npad = int(2**np.ceil(np.log2(N * zero_pad_factor)))
    if window:
        xw = x * np.hanning(N)
    else:
        xw = x
    X = fftshift(fft(xw, n=Npad))
    S = np.abs(X)**2
    freq = fftshift(fftfreq(Npad, d=dt))
    omega = 2 * np.pi * freq
    return omega, S

def compute_g2_vs_tau(a1, a2):
    """
    Computes the second-order temporal coherence g^(2)(tau).
    """
    N = len(a1)
    I1 = np.abs(a1)**2
    I2 = np.abs(a2)**2
    taus = np.arange(-N + 1, N)
    g2 = np.zeros(len(taus), dtype=float)
    norm = np.mean(I1) * np.mean(I2) + eps
    for i, tau in enumerate(taus):
        if tau >= 0:
            x1 = I1[:N - tau]
            x2 = I2[tau:]
        else:
            x1 = I1[-tau:]
            x2 = I2[:N + tau]
        g2[i] = np.mean(x1 * x2) / norm
    return taus, g2


def load_and_process_lorentzian_fit_data(filepath):
    """
    Loads the pre-fitted Lorentzian curve data from the specified CSV file.
    """
    print(f"Loading Lorentzian fit data from {filepath}...")
    df = pd.read_csv(filepath)
    # The columns are 'fit_power_dBm' and 'frequency_Hz'
    power_dbm = df['fit_power_dBm'].values
    freq_hz = df['frequency_Hz'].values
    freq_mhz = freq_hz / 1e6
    print("Lorentzian fit data loaded and processed successfully.")
    return freq_mhz, power_dbm

# --- Main Execution Block ---
if __name__ == '__main__':
    
    # =========================================================================
    # --- 1. SET YOUR PARAMETERS HERE ---
    # =========================================================================
    # Enter the parameters you want to test. You can copy the 'p_opt' (from fitWatchdbm.py)
    # values from a successful fit here to validate the result.

    test_params = {
        'alpha_s1': 10.0000, 'alpha_s2': 3.3366, 'alpha_s3': 10.0000,
        'alpha_p1': 7.5395, 'alpha_p2': 4.6094, 'alpha_p3': 9.1923,
        'Delta_s1': 15.5236, 'Delta_s2': 8.2157, 'Delta_s3': 21.2983,
        'gtilde1': 2.9253, 'gtilde2': 2.0039, 'gtilde3': 2.7371,
        'A_p_common': 2.0802,
        'a0_1_real': -2.6868, 'a0_1_imag': 2.4174,
        'a0_2_real': 0.6793, 'a0_2_imag': -1.8937,
        'a0_3_real': -0.0886, 'a0_3_imag': 0.8508,
        'b0_dagger1_real': -4.5544, 'b0_dagger1_imag': 3.8397,
        'b0_dagger2_real': 0.1064, 'b0_dagger2_imag': -2.5901,
        'b0_dagger3_real': 1.3072, 'b0_dagger3_imag': 4.5666,
        'dbm_offset': -15.9456 
    }
    
    # --- 2. LOAD THE TARGET LORENTZIAN CURVE ---
    filepath = '/home/apolloin/Desktop/Brillouin_Scattering/Brillouin-scattering-Project/QuTip_Simulation/heterodyne_fit_curve.csv'
    try:
        target_freq_mhz, target_power_dbm = load_and_process_lorentzian_fit_data(filepath)
    except FileNotFoundError:
        print(f"ERROR: The file was not found at '{filepath}'.")
        print("Please update the 'filepath' variable with the correct location.")
        exit()
        
    # --- 3. CALCULATE THE THEORETICAL SPECTRUM FOR THE GIVEN PARAMETERS ---
    print("Calculating theoretical spectrum for the specified parameters...")

    # Unpack parameters from the dictionary
    p = test_params
    a0_1 = p['a0_1_real'] + 1j * p['a0_1_imag']
    a0_2 = p['a0_2_real'] + 1j * p['a0_2_imag']
    a0_3 = p['a0_3_real'] + 1j * p['a0_3_imag']
    b0_dagger1 = p['b0_dagger1_real'] + 1j * p['b0_dagger1_imag']
    b0_dagger2 = p['b0_dagger2_real'] + 1j * p['b0_dagger2_imag']
    b0_dagger3 = p['b0_dagger3_real'] + 1j * p['b0_dagger3_imag']

    params1 = {'A_p': p['A_p_common'], 'z_max': 50.0, 'num_points': 8192, 'alpha_s': p['alpha_s1'], 'alpha_p': p['alpha_p1'], 'Delta_s': p['Delta_s1'], 'a0': a0_1, 'gtilde': p['gtilde1'], 'b0_dagger': b0_dagger1}
    params2 = {'A_p': p['A_p_common'], 'z_max': 50.0, 'num_points': 8192, 'alpha_s': p['alpha_s2'], 'alpha_p': p['alpha_p2'], 'Delta_s': p['Delta_s2'], 'a0': a0_2, 'gtilde': p['gtilde2'], 'b0_dagger': b0_dagger2}
    params3 = {'A_p': p['A_p_common'], 'z_max': 50.0, 'num_points': 8192, 'alpha_s': p['alpha_s3'], 'alpha_p': p['alpha_p3'], 'Delta_s': p['Delta_s3'], 'a0': a0_3, 'gtilde': p['gtilde3'], 'b0_dagger': b0_dagger3}

    z1, a1 = generate_single_mode(params1)
    z2, a2 = generate_single_mode(params2)
    z3, a3 = generate_single_mode(params3)
    z = z1
    a_total = a1 + a2 + a3
    
    z_to_ns = 12.5 / np.pi
    
    # --- Calculate Power Spectrum (Direct Method) ---
    omega_sim, S_sim_arb = compute_power_spectrum(a_total, z, zero_pad_factor=8)
    freq_sim_mhz = (omega_sim / (2 * np.pi)) * (1000 / z_to_ns)
    
    S_sim_dbm = 10 * np.log10(S_sim_arb + eps) + p['dbm_offset']
    interp_func = interp1d(freq_sim_mhz, S_sim_dbm, kind='linear', bounds_error=False, fill_value=np.min(S_sim_dbm))
    sim_power_dbm_interp = interp_func(target_freq_mhz)

    print("Calculation complete.")

    # --- 4. CALCULATE G2 COHERENCE ---
    print("Calculating second-order coherence g2(tau)...")
    taus, g2_total = compute_g2_vs_tau(a_total, a_total)
    delay_ns = (taus * (z[1] - z[0])) * z_to_ns


    # --- 5. PLOT THE RESULTS ---
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plot 1: Power Spectrum
    plt.figure(figsize=(14, 8))
    plt.plot(target_freq_mhz, target_power_dbm, '.', label='Target Lorentzian Curve', alpha=0.5)
    plt.plot(target_freq_mhz, sim_power_dbm_interp, 'r-', label='Calculated Theoretical Spectrum', linewidth=2)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dBm)')
    plt.title('Target Lorentzian Curve vs. Calculated Theoretical Spectrum')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=np.min(target_power_dbm) - 5, top=np.max(target_power_dbm) + 5)

    # Plot 2: g2 Function
    plt.figure(figsize=(12, 6))
    plt.plot(delay_ns, g2_total, label='g$^{(2)}$(a_total, a_total)')
    plt.xlabel('Delay τ (ns)')
    plt.ylabel('g$^{(2)}$(τ)')
    plt.title('Second-Order Coherence for Calculated Spectrum')
    plt.grid(True)
    plt.xlim(-20, 20) # A reasonable zoom for g2

    # --- Add this section to investigate intensity ---
    print("Investigating intensity profile...")
    intensity_total = np.abs(a_total)**2

    plt.figure(figsize=(12, 6))
    plt.plot(z, intensity_total)
    plt.xlabel('Simulation Length z (arbitrary units)')
    plt.ylabel('Intensity |a_total|^2 (arbitrary units)')
    plt.title('Intensity Profile of the Total Field')
    plt.grid(True)
    plt.show()
