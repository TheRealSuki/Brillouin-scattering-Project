'''
Function that was used to check one particular parameter combination after running the fitWatchdbm_new.py script.
'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, fftshift, fftfreq
from scipy.interpolate import interp1d

# A small constant to prevent division by zero
eps = 1e-20

# --- Core Simulation Functions (Copied from the main script) ---

def generate_single_mode(p):
    """
    Calculates the evolution of a single mode based on its parameters using an
    analytical solution.
    """
    from numpy.lib import scimath

    k_j = -1j * p['gtilde'] * p['A_p']
    alpha_m = p['alpha_p']
    chi_s = -p['alpha_s']/2 + 1j * p['Delta_s']
    chi_m = -alpha_m/2 - 1j * p['Delta_m']

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

def load_dataset(filepath):
    """
    Loads curve data from a single CSV file.
    """
    print(f"Loading target data from {filepath}...")
    df = pd.read_csv(filepath)
    power_dbm = df['fit_power_dBm'].values
    freq_hz = df['frequency_Hz'].values
    freq_mhz = freq_hz / 1e6
    print("Data loaded successfully.")
    return freq_mhz, power_dbm

# --- Main Execution Block ---
if __name__ == '__main__':
    # --- 1. USER INPUT SECTION ---
    
    target_data_filepath = '/home/apolloin/Desktop/Brillouin_Scattering/Brillouin-scattering-Project/Models_for_g2_S(w)_connection/OldModel/fit_lorentzian_3.csv'

    # Define the parameters for each mode you want to simulate.
    mode_params_list = [
        {
            # Mode 1 Parameters
            'alpha_s': 31.735, 
            'alpha_p': 39.134, 
            'Delta_s': 22, # Does oscillations in the line, small oscillations (can make it Lorentzian I guess?) shifts
            'Delta_m': 7.5,  # Does oscillations in the line, small oscillations (Lorentzain width I guess?) narrows up 
            'gtilde': 4.7697,  # Coupling strength
            'A_p': 4.6790,      # Pump amplitude
            'a0': 0,       # Initial Stokes field (complex), honestly it does not contribute anything
            'b0_dagger': 2.0439 + 1j * -2.0229, # Initial Phonon field (complex)
            'dbm_offset': -30.8691
        },
        # {
        #     # Mode 2 Parameters (Uncomment and edit to add a second mode)
        #     'alpha_s': 0.1, 
        #     'alpha_p': 1.0, 
        #     'Delta_s': 278.0 / 40.0,
        #     'Delta_m': 0.0, 
        #     'gtilde': 0.4, 
        #     'A_p': 0.1,
        #     'a0': 0.05 + 0.0j,
        #     'b0_dagger': 0.08 + 0.0j,
        #     'dbm_offset': -80.0
        # },
    ]

    # Simulation constants (usually don't need to be changed)
    simulation_constants = {'z_max': 50.0, 'num_points': 8192}

    # --- 2. EXECUTION & PLOTTING ---
    try:
        # Load the target experimental data
        target_freq_mhz, target_power_dbm = load_dataset(target_data_filepath)

        # Generate the field evolution for each mode using the specified parameters
        z_final = None
        a_components = []
        for i, p_manual in enumerate(mode_params_list):
            print(f"Simulating Mode {i+1}...")
            # Combine user params with simulation constants
            params = {**p_manual, **simulation_constants} 
            z, a_z = generate_single_mode(params)
            if z_final is None: z_final = z
            a_components.append(a_z)
        
        # Sum the modes and calculate the final power spectrum
        a_total_final = sum(a_components)
        z_to_ns = 12.5 / np.pi
        omega_sim, S_sim_arb = compute_power_spectrum(a_total_final, z_final, zero_pad_factor=8)
        freq_sim_mhz = (omega_sim / (2 * np.pi)) * (1000 / z_to_ns)
        
        # Use the offset from the first mode (or average them if you prefer)
        final_dbm_offset = mode_params_list[0]['dbm_offset']
        S_total_dbm_raw = 10 * np.log10(S_sim_arb + eps) + final_dbm_offset

        # Interpolate the final spectrum onto the experimental data's frequency axis for a direct comparison
        interp_func_final = interp1d(freq_sim_mhz, S_total_dbm_raw, kind='linear', bounds_error=False, fill_value=np.min(S_total_dbm_raw))
        S_total_dbm_interpolated = interp_func_final(target_freq_mhz)

        # --- Final Plot ---
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(14, 8))
        plt.plot(target_freq_mhz, target_power_dbm, '.', label='Target Experimental Data', alpha=0.6)
        plt.plot(target_freq_mhz, S_total_dbm_interpolated, 'r-', label='Simulated Model with Manual Parameters', linewidth=2)
        plt.title('Experimental Data vs. Manually Specified Model')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power (dBm)')
        plt.legend()
        plt.grid(True)
        plt.show()

    except FileNotFoundError:
        print(f"ERROR: The file was not found at '{target_data_filepath}'. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
