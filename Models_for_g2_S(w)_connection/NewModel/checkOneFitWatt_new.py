'''
Function that was used to check one particular parameter combination after running the fitWatchWatt_new.py script.
'''



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.fft import fft, fftshift, fftfreq

# A small constant to prevent division by zero
eps = 1e-20

# --- Core Simulation Functions (Copied from your main script for consistency) ---

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

def compute_power_spectrum(x, z, zero_pad_factor=8, window=True):
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


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- 1. SET YOUR TARGET DATA FILE ---
    # This is the experimental data you want to plot against.
    target_data_filepath = '/home/apolloin/Desktop/Brillouin_Scattering/Brillouin-scattering-Project/Models_for_g2_S(w)_connection/NewModel/fit_lorentzian_1_watt.csv'

    # --- 2. MANUALLY DEFINE YOUR PARAMETERS HERE ---
    # I've pre-filled this with the initial 'p0' values from your last fitting script as an example.
    # Change these values to test any combination you like.

    manual_params_dict = {
        'alpha_s': 31.735,
        'alpha_p': 36.134,
        'Delta_s': 2.2122e+01,
        'Delta_m': 6.6977e+00,
        'gtilde': 4.7697,
        'A_p': 4.6790,
        'a0_real': 0,
        'a0_imag': 0,   
        'b0_d_real': 5.7993e-06,
        'b0_d_imag': -4.8773e-07,
        'scaling_factor': 6.8426e-11,
        'vertical_shift': 1.2472e-09
    }

    try:
        # --- 3. LOAD THE TARGET DATA ---
        print(f"Loading target data from {target_data_filepath}...")
        df = pd.read_csv(target_data_filepath)
        target_freq_mhz = df['frequency_Hz'].values / 1e6
        target_power_watt = df['fit_power_watt'].values
        print("Data loaded successfully.")

        # --- 4. RUN THE SIMULATION WITH YOUR MANUAL PARAMETERS ---
        print("\nRunning simulation with manual parameters...")
        sim_params = {
            'A_p': manual_params_dict['A_p'], 'z_max': 50.0, 'num_points': 8192,
            'alpha_s': manual_params_dict['alpha_s'], 'alpha_p': manual_params_dict['alpha_p'],
            'Delta_s': manual_params_dict['Delta_s'], 'Delta_m': manual_params_dict['Delta_m'],
            'a0': manual_params_dict['a0_real'] + 1j * manual_params_dict['a0_imag'],
            'gtilde': manual_params_dict['gtilde'],
            'b0_dagger': manual_params_dict['b0_d_real'] + 1j * manual_params_dict['b0_d_imag']
        }
        z, a_z = generate_single_mode(sim_params)

        
        # --- 5. COMPUTE, SCALE, AND SHIFT THE SPECTRUM ---
        z_to_ns = 12.5 / np.pi
        omega_sim, S_sim_arb = compute_power_spectrum(a_z, z, zero_pad_factor=8)
        freq_sim_mhz = (omega_sim / (2 * np.pi)) * (1000 / z_to_ns)

        # Apply the scaling and shift
        S_sim_watt_raw = (S_sim_arb * manual_params_dict['scaling_factor']) + manual_params_dict['vertical_shift']
        
        # --- 6. INTERPOLATE ONTO THE TARGET FREQUENCY AXIS ---
        interp_func = interp1d(freq_sim_mhz, S_sim_watt_raw, kind='linear', bounds_error=False, fill_value=np.min(S_sim_watt_raw))
        S_sim_watt_final = interp_func(target_freq_mhz)
        print("Simulation and processing complete.")

        # --- 7. PLOT THE RESULTS ---
        print("Generating plot...")
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(14, 8))
        
        plt.plot(target_freq_mhz, target_power_watt, '.', label='Target Experimental Data', alpha=0.7)
        plt.plot(target_freq_mhz, S_sim_watt_final, 'r-', label='Manually Generated Spectrum', linewidth=2)
        
        plt.title('Manual Simulation vs. Experimental Data')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power (W)')
        plt.legend()
        plt.grid(True)
        
        # Add a text box to display the parameters used for the plot
        param_text = '\n'.join([f'{k}: {v:.4e}' for k, v in manual_params_dict.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gca().text(0.02, 0.98, param_text, transform=plt.gca().transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)
        
        plt.show()

    except FileNotFoundError:
        print(f"ERROR: The file was not found at '{target_data_filepath}'.")
        print("Please update the 'target_data_filepath' variable in the script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
