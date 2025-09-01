'''
Honestly, this script is very drafty. Basically, it is used to find the fit parameters by manually changing it as the solver is 
not able to fit the parameters by itself. Was able to fit them luckily - hopefully got the right ones too.

This was done in Watt - linear scale.

'''





import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.fft import fft, fftshift, fftfreq
from scipy.interpolate import interp1d

# A small constant to prevent division by zero
eps = 1e-20

# --- Core Simulation Functions (Unchanged) ---

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

# --- Data Loading (Updated for Watts) ---

def load_multiple_datasets(filepaths):
    """
    Loads curve data from a list of CSV files.
    """
    datasets = []
    print("Loading multiple datasets...")
    for i, filepath in enumerate(filepaths):
        print(f"  - Loading Mode {i+1} data from {filepath}...")
        df = pd.read_csv(filepath)
        power_watt = df['fit_power_watt'].values
        freq_hz = df['frequency_Hz'].values
        freq_mhz = freq_hz / 1e6
        datasets.append((freq_mhz, power_watt))
    print("All datasets loaded successfully.")
    return datasets

# --- REFACTORED FITTING MODEL (Updated for Watts) ---

def simulate_and_interpolate_spectrum(target_freq_mhz, params):
    """
    Core simulation function that returns the power spectrum in LINEAR, ARBITRARY units,
    interpolated onto the target frequencies.
    """
    z, a_z = generate_single_mode(params)
    z_to_ns = 12.5 / np.pi
    omega_sim, S_sim_arb = compute_power_spectrum(a_z, z, zero_pad_factor=8)
    freq_sim_mhz = (omega_sim / (2 * np.pi)) * (1000 / z_to_ns)

    interp_func = interp1d(freq_sim_mhz, S_sim_arb, kind='linear', bounds_error=False, fill_value=np.min(S_sim_arb))
    S_interpolated_watt = interp_func(target_freq_mhz)
    
    return S_interpolated_watt

def create_fitting_model_with_progress(exp_power_data, param_names, mode_index):
    """
    A wrapper function that creates the model for fitting a SINGLE mode in linear units (Watts).
    """
    iteration_counter = [0] 

    def model_for_fitting(freq_mhz, *fit_params):
        """
        This is the function that `curve_fit` calls.
        """
        # --- CHANGE: Unpack 'vertical_shift' as well ---
        (
            alpha_s, alpha_p, Delta_s, Delta_m, gtilde, A_p,
            a0_real, a0_imag, b0_dagger_real, b0_dagger_imag,
            scaling_factor, vertical_shift
        ) = fit_params

        params = {
            'A_p': A_p, 'z_max': 50.0, 'num_points': 8192,
            'alpha_s': alpha_s, 'alpha_p': alpha_p, 'Delta_s': Delta_s,
            'Delta_m': Delta_m, 'a0': a0_real + 1j * a0_imag, 
            'gtilde': gtilde, 'b0_dagger': b0_dagger_real + 1j * b0_dagger_imag
        }

        S_interpolated_arb = simulate_and_interpolate_spectrum(freq_mhz, params)
        
        # --- CHANGE: Apply scaling factor and then add vertical shift for noise floor ---
        final_model_output = (S_interpolated_arb * scaling_factor) + vertical_shift
        
        # --- Progress Printing ---
        iteration_counter[0] += 1
        if iteration_counter[0] % 100 == 0:
            error = np.sum((final_model_output - exp_power_data)**2)
            print(f"\n--- [Mode {mode_index+1}] Iteration: {iteration_counter[0]} | Current Error: {error:.4e} ---")
            current_params = dict(zip(param_names, fit_params))
            for i, (name, val) in enumerate(current_params.items()):
                print(f"{name:>15s}: {val:<10.4e}", end='   | ')
                if (i + 1) % 4 == 0: print()
            print("\n" + "-"*80)

        return final_model_output

    return model_for_fitting

# --- Main Execution Block ---
if __name__ == '__main__':
    mode_data_filepaths = [
        '/home/apolloin/Desktop/Brillouin_Scattering/Brillouin-scattering-Project/Models_for_g2_S(w)_connection/NewModel/fit_lorentzian_3_watt.csv'
    ]

    try:
        mode_datasets = load_multiple_datasets(mode_data_filepaths)
        all_optimized_params = []
        plt.style.use('seaborn-v0_8-darkgrid')

        for i, (fit_freq_mhz, fit_power_watt) in enumerate(mode_datasets):
            print(f"\n{'='*25} FITTING MODE {i+1} {'='*25}")

            peak_freq_mhz = fit_freq_mhz[np.argmax(fit_power_watt)]
            delta_s_center_guess = peak_freq_mhz / 40.0

            # --- CHANGE: Added 'vertical_shift' to the model ---
            param_names = [
                'alpha_s', 'alpha_p', 'Delta_s', 'Delta_m', 'gtilde', 'A_p',
                'a0_real', 'a0_imag', 'b0_d_real', 'b0_d_imag', 'scaling_factor',
                'vertical_shift'
            ]
            
            # --- CHANGE: Added intelligent guesses for new parameters ---
            peak_watt = np.max(fit_power_watt)
            noise_floor_guess = np.min(fit_power_watt)
            # Adjust scaling factor guess to account for the noise floor
            scaling_factor_guess = (peak_watt - noise_floor_guess) / 0.1 


            p0 = [31.735, 36.134, 2.2122e+01, 6.6977e+00, 4.7697, 4.6790, 0.0, 0.0, 5.7993e-01, -4.8773e-01, 6.8425e-11, 1.2472e-09]
            bounds = ([20, 20, 0, 0, 3, 4.6790, 0, 0, -50, -50, 6.8425e-11, -np.inf],
                      [50, 50, 30, 25, 10, 4.679001, 0.0001, 0.0001, 55, 50, 6.8426e-11, np.inf])

            model_to_fit = create_fitting_model_with_progress(fit_power_watt, param_names, i)
            p_opt, p_cov = curve_fit(model_to_fit, fit_freq_mhz, fit_power_watt, p0=p0, bounds=bounds, max_nfev=10000, ftol=1e-15, xtol=1e-15, gtol=1e-15)

            print(f"\n--- Optimized Parameters for Mode {i+1} ---")
            for name, val in zip(param_names, p_opt):
                print(f"{name:>15s}: {val:.4e}")
            all_optimized_params.append(p_opt)

            fit_power_final_watt = model_to_fit(fit_freq_mhz, *p_opt)
            plt.figure(figsize=(12, 7))
            plt.plot(fit_freq_mhz, fit_power_watt, '.', label=f'Target Data for Mode {i+1}')
            plt.plot(fit_freq_mhz, fit_power_final_watt, 'r-', label=f'Fitted Model for Mode {i+1}', linewidth=2)
            plt.title(f'Individual Fit for Mode {i+1}')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Power (W)')
            plt.legend()
            plt.grid(True)
            plt.show()

        # --- Final Plot Section (Updated for Watts) ---
        print("\n\n--- Generating Final Combined Plot ---")
        
        z_final = None
        a_components = []
        final_scaling_factors = []
        final_vertical_shifts = []

        for i, p_opt in enumerate(all_optimized_params):
            # Use a dictionary to unpack safely
            p_dict = dict(zip(param_names, p_opt))
            
            params = {
                'A_p': p_dict['A_p'], 'z_max': 50.0, 'num_points': 8192,
                'alpha_s': p_dict['alpha_s'], 'alpha_p': p_dict['alpha_p'], 
                'Delta_s': p_dict['Delta_s'], 'Delta_m': p_dict['Delta_m'], 
                'a0': p_dict['a0_real'] + 1j * p_dict['a0_imag'], 'gtilde': p_dict['gtilde'],
                'b0_dagger': p_dict['b0_d_real'] + 1j * p_dict['b0_d_imag']
            }
            z, a_z = generate_single_mode(params)
            if z_final is None: z_final = z
            a_components.append(a_z)
            final_scaling_factors.append(p_dict['scaling_factor'])
            final_vertical_shifts.append(p_dict['vertical_shift'])
        
        a_total_final = sum(a_components)
        z_to_ns = 12.5 / np.pi
        omega_sim, S_sim_arb = compute_power_spectrum(a_total_final, z_final, zero_pad_factor=8)
        freq_sim_mhz = (omega_sim / (2 * np.pi)) * (1000 / z_to_ns)
        
        # --- CHANGE: Apply both scaling factor and vertical shift ---
        final_scaling_factor = np.mean(final_scaling_factors)
        final_vertical_shift = np.mean(final_vertical_shifts)
        S_total_watt_raw = (S_sim_arb * final_scaling_factor) + final_vertical_shift

        final_plot_freq_mhz = mode_datasets[0][0]
        interp_func_final = interp1d(freq_sim_mhz, S_total_watt_raw, kind='linear', bounds_error=False, fill_value=np.min(S_total_watt_raw))
        S_total_watt_interpolated = interp_func_final(final_plot_freq_mhz)

        plt.figure(figsize=(14, 8))
        plt.plot(final_plot_freq_mhz, mode_datasets[0][1], '.', label=f'Target Data for Mode 1', alpha=0.6)
        plt.plot(final_plot_freq_mhz, S_total_watt_interpolated, 'r-', label='Sum of Individually Fitted Models', linewidth=2)
        plt.title('Final Combined Model vs. Target Data')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power (W)')
        plt.legend()
        plt.grid(True)
        #plt.show()

    except FileNotFoundError as e:
        print(f"ERROR: A file was not found. Please check your filepaths.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

