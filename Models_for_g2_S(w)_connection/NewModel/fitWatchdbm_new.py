'''
Honestly, this script is very drafty. Basically, it is used to find the fit parameters by manually changing it as the solver is 
not able to fit the parameters by itself. Was able to fit them luckily - hopefully got the right ones too.

This was done in dbm - linear scale.

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

# --- Data Loading (Unchanged) ---

def load_multiple_datasets(filepaths):
    """
    Loads curve data from a list of CSV files.
    """
    datasets = []
    print("Loading multiple datasets...")
    for i, filepath in enumerate(filepaths):
        print(f"  - Loading Mode {i+1} data from {filepath}...")
        df = pd.read_csv(filepath)
        power_dbm = df['fit_power_dBm'].values
        freq_hz = df['frequency_Hz'].values
        freq_mhz = freq_hz / 1e6
        datasets.append((freq_mhz, power_dbm))
    print("All datasets loaded successfully.")
    return datasets

# --- NEW REFACTORED FITTING MODEL ---

def simulate_and_interpolate_spectrum(target_freq_mhz, params):
    """
    Core simulation function that runs the model for a given set of parameters
    and returns the power spectrum interpolated onto the target frequencies.
    """
    # Step 1: Calculate the field evolution a(z)
    z, a_z = generate_single_mode(params)

    # Step 2: Calculate the power spectrum S(w)
    z_to_ns = 12.5 / np.pi
    omega_sim, S_sim_arb = compute_power_spectrum(a_z, z, zero_pad_factor=8)
    freq_sim_mhz = (omega_sim / (2 * np.pi)) * (1000 / z_to_ns)

    # Step 3: Interpolate S(w) onto the target frequencies from the data file
    S_sim_dbm = 10 * np.log10(S_sim_arb + eps)
    interp_func = interp1d(freq_sim_mhz, S_sim_dbm, kind='linear', bounds_error=False, fill_value=np.min(S_sim_dbm))
    S_interpolated_dbm = interp_func(target_freq_mhz)
    
    return S_interpolated_dbm

def create_fitting_model_with_progress(exp_power_data, param_names, mode_index):
    """
    A wrapper function that creates the model for fitting a SINGLE mode.
    This function now calls the cleaner `simulate_and_interpolate_spectrum`.
    """
    iteration_counter = [0] # Use a list to make it mutable inside the closure

    def model_for_fitting(freq_mhz, *fit_params):
        """
        This is the function that `curve_fit` calls. It unpacks parameters,
        runs the simulation, and returns the final model output for comparison.
        """
        # Unpack the fitting parameters
        (
            alpha_s, alpha_p, Delta_s, Delta_m, gtilde, A_p,
            a0_real, a0_imag, b0_dagger_real, b0_dagger_imag,
            dbm_offset
        ) = fit_params

        # Assemble the dictionary of physical parameters for the simulation
        params = {
            'A_p': A_p, 'z_max': 50.0, 'num_points': 8192,
            'alpha_s': alpha_s, 'alpha_p': alpha_p, 'Delta_s': Delta_s,
            'Delta_m': Delta_m, 'a0': a0_real + 1j * a0_imag, 
            'gtilde': gtilde, 'b0_dagger': b0_dagger_real + 1j * b0_dagger_imag
        }

        # Run the core simulation and get the interpolated spectrum
        S_interpolated_raw_db = simulate_and_interpolate_spectrum(freq_mhz, params)
        
        # Apply the vertical offset to match the data's power level
        final_model_output = S_interpolated_raw_db + dbm_offset
        
        # --- Progress Printing ---
        iteration_counter[0] += 1
        if iteration_counter[0] % 100 == 0:
            error = np.sum((final_model_output - exp_power_data)**2)
            print(f"\n--- [Mode {mode_index+1}] Iteration: {iteration_counter[0]} | Current Error: {error:.2f} ---")
            current_params = dict(zip(param_names, fit_params))
            for i, (name, val) in enumerate(current_params.items()):
                print(f"{name:>15s}: {val:<8.4f}", end='   | ')
                if (i + 1) % 4 == 0: print()
            print("\n" + "-"*80)

        return final_model_output

    return model_for_fitting

# --- Main Execution Block ---
if __name__ == '__main__':
    # File paths for the datasets to be loaded
    mode_data_filepaths = [
        '/home/apolloin/Desktop/Brillouin_Scattering/Brillouin-scattering-Project/QuTip_Simulation/fit_lorentzian_3.csv'
    ]

    try:
        # 1. Load all datasets
        mode_datasets = load_multiple_datasets(mode_data_filepaths)
        
        all_optimized_params = []
        plt.style.use('seaborn-v0_8-darkgrid')

        # 2. Loop through each dataset and perform a separate fit for each mode
        for i, (fit_freq_mhz, fit_power_dbm) in enumerate(mode_datasets):
            print(f"\n{'='*25} FITTING MODE {i+1} {'='*25}")

            peak_freq_mhz = fit_freq_mhz[np.argmax(fit_power_dbm)]
            delta_s_center_guess = peak_freq_mhz / 40.0

            # Define initial guesses (p0) and bounds for a SINGLE mode
            param_names = [
                'alpha_s', 'alpha_p', 'Delta_s', 'Delta_m', 'gtilde', 'A_p',
                'a0_real', 'a0_imag', 'b0_d_real', 'b0_d_imag', 'dbm_offset'
            ]

            p0 = [32.5097, 34.5938, 21.320, 5.121, 4.7014, 4.6790, 0, 0, 50.2304, -80.9766, -30.8691]
            bounds = ([0,  0  ,        0,        0,         0,   4.6790,     0,        0,    -np.inf,  -np.inf,   -np.inf],
                      [50, 50,         28,        25,    10,     4.6791,  0.00001,  0.0001,  np.inf,   np.inf,    np.inf])

            # Create and run the fitting model for the current mode
            model_to_fit = create_fitting_model_with_progress(fit_power_dbm, param_names, i)
            p_opt, p_cov = curve_fit(model_to_fit, fit_freq_mhz, fit_power_dbm, p0=p0, bounds=bounds, max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)

            print(f"\n--- Optimized Parameters for Mode {i+1} ---")
            for name, val in zip(param_names, p_opt):
                print(f"{name:>15s}: {val:.4f}")
            all_optimized_params.append(p_opt)

            # Plot the individual fit for this mode
            fit_power_dbm_final = model_to_fit(fit_freq_mhz, *p_opt)
            plt.figure(figsize=(12, 7))
            plt.plot(fit_freq_mhz, fit_power_dbm, '.', label=f'Target Data for Mode {i+1}')
            plt.plot(fit_freq_mhz, fit_power_dbm_final, 'r-', label=f'Fitted Model for Mode {i+1}', linewidth=2)
            plt.title(f'Individual Fit for Mode {i+1}')
            plt.xlabel('Frequency (MHz)')
            plt.ylabel('Power (dBm)')
            plt.legend()
            plt.grid(True)
            plt.show()

        # 3. After all fits are complete, combine the results for a final plot
        print("\n\n--- Generating Final Combined Plot ---")
        
        # Generate the three modes using their individually optimized parameters
        z_final = None
        a_components = []
        for i, p_opt in enumerate(all_optimized_params):
            (alpha_s, alpha_p, Delta_s, Delta_m, gtilde, A_p, 
             a0_real, a0_imag, b0_d_real, b0_d_imag, _) = p_opt
            
            params = {
                'A_p': A_p, 'z_max': 50.0, 'num_points': 8192,
                'alpha_s': alpha_s, 'alpha_p': alpha_p, 'Delta_s': Delta_s,
                'Delta_m': Delta_m, 'a0': a0_real + 1j * a0_imag, 'gtilde': gtilde,
                'b0_dagger': b0_d_real + 1j * b0_d_imag
            }
            z, a_z = generate_single_mode(params)
            if z_final is None: z_final = z
            a_components.append(a_z)
        
        # Sum the modes and calculate the final power spectrum
        a_total_final = sum(a_components)
        z_to_ns = 12.5 / np.pi
        omega_sim, S_sim_arb = compute_power_spectrum(a_total_final, z_final, zero_pad_factor=8)
        freq_sim_mhz = (omega_sim / (2 * np.pi)) * (1000 / z_to_ns)
        
        # For the offset, we can average the fitted offsets or use the one from the dominant mode
        final_dbm_offset = np.mean([p[-1] for p in all_optimized_params])
        S_total_dbm_raw = 10 * np.log10(S_sim_arb + eps) + final_dbm_offset

        # --- FIX: Interpolate the final spectrum onto the original data's frequency axis ---
        # This ensures the final plot uses the same x-axis as the individual fit plot, making them identical for a single mode.
        # We use the frequencies from the first loaded dataset as our reference axis.
        final_plot_freq_mhz = mode_datasets[0][0]
        interp_func_final = interp1d(freq_sim_mhz, S_total_dbm_raw, kind='linear', bounds_error=False, fill_value=np.min(S_total_dbm_raw))
        S_total_dbm_interpolated = interp_func_final(final_plot_freq_mhz)


        # Final Plot: Original Data vs. Sum of Individually Fitted Modes
        plt.figure(figsize=(14, 8))
        # Plot the target data from the first file for comparison
        plt.plot(final_plot_freq_mhz, mode_datasets[0][1], '.', label=f'Target Data for Mode 1', alpha=0.6)
        # Plot the final, interpolated model result
        plt.plot(final_plot_freq_mhz, S_total_dbm_interpolated, 'r-', label='Sum of Individually Fitted Models', linewidth=2)
        plt.title('Final Combined Model vs. Target Data')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power (dBm)')
        plt.legend()
        plt.grid(True)
        plt.show()

    except FileNotFoundError as e:
        print(f"ERROR: A file was not found. Please check your filepaths.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


