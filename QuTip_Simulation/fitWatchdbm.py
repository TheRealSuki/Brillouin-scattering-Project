import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.fft import fft, fftshift, fftfreq
from scipy.interpolate import interp1d

# A small constant to prevent division by zero
eps = 1e-20

# --- User's Original Simulation Functions ---

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

# --- New/Modified Functions for Data Loading and Fitting ---

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

def create_fitting_model_with_progress(exp_power_data, param_names):
    """
    A wrapper function that creates the model for fitting and includes
    a progress printout on each iteration.
    """
    iteration_counter = [0] # Use a list to make it mutable inside the closure

    def model_for_fitting(freq_mhz, *fit_params):
        """
        This is the core function that `curve_fit` will use.
        """
        # --- Unpack all the fitting parameters ---
        (
            alpha_s1, alpha_s2, alpha_s3,
            alpha_p1, alpha_p2, alpha_p3,
            Delta_s1, Delta_s2, Delta_s3,
            gtilde1, gtilde2, gtilde3,
            A_p_common,
            a0_1_real, a0_1_imag,
            a0_2_real, a0_2_imag,
            a0_3_real, a0_3_imag,
            b0_dagger1_real, b0_dagger1_imag,
            b0_dagger2_real, b0_dagger2_imag,
            b0_dagger3_real, b0_dagger3_imag,
            dbm_offset
        ) = fit_params

        # --- Set up simulation parameters based on fit values ---
        z_to_ns = 12.5 / np.pi
        
        a0_1 = a0_1_real + 1j * a0_1_imag
        a0_2 = a0_2_real + 1j * a0_2_imag
        a0_3 = a0_3_real + 1j * a0_3_imag
        b0_dagger1 = b0_dagger1_real + 1j * b0_dagger1_imag
        b0_dagger2 = b0_dagger2_real + 1j * b0_dagger2_imag
        b0_dagger3 = b0_dagger3_real + 1j * b0_dagger3_imag

        params1 = {'A_p': A_p_common, 'z_max': 50.0, 'num_points': 8192, 'alpha_s': alpha_s1, 'alpha_p': alpha_p1, 'Delta_s': Delta_s1, 'a0': a0_1, 'gtilde': gtilde1, 'b0_dagger': b0_dagger1}
        params2 = {'A_p': A_p_common, 'z_max': 50.0, 'num_points': 8192, 'alpha_s': alpha_s2, 'alpha_p': alpha_p2, 'Delta_s': Delta_s2, 'a0': a0_2, 'gtilde': gtilde2, 'b0_dagger': b0_dagger2}
        params3 = {'A_p': A_p_common, 'z_max': 50.0, 'num_points': 8192, 'alpha_s': alpha_s3, 'alpha_p': alpha_p3, 'Delta_s': Delta_s3, 'a0': a0_3, 'gtilde': gtilde3, 'b0_dagger': b0_dagger3}

        # --- Run the simulation ---
        z1, a1 = generate_single_mode(params1)
        z2, a2 = generate_single_mode(params2)
        z3, a3 = generate_single_mode(params3)
        
        z = z1
        a_total = a1 + a2 + a3

        # --- Calculate Power Spectrum (Direct Method) ---
        omega_sim, S_sim_arb = compute_power_spectrum(a_total, z, zero_pad_factor=8)
        freq_sim_mhz = (omega_sim / (2 * np.pi)) * (1000 / z_to_ns)

        # --- Scale and Interpolate ---
        S_sim_dbm = 10 * np.log10(S_sim_arb + eps) + dbm_offset
        interp_func = interp1d(freq_sim_mhz, S_sim_dbm, kind='linear', bounds_error=False, fill_value=np.min(S_sim_dbm))
        S_interpolated_dbm = interp_func(freq_mhz)
        
        # --- Progress Printing ---
        iteration_counter[0] += 1
        if iteration_counter[0] % 100 == 0:
            error = np.sum((S_interpolated_dbm - exp_power_data)**2)
            print(f"\n--- Iteration: {iteration_counter[0]} | Current Error: {error:.2f} ---")
            
            current_params = dict(zip(param_names, fit_params))
            for i, (name, val) in enumerate(current_params.items()):
                print(f"{name:>15s}: {val:<8.4f}", end='   | ')
                if (i + 1) % 4 == 0:
                    print()
            print("\n" + "-"*80)

        return S_interpolated_dbm

    return model_for_fitting

# --- Main Execution Block ---
if __name__ == '__main__':
    filepath = '/home/apolloin/Desktop/Brillouin_Scattering/Brillouin-scattering-Project/QuTip_Simulation/heterodyne_fit_curve.csv'

    try:
        # 1. Load Data (from the Lorentzian fit CSV)
        fit_freq_mhz, fit_power_dbm = load_and_process_lorentzian_fit_data(filepath)
        
        peak_freq_index = np.argmax(fit_power_dbm)
        peak_freq_mhz = fit_freq_mhz[peak_freq_index]
        print(f"Lorentzian fit peak found at {peak_freq_mhz:.2f} MHz. Using this for initial guess.")

        delta_s_center_guess = peak_freq_mhz / 40.0

        # 2. Define Initial Guesses (p0) and Bounds for all variables
        param_names = [
            'alpha_s1', 'alpha_s2', 'alpha_s3', 'alpha_p1', 'alpha_p2', 'alpha_p3',
            'Delta_s1', 'Delta_s2', 'Delta_s3',
            'gtilde1', 'gtilde2', 'gtilde3', 'A_p_common',
            'a0_1_real', 'a0_1_imag', 'a0_2_real', 'a0_2_imag', 'a0_3_real', 'a0_3_imag',
            'b0_d1_real', 'b0_d1_imag', 'b0_d2_real', 'b0_d2_imag', 'b0_d3_real', 'b0_d3_imag',
            'dbm_offset'
        ]

        p0 = [
            0.08, 0.07, 0.07, 1.0, 1.0, 1.0,  # alphas
            delta_s_center_guess, delta_s_center_guess + 2.0, delta_s_center_guess + 4.0, # Deltas
            0.5, 0.5, 0.5,                   # gtildes
            0.1,                             # A_p
            0.09, 0.0, 0.01, 0.0, 0.005, 0.0, # a0s
            0.1, 0.0, 0.1, 0.0, 0.1, 0.0,     # b0s
            -80.0                            # dbm_offset guess
        ]

        lower_bounds = [0]*6 + [-50, -50, -50] + [0]*4 + [-5]*12 + [-200]
        upper_bounds = [10]*6 + [50, 50, 50] + [10]*4 + [5]*12 + [0]
        bounds = (lower_bounds, upper_bounds)

        # 3. Perform the Curve Fit
        print("\nStarting the fitting process (fitting model to Lorentzian curve)...")
        print(f"Fitting {len(p0)} parameters. This may be very slow and challenging to converge.")
        
        model_to_fit = create_fitting_model_with_progress(fit_power_dbm, param_names)

        p_opt, p_cov = curve_fit(
            model_to_fit, 
            fit_freq_mhz, 
            fit_power_dbm, 
            p0=p0, 
            bounds=bounds,
            max_nfev=10000 
        )
        print("Fitting complete!")

        # 4. Print and Analyze Results
        print("\n--- Optimized Parameters (Full Model vs. Lorentzian fit) ---")
        for name, val in zip(param_names, p_opt):
            print(f"{name:>20s}: {val:.4f}")

        # 5. Plot the Results
        fit_power_dbm_final = model_to_fit(fit_freq_mhz, *p_opt)

        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(14, 8))
        
        plt.plot(fit_freq_mhz, fit_power_dbm, '.', label='Target Lorentzian Curve', alpha=0.5)
        plt.plot(fit_freq_mhz, fit_power_dbm_final, 'r-', label='Fitted Theoretical Model', linewidth=2)
        
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power (dBm)')
        plt.title('Lorentzian Curve vs. Fitted Theoretical Model')
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=np.min(fit_power_dbm) - 5, top=np.max(fit_power_dbm) + 5)
        plt.show()

    except FileNotFoundError:
        print(f"ERROR: The file was not found at '{filepath}'.")
        print("Please update the 'filepath' variable in the script with the correct location.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

