import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import argparse
from numpy.fft import fft, fftshift, fftfreq
from scipy.signal import find_peaks






# Define the Lorentzian function for a single peak
def lorentzian(x, x0, fwhm, A):
    """ A Lorentzian peak function defined by its FWHM. """
    gamma = fwhm / 2.0  # HWHM
    return A * (gamma**2 / ((x - x0)**2 + gamma**2))

# Define the model for the sum of three Lorentzians for curve_fit
def three_lorentzian_model(x, c1, f1, a1, c2, f2, a2, c3, f3, a3):
    """ A model consisting of the sum of three Lorentzian peaks. """
    return (lorentzian(x, c1, f1, a1) +
            lorentzian(x, c2, f2, a2) +
            lorentzian(x, c3, f3, a3))

def analyze_session3():
    """
    Loads experimental g^(2)(τ) data, calculates its power spectrum, and fits
    a multi-Lorentzian model based on a user's initial guess.
    """
    # =========================================================================
    # 1. Load and Process the Experimental Data to get the Power Spectrum
    # =========================================================================
    
    # --- File Paths and Data Selection ---
    folder_path = 'Measurements/Session_3_Photon_Statistics_And_Heterodyne/3_1GHz_187microWatt_slightly_lasing/data_photon_stat/'
    choice = input("Enter '1' for tx_1_ns_... data or '2' for tx_.1_ns_... data: ")
    
    if choice == '1':
        file_name = 'tx_1_ns_over_weekend_1.csv'
        time_conversion_factor = 1e-9
    elif choice == '2':
        file_name = 'tx_.1_ns_over_weekend_1.csv'
        time_conversion_factor = 1e-10
    else:
        print("Invalid choice.")
        return

    full_path = os.path.join(folder_path, file_name)
    total_power_watts = 187e-6
    
    print(f"Loading data from: {full_path}")
    try:
        data = pd.read_csv(full_path, header=0)
        data.columns = ['Time_ns', 'Counts']
    except FileNotFoundError:
        print(f"ERROR: The file was not found at {full_path}")
        return

    peak_index = data['Counts'].idxmax()
    zero_delay_time_ns = data.loc[peak_index, 'Time_ns']
    print(f"Centering data: Found g^(2)(0) peak at {zero_delay_time_ns:.2f} ns.")
    
    processed_data = data.loc[peak_index:].copy()
    processed_data['Time_ns'] = processed_data['Time_ns'] - zero_delay_time_ns

    baseline_window = int(len(processed_data) * 0.25) 
    baseline_counts = processed_data['Counts'][-baseline_window:].mean()
    g2_experimental = processed_data['Counts'] / baseline_counts
    time_s = processed_data['Time_ns'].values * time_conversion_factor
    
    # --- Calculate the Experimental Spectrum ---
    g2_fluctuations = g2_experimental - 1
    N_t = len(time_s)
    dt = np.mean(np.diff(time_s)) if N_t > 1 else time_conversion_factor
    S_w_au = fft(g2_fluctuations)
    wlist_Hz = fftfreq(N_t, dt)
    S_w_au_shifted = np.abs(fftshift(S_w_au))
    wlist_Hz_shifted = fftshift(wlist_Hz)
    
    # Keep only the positive frequency part for analysis
    positive_freq_mask = wlist_Hz_shifted >= 0
    exp_freq_hz = wlist_Hz_shifted[positive_freq_mask]
    exp_psd_au = S_w_au_shifted[positive_freq_mask]

    # =========================================================================
    # 2. DEFINE INITIAL GUESS FOR THE LORENTZIAN PEAKS
    # =========================================================================
    
    # --- Peak 1 Parameters ---
    center_1_hz = 0
    fwhm_1_hz   = 40e6
    amp_1_au    = 7
    
    # --- Peak 2 Parameters ---
    center_2_hz = 80e6
    fwhm_2_hz   = 40e6
    amp_2_au    = 2
    
    # --- Peak 3 Parameters ---
    center_3_hz = 160e6
    fwhm_3_hz   = 60e6
    amp_3_au    = 0.3

    # =========================================================================
    # 3. Perform Automated Fit
    # =========================================================================
    
    initial_p0 = [center_1_hz, fwhm_1_hz, amp_1_au, 
                  center_2_hz, fwhm_2_hz, amp_2_au, 
                  center_3_hz, fwhm_3_hz, amp_3_au]

    lower_bounds = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

    try:
        print("\nPerforming automated fit using your guess as a starting point...")
        popt, pcov = curve_fit(three_lorentzian_model, exp_freq_hz, exp_psd_au, 
                               p0=initial_p0, bounds=(lower_bounds, upper_bounds))
        fit_successful = True
        
        print("\n--- Optimized Fit Results ---")
        for i in range(3):
            c, f, a = popt[i*3:(i+1)*3]
            print(f"Peak {i+1}: Center = {c/1e6:.2f} MHz, FWHM = {f/1e6:.2f} MHz, Amplitude = {a:.2f}")
        print("-----------------------------\n")

    except RuntimeError:
        print("Automated fit failed. Your initial guess might be too far off.")
        fit_successful = False
        popt = initial_p0 # Use initial guess if fit fails

    # =========================================================================
    # 4. Calculate Final Spectra for Plotting
    # =========================================================================
    
    # Create the initial guess spectrum from parameters
    initial_guess_spectrum = three_lorentzian_model(exp_freq_hz, *initial_p0)
    
    # Create the final, optimized fit spectrum
    final_fit_spectrum = three_lorentzian_model(exp_freq_hz, *popt)
    
    # --- Convert to Physical Power Units ---
    integrated_power_au = np.trapz(exp_psd_au, exp_freq_hz)
    scaling_factor = total_power_watts / integrated_power_au
    psd_watts_hz = exp_psd_au * scaling_factor
    epsilon = 1e-30
    psd_dBm_hz = 10 * np.log10((psd_watts_hz + epsilon) / 1e-3)

    # =========================================================================
    # 5. Plot the Results in a 2x2 Grid
    # =========================================================================
    
    print("Plotting the results...")
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Full Analysis: From Photon Statistics to Fitted Spectrum', fontsize=18)

    # --- Plot 1: Processed g^(2)(τ) Data ---
    ax = axs[0, 0]
    ax.plot(processed_data['Time_ns'], g2_experimental, 'b-')
    ax.set_title(r'1. Processed Experimental Data $g^{(2)}(\tau)$')
    ax.set_xlabel(r'Delay Time $\tau$ (ns)')
    ax.set_ylabel(r'$g^{(2)}(\tau)$')
    ax.grid(True, linestyle='--')

    # --- Plot 2: The Main Fit Plot ---
    ax = axs[0, 1]
    ax.plot(exp_freq_hz / 1e6, exp_psd_au, 'r-', label='Experimental Data (from g2)', alpha=0.4)
    ax.plot(exp_freq_hz / 1e6, initial_guess_spectrum, 'b:', linewidth=2, label='Initial Guess')
    if fit_successful:
        ax.plot(exp_freq_hz / 1e6, final_fit_spectrum, 'k--', linewidth=2.5, label='Final Optimized Fit')
    ax.set_title('2. Experimental Spectrum & Lorentzian Fit')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Power Spectral Density (a.u.)')
    ax.grid(True, linestyle='--')
    ax.legend()
    ax.set_xlim(0, 800)
    ax.set_yscale('log')
    
    # --- Plot 3: Spectrum in Watts ---
    ax = axs[1, 0]
    ax.plot(exp_freq_hz / 1e6, psd_watts_hz, 'r-')
    ax.set_title(r'3. Calculated Power Spectrum (W/Hz)')
    ax.set_xlabel(r'Frequency (MHz)')
    ax.set_ylabel(r'Power Spectral Density (W/Hz)')
    ax.grid(True, linestyle='--')
    ax.set_yscale('log')
    ax.set_xlim(0, 800)

    # --- Plot 4: Spectrum in dBm ---
    ax = axs[1, 1]
    ax.plot(exp_freq_hz / 1e6, psd_dBm_hz, 'g-')
    ax.set_title(r'4. Calculated Power Spectrum (dBm/Hz)')
    ax.set_xlabel(r'Frequency (MHz)')
    ax.set_ylabel(r'Power Spectral Density (dBm/Hz)')
    ax.grid(True, linestyle='--')
    ax.set_xlim(0, 800)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


# --- Main execution block with command-line argument parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze photon statistics and heterodyne data from different experimental sessions.'
    )
    parser.add_argument('--type', type=str, default='analysis', 
                        choices=['analysis'])
    args = parser.parse_args()

    if args.type == 'analysis':
        print("Running code for Analysis")
        analyze_session3()
