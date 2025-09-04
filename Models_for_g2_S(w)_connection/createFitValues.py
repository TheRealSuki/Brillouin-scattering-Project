'''
This script was used to generate the data for the model fit. As it is easier for the mode to fit lorentzian functions
rather than dot data.
'''



import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# A small constant to prevent division by zero
eps = 1e-20

def dbm_to_watts(dbm):
    return 10 ** ((dbm - 30) / 10)    

# --- Lorentzian Fitting Functions ---
def lorentzian(x, x0, gamma, A, y0):
    """ A single Lorentzian peak function. """
    return A * gamma**2 / ((x - x0)**2 + gamma**2) + y0

def triple_lorentzian(x, x01, g1, A1, x02, g2, A2, x03, g3, A3, y0):
    """ A model of three Lorentzian peaks. """
    return (lorentzian(x, x01, g1, A1, 0) +
            lorentzian(x, x02, g2, A2, 0) +
            lorentzian(x, x03, g3, A3, 0) +
            y0)

def load_and_process_heterodyne_data(filepath):
    """
    Loads the heterodyne data from the specified CSV file.
    Returns frequency in Hz.
    """
    print(f"Loading data from {filepath}...")
    # --- FIX: Use 'usecols' to explicitly read only the first two columns ---
    df = pd.read_csv(filepath, header=None, skiprows=41, names=['frequency_hz', 'power_dbm'], usecols=[0, 1])
    
    # Drop any rows with NaN/missing values to prevent calculation errors
    df.dropna(inplace=True)
    
    print("Data loaded and processed successfully.")
    return df['frequency_hz'].values, df['power_dbm'].values

# --- Main Execution Block ---
if __name__ == '__main__':
    
    # --- 1. LOAD EXPERIMENTAL DATA ---
    filepath = '/home/apolloin/Desktop/Brillouin_Scattering/Brillouin-scattering-Project/Measurements/Session_3_Photon_Statistics_And_Heterodyne/3_2GHz_187microWatt_slightly_lasing/data_heterodyne/g21Ghz001.csv'
    
    try:
        exp_freq_hz, exp_power_dbm = load_and_process_heterodyne_data(filepath)
    except FileNotFoundError:
        print(f"ERROR: The file was not found at '{filepath}'.")
        print("Please update the 'filepath' variable with the correct location.")
        exit()

    # --- 2. GENERATE INITIAL GUESSES FROM KNOWN PEAK LOCATIONS ---
    print("Generating initial guesses based on known peak locations...")
    
    # Define the known approximate peak locations in Hz
    known_peaks_hz = np.array([200e6, 280e6, 360e6])

    # Find the indices of the data points closest to the known peak frequencies
    indices = [np.abs(exp_freq_hz - pk).argmin() for pk in known_peaks_hz]
    
    # Get the actual frequencies and amplitudes at those indices
    initial_freqs = exp_freq_hz[indices]
    y0_guess = np.min(exp_power_dbm)
    initial_amps = exp_power_dbm[indices] - y0_guess
    # A reasonable guess for width in Hz (e.g., 10 MHz = 10e6 Hz)
    initial_widths = [10e6] * 3 

    # Assemble the initial guess parameter list [x0, gamma, A, x0, gamma, A, ...]
    p0 = [
        initial_freqs[0], initial_widths[0], initial_amps[0],
        initial_freqs[1], initial_widths[1], initial_amps[1],
        initial_freqs[2], initial_widths[2], initial_amps[2],
        y0_guess
    ]

    print(f"Using known peaks near (Hz): {[f'{f:.2f}' for f in known_peaks_hz]}")
    print(f"Found initial peak guesses at (Hz): {[f'{f:.2f}' for f in initial_freqs]}")


    # --- 3. PERFORM THE TRIPLE LORENTZIAN FIT ---
    try:
        print("Performing curve fit...")
        popt, _ = curve_fit(triple_lorentzian, exp_freq_hz, exp_power_dbm, p0=p0, maxfev=20000)
        print("Fit successful.")

        # --- 4. PLOT THE RESULTS ---
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(14, 8))
        
        # Plot the raw data
        plt.plot(exp_freq_hz, exp_power_dbm, '.', label='Experimental Data', alpha=0.5)
        
        # Create a dense frequency axis for a smooth plot of the fit
        dense_freq_hz = np.linspace(exp_freq_hz.min(), exp_freq_hz.max(), 2000)
        
        # Plot the total fit and the individual Lorentzian components
        fit_total = triple_lorentzian(dense_freq_hz, *popt)
        plt.plot(dense_freq_hz, fit_total, 'k-', label='Total Lorentzian Fit', linewidth=2)
        
        lorentz1 = lorentzian(dense_freq_hz, popt[0], popt[1], popt[2], popt[9])
        lorentz2 = lorentzian(dense_freq_hz, popt[3], popt[4], popt[5], popt[9])
        lorentz3 = lorentzian(dense_freq_hz, popt[6], popt[7], popt[8], popt[9])

        plt.plot(dense_freq_hz, lorentz1, '--', label=f'Peak 1 (f0={popt[0]/1e6:.2f} MHz)')
        plt.plot(dense_freq_hz, lorentz2, '--', label=f'Peak 2 (f0={popt[3]/1e6:.2f} MHz)')
        plt.plot(dense_freq_hz, lorentz3, '--', label=f'Peak 3 (f0={popt[6]/1e6:.2f} MHz)')
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dBm)')
        plt.title('Heterodyne Data with Triple Lorentzian Fit (2GHz Filter)')
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=np.min(exp_power_dbm) - 5, top=np.max(exp_power_dbm) + 5)
        plt.show()

        plt.plot(dense_freq_hz, dbm_to_watts(fit_total), 'k-', label='Total Lorentzian Fit', linewidth=2)

        plt.plot(dense_freq_hz, dbm_to_watts(lorentz1), '--', label=f'Peak 1 (f0={popt[0]/1e6:.2f} MHz)')
        plt.plot(dense_freq_hz, dbm_to_watts(lorentz2), '--', label=f'Peak 2 (f0={popt[3]/1e6:.2f} MHz)')
        plt.plot(dense_freq_hz, dbm_to_watts(lorentz3), '--', label=f'Peak 3 (f0={popt[6]/1e6:.2f} MHz)')

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (Watt)')
        plt.title('Heterodyne Data with Triple Lorentzian Fit (2GHz Filter)')
        plt.legend()
        plt.grid(True)
        plt.show()


        # --- 5. SAVE THE FITTED CURVE DATA ---
        # Create a DataFrame with the fitted curve data
        # Column 1: Fitted Power (dBm), Column 2: Frequency (Hz)
        fit_curve_df = pd.DataFrame({
            'fit_power_dBm': fit_total,
            'frequency_Hz': dense_freq_hz
        })
        fit_parameters = pd.DataFrame({
            'Parameter': ['x01', 'g1', 'A1', 'x02', 'g2', 'A2', 'x03', 'g3', 'A3', 'y0'],
            'Value': popt
        })
        fit_lorentzian_1 = pd.DataFrame({
            'fit_power_dBm': lorentz1,
            'frequency_Hz': dense_freq_hz
        })
        fit_lorentzian_2 = pd.DataFrame({
            'fit_power_dBm': lorentz2,
            'frequency_Hz': dense_freq_hz
        })
        fit_lorentzian_3 = pd.DataFrame({
            'fit_power_dBm': lorentz3,
            'frequency_Hz': dense_freq_hz
        })
        fit_lorentzian_1_watt = pd.DataFrame({
            'fit_power_watt': dbm_to_watts(lorentz1),
            'frequency_Hz': dense_freq_hz
        })
        fit_lorentzian_2_watt = pd.DataFrame({
            'fit_power_watt': dbm_to_watts(lorentz2),
            'frequency_Hz': dense_freq_hz
        })
        fit_lorentzian_3_watt = pd.DataFrame({
            'fit_power_watt': dbm_to_watts(lorentz3),
            'frequency_Hz': dense_freq_hz
        })

        #Currently it save to the same folder, but the lorentzian csv-s were moved to the newmodel and oldmodel folders.
        parameter_filename = 'fit_parameters.csv'
        output_filename = 'heterodyne_fit_curve.csv'
        fit_curve_df.to_csv(output_filename, index=False)
        fit_parameters.to_csv(parameter_filename, index=False)
        fit_lorentzian_1.to_csv('fit_lorentzian_1.csv', index=False)
        fit_lorentzian_2.to_csv('fit_lorentzian_2.csv', index=False)
        fit_lorentzian_3.to_csv('fit_lorentzian_3.csv', index=False)
        fit_lorentzian_1_watt.to_csv('fit_lorentzian_1_watt.csv', index=False)
        fit_lorentzian_2_watt.to_csv('fit_lorentzian_2_watt.csv', index=False)
        fit_lorentzian_3_watt.to_csv('fit_lorentzian_3_watt.csv', index=False)
        
        print(f"\nFitted curve data saved to '{output_filename}'")
        print(f"Fitted parameters saved to '{parameter_filename}'")


    except RuntimeError as e:
        print(f"Error during fitting: {e}")
        print("The fit could not converge. Try adjusting the initial width guess.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

