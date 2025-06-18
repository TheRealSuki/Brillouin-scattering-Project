import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit



# Lorentzian model
def lorentzian(x, x0, gamma, A, y0):
    return A * gamma**2 / ((x - x0)**2 + gamma**2) + y0

def triple_lorentzian(x, x01, g1, A1, x02, g2, A2, x03, g3, A3, y0):
    return (lorentzian(x, x01, g1, A1, 0) +
            lorentzian(x, x02, g2, A2, 0) +
            lorentzian(x, x03, g3, A3, 0) +
            y0)


folder_path = 'OSA_Measurements/Session_1_No_Polariser/'
results = []

for csv_file in glob.glob(os.path.join(folder_path, '*.csv')):
	with open(csv_file, 'r') as f:
		lines = f.readlines()
		# Get 2nd and 3rd lines (index 1 and 2)
		freqs_line = lines[1].strip()
		powers_line = lines[2].strip()
		# Split by comma, convert to float
		freqs = np.array([float(x) for x in freqs_line.split(',')])
		powers = np.array([float(x) for x in powers_line.split(',')])

	# Remove -120 dBm floor
	mask = powers > -119
	freqs_fit = freqs[mask]
	powers_fit = powers[mask]

	known_peaks = [
		193394525726376.62,
		193405005948095.44,
		193416236017615.62
	]
	known_peaks = np.array(known_peaks)

	indices = [np.abs(freqs_fit - pk).argmin() for pk in known_peaks]
	initial_freqs = freqs_fit[indices]
	initial_amps = powers_fit[indices] - np.median(powers_fit)
	initial_widths = [np.ptp(freqs_fit)/30]*3

	order = np.argsort(initial_freqs)
	initial_freqs = initial_freqs[order]
	initial_amps = initial_amps[order]
	initial_widths = np.array(initial_widths)[order]

	y0_guess = np.median(powers_fit)

	p0 = [
		initial_freqs[0], initial_widths[0], initial_amps[0],
		initial_freqs[1], initial_widths[1], initial_amps[1],
		initial_freqs[2], initial_widths[2], initial_amps[2],
		y0_guess
	]

	try:
		popt, _ = curve_fit(triple_lorentzian, freqs_fit, powers_fit, p0=p0, maxfev=20000)
		# Extract peaks and amplitudes
		peak_freqs = [popt[0], popt[3], popt[6]]
		peak_amps = [popt[2], popt[5], popt[8]]
		baseline = popt[9]
	except Exception as e:
		# If fit fails, fill with NaNs
		peak_freqs = [np.nan, np.nan, np.nan]
		peak_amps = [np.nan, np.nan, np.nan]
		baseline = np.nan


	# Get just the filename
	base_name = os.path.basename(csv_file)

	# Extract substring after "OSA "
	if "OSA " in base_name:
	    just_name = base_name.split("OSA ", 1)[1]
	else:
	    just_name = base_name  # fallback in case "OSA " is missing
	    
	# Save all results in a row
	row = peak_amps + peak_freqs + [baseline, just_name[:-4]]
	results.append(row)

	# Prepare DataFrame and save to CSV
	columns = [
		'peak1_amp', 'peak2_amp', 'peak3_amp',
		'peak1_freq', 'peak2_freq', 'peak3_freq',
		'baseline', 'powerOf10PercentBeamSplitter(MicroWatt)'
	]
	df = pd.DataFrame(results, columns=columns)
	
	output_folder = 'OSA_Measurements_Analysis/Session_1/'
	output_path = os.path.join(output_folder, 'all_peaks_summary.csv')
	df.to_csv(output_path, index=False)




