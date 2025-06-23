import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import simps
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import argparse





# Lorentzian model
def lorentzian(x, x0, gamma, A, y0):
    return A * gamma**2 / ((x - x0)**2 + gamma**2) + y0

def triple_lorentzian(x, x01, g1, A1, x02, g2, A2, x03, g3, A3, y0):
    return (lorentzian(x, x01, g1, A1, 0) +
            lorentzian(x, x02, g2, A2, 0) +
            lorentzian(x, x03, g3, A3, 0) +
            y0)
def six_lorentzian(x, x01, g1, A1, x02, g2, A2, x03, g3, A3, 
                   x04, g4, A4, x05, g5, A5, x06, g6, A6, y0):
    return (lorentzian(x, x01, g1, A1, 0) +
            lorentzian(x, x02, g2, A2, 0) +
            lorentzian(x, x03, g3, A3, 0) +
            lorentzian(x, x04, g4, A4, 0) +
            lorentzian(x, x05, g5, A5, 0) +
            lorentzian(x, x06, g6, A6, 0) +
            y0)
def dbm_to_watts(dbm):
    return 10 ** ((dbm - 30) / 10)       
        
def analyze_session1():
	folder_path = 'OSA_Measurements/Session_1_No_Polariser/'
	image_folder = 'OSA_Measurements_Analysis/Session_1/imagesLorentzians/'
	os.makedirs(folder_path, exist_ok=True)
	os.makedirs(image_folder, exist_ok=True)
	results = []
	for csv_file in glob.glob(os.path.join(folder_path, '*.csv')):
		# Get just the filename
		base_name = os.path.basename(csv_file)

		# Extract substring after "OSA "
		if "OSA " in base_name:
		    just_name = base_name.split("OSA ", 1)[1]
		else:
		    just_name = base_name  # fallback in case "OSA " is missing
		    
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
		powers_fit_dbm = powers_fit

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
			popt, _ = curve_fit(triple_lorentzian, freqs_fit, powers_fit_dbm, p0=p0, maxfev=20000)
			peak_freqs = [popt[0], popt[3], popt[6]]
			peak_amps = [popt[2], popt[5], popt[8]]
			baseline = popt[9]

			# ---- Stationary points and integration section ----
			# Evaluate fit on a dense grid
			dense_x = np.linspace(freqs_fit.min(), freqs_fit.max(), 5000)
			fit_y = triple_lorentzian(dense_x, *popt)

			# Find maxima (peaks) and minima (between peaks)
			maxima_idx = argrelextrema(fit_y, np.greater)[0]
			minima_idx = argrelextrema(fit_y, np.less)[0]
			peak_xs = dense_x[maxima_idx]
			min_xs = dense_x[minima_idx]

			# --------- PLOTTING SECTION ---------
			plt.figure(figsize=(8,5))
			plt.plot(dense_x, fit_y, label='Triple Lorentzian Fit')
			plt.plot(peak_xs, fit_y[maxima_idx], 'ro', label='Peaks')
			plt.plot(min_xs, fit_y[minima_idx], 'go', label='Minima')
			plt.scatter(freqs_fit, powers_fit_dbm, s=10, label='Data', alpha=0.7)
			plt.title(f'Fit for: {just_name[:-4]}')
			plt.xlabel('Frequency (Hz)')
			plt.ylabel('Power (dBm)')
			plt.legend()
			plt.tight_layout()
			save_path = os.path.join(image_folder, f'{just_name[:-4]}_dBm.png')
			plt.savefig(save_path, dpi=180)
			plt.close()
			#plt.show()
			# -------------------------------------
			# ---- Use first two minima as integration boundaries ----
			if len(min_xs) >= 2:
			    min1 = min_xs[0]
			    min2 = min_xs[1]
			else:
			    # Not enough minima detected, fallback to fit region
			    min1 = dense_x[len(dense_x)//3]
			    min2 = dense_x[2*len(dense_x)//3]

			# Convert fit_y to Watts and subtract baseline (in Watts)
			fit_y_watt = dbm_to_watts(fit_y)
			baseline_watt = dbm_to_watts(baseline)
			net_fit_y = fit_y_watt - baseline_watt

			# Integration indices
			idx1 = np.searchsorted(dense_x, min1)
			idx2 = np.searchsorted(dense_x, min2)
			
			# --------- PLOTTING SECTION 2 ---------
			plt.figure(figsize=(10, 6))
			plt.plot(dense_x, fit_y_watt, label='Fitted curve (Watts)', color='blue')
			plt.plot(dense_x, np.full_like(dense_x, baseline_watt), 
				 label='Baseline (Watts)', color='gray', linestyle='--')
			plt.plot(dense_x, net_fit_y, label='Net fit (Watts - baseline)', color='red')

			# Shade integration regions
			plt.axvspan(dense_x[0], dense_x[idx1], color='lightblue', alpha=0.3, label='Peak 1 region')
			plt.axvspan(dense_x[idx1], dense_x[idx2], color='lightgreen', alpha=0.3, label='Peak 2 region')
			plt.axvspan(dense_x[idx2], dense_x[-1], color='navajowhite', alpha=0.3, label='Peak 3 region')

			# Mark minima
			plt.axvline(dense_x[idx1], color='green', linestyle=':', label='Boundary 1 (min1)')
			plt.axvline(dense_x[idx2], color='orange', linestyle=':', label='Boundary 2 (min2)')

			plt.xlabel('Frequency (Hz)')
			plt.ylabel('Power (Watt)')
			plt.title('Fit and Integration Regions (in Watts)')
			plt.legend(loc='upper right')
			plt.tight_layout()
			save_path = os.path.join(image_folder, f'{just_name[:-4]}_Watt.png')
			plt.savefig(save_path, dpi=180)
			plt.close()
			#plt.show()
			# -------------------------------------
			area1 = simps(net_fit_y[:idx1], dense_x[:idx1])
			area2 = simps(net_fit_y[idx1:idx2], dense_x[idx1:idx2])
			area3 = simps(net_fit_y[idx2:], dense_x[idx2:])
		except Exception as e:
			peak_freqs = [np.nan, np.nan, np.nan]
			peak_amps = [np.nan, np.nan, np.nan]
			baseline = np.nan
			area1 = area2 = area3 = np.nan

		# Save all results in a row (add areas)
		row = peak_amps + peak_freqs + [baseline, just_name[:-4], area1, area2, area3]
		results.append(row)

	# Prepare DataFrame and save to CSV
	columns = [
	    'peak1_amp', 'peak2_amp', 'peak3_amp',
	    'peak1_freq', 'peak2_freq', 'peak3_freq',
	    'baseline', 'powerOf10PercentBeamSplitter(MicroWatt)',
	    'peak1_area_W_Hz', 'peak2_area_W_Hz', 'peak3_area_W_Hz'
	]
	df = pd.DataFrame(results, columns=columns)

	output_folder = 'OSA_Measurements_Analysis/Session_1/'
	os.makedirs(output_folder, exist_ok=True)
	output_path = os.path.join(output_folder, 'all_peaks_summary.csv')
	df.to_csv(output_path, index=False)

	print("Analysis and integration complete! Results saved to:", output_path)



def analyze_session2():
	folder_path = 'OSA_Measurements/Session_2_Heterodyne_Measurement_MHzRange/'
	image_folder = 'OSA_Measurements_Analysis/Session_2/imagesLorentzians/'
	os.makedirs(folder_path, exist_ok=True)
	os.makedirs(image_folder, exist_ok=True)
	results = []
	for csv_file in glob.glob(os.path.join(folder_path, '*.csv')):
		just_name = os.path.basename(csv_file)

		with open(csv_file, 'r') as f:
			lines = f.readlines()
			# Skip the first 44 lines (indexes 0-43), line 44 is header, data starts at line 45 (index 45)
			data_lines = lines[45:]  # This gets all data rows (as strings)
			freqs = []
			powers = []
			for line in data_lines:
				parts = line.strip().split(',')
				if len(parts) >= 2:  # Ensure both frequency and power exist
					try:
						freq = float(parts[0])
						power = float(parts[1])
						freqs.append(freq)
						powers.append(power)
					except ValueError:
						continue  # Skip any rows with bad formatting
			freqs = np.array(freqs)
			powers = np.array(powers)

		mask = (freqs > 50000000) & (powers > -121) & (powers < -70)
		freqs_fit = freqs[mask]
		powers_fit = powers[mask]
		powers_fit_dbm = powers_fit

		# Your 6 peak initial guesses
		known_peaks = np.array([
		    69712351.95,
		    149676988.2,
		    228087986.5,
		    303891709,
		    375634517.8,
		    752622673.4
		])
		indices = [np.abs(freqs_fit - pk).argmin() for pk in known_peaks]
		initial_freqs = freqs_fit[indices]
		initial_amps = powers_fit[indices] - (-121)
		initial_widths = [np.ptp(freqs_fit)/30]*6

		order = np.argsort(initial_freqs)
		initial_freqs = initial_freqs[order]
		initial_amps = initial_amps[order]
		initial_widths = np.array(initial_widths)[order]
		y0_guess = -121

		# 6 peaks: freq, width, amp for each, plus baseline
		
		p0 = [
		    initial_freqs[0], initial_widths[0], initial_amps[0],
		    initial_freqs[1], initial_widths[1], initial_amps[1],
		    initial_freqs[2], initial_widths[2], initial_amps[2],
		    initial_freqs[3], initial_widths[3], initial_amps[3],
		    initial_freqs[4], initial_widths[4], initial_amps[4],
		    initial_freqs[5], initial_widths[5], initial_amps[5],
		    y0_guess
		]
		print(just_name)
		print("---------------")
		print(p0)
		print("---------------")
		center_tol = 1e6  # 1 MHz
		lower_bounds = []
		upper_bounds = []
		for i in range(6):
			if i < 5:
				# Center
				lower_bounds.append(initial_freqs[i] - center_tol)
				# Width
				lower_bounds.append(1e3)
				# Amplitude
				lower_bounds.append(1e-6)  # first 5 amplitudes must be positive
			else:
				lower_bounds.extend([initial_freqs[i] - 2e6, 1e3, 0])
			#Center, Width, Amplitude - Upper Bounds
			upper_bounds.append(initial_freqs[i] + center_tol)
			upper_bounds.append(np.ptp(freqs_fit))
			upper_bounds.append(np.max(powers_fit_dbm)*2)
		lower_bounds.append(np.min(powers_fit_dbm) - 2)
		upper_bounds.append(np.max(powers_fit_dbm) + 2)

		try:

			popt, _ = curve_fit(six_lorentzian, freqs_fit, powers_fit_dbm, p0=p0, maxfev=100000)
			peak_freqs = [popt[0], popt[3], popt[6], popt[9], popt[12], popt[15]]
			peak_amps = [popt[2], popt[5], popt[8], popt[11], popt[14], popt[17]]
			baseline = popt[18]


			# Dense grid for fit and analysis
			dense_x = np.linspace(freqs_fit.min(), freqs_fit.max(), 10000)

			fit_y = six_lorentzian(dense_x, *popt)


			
			maxima_idx = argrelextrema(fit_y, np.greater)[0]
			minima_idx = argrelextrema(fit_y, np.less)[0]
			peak_xs = dense_x[maxima_idx]
			min_xs = dense_x[minima_idx]


			plt.figure(figsize=(8,5))
			plt.plot(dense_x, fit_y, label='6-Lorentzian Fit')
			plt.plot(peak_xs, fit_y[maxima_idx], 'ro', label='Peaks')
			plt.plot(min_xs, fit_y[minima_idx], 'go', label='Minima')
			plt.scatter(freqs_fit, powers_fit_dbm, s=10, label='Data', alpha=0.7)
			plt.title(f'Fit for: {just_name}')
			plt.xlabel('Frequency (Hz)')
			plt.ylabel('Power (dBm)')
			plt.legend()
			plt.tight_layout()
			save_path = os.path.join(image_folder, f'{just_name}_dBm.png')
			plt.savefig(save_path, dpi=180)
			plt.close()

			# Use minima to define integration regions for 6 peaks = 5 minima boundaries
			# If less than 5 minima, fill with equally spaced boundaries
			
			if len(min_xs) >= 5:
				boundaries = [dense_x[0]] + list(min_xs[:5]) + [dense_x[-1]]
			else:
				# fallback: equally spaced boundaries
				boundaries = list(np.linspace(dense_x[0], dense_x[-1], 7))
			fit_y_watt = dbm_to_watts(fit_y)
			baseline_watt = dbm_to_watts(baseline)
			net_fit_y = fit_y_watt - baseline_watt

			idxs = [np.searchsorted(dense_x, b) for b in boundaries]

			# Plot with integration regions for all 6
			plt.figure(figsize=(10, 6))
			plt.plot(dense_x, fit_y_watt, label='Fitted curve (Watts)', color='blue')
			plt.plot(dense_x, np.full_like(dense_x, baseline_watt), 
			     label='Baseline (Watts)', color='gray', linestyle='--')
			plt.plot(dense_x, net_fit_y, label='Net fit (Watts - baseline)', color='red')

			colors = ['lightblue', 'lightgreen', 'navajowhite', 'pink', 'plum', 'lightcoral']
			for i in range(6):
				plt.axvspan(dense_x[idxs[i]], dense_x[idxs[i+1]], color=colors[i], alpha=0.3, label=f'Peak {i+1} region')

			for i in range(1, 6):
				plt.axvline(dense_x[idxs[i]], color='gray', linestyle=':', label=f'Boundary {i}')

			plt.xlabel('Frequency (Hz)')
			plt.ylabel('Power (Watt)')
			plt.title('Fit and Integration Regions (in Watts)')
			plt.legend(loc='upper right', fontsize=8)
			plt.tight_layout()
			
			#MANUALLY ADDING A MAX PLOT, SO IT IS VISIBLE AT ALL TIMES
			max_dbm = -85
			max_watt = dbm_to_watts(max_dbm)
			plt.ylim(bottom=0, top=max_watt * 1.1)
			#MANUAL PART END

			save_path = os.path.join(image_folder, f'{just_name}_Watt.png')
			plt.savefig(save_path, dpi=180)
			plt.close()

			# Integrate area under each peak
			areas = []
			for i in range(6):
				area = simps(net_fit_y[idxs[i]:idxs[i+1]], dense_x[idxs[i]:idxs[i+1]])
				areas.append(area)
			

		except Exception as e:
			peak_freqs = [np.nan]*6
			peak_amps = [np.nan]*6
			baseline = np.nan
			areas = [np.nan]*6
			print("Issue")

		row = peak_amps + peak_freqs + [baseline, just_name[:-4]] + areas
		results.append(row)

	columns = (
	[f'peak{i+1}_amp' for i in range(6)] +
	[f'peak{i+1}_freq' for i in range(6)] +
	['baseline', 'powerOf10PercentBeamSplitter(MicroWatt)'] +
	[f'peak{i+1}_area_W_Hz' for i in range(6)]
	)
	df = pd.DataFrame(results, columns=columns)

	output_folder = 'OSA_Measurements_Analysis/Session_2/'
	os.makedirs(output_folder, exist_ok=True)
	output_path = os.path.join(output_folder, 'all_peaks_summary.csv')
	df.to_csv(output_path, index=False)

	print("Analysis and integration complete! Results saved to:", output_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=str, default='session_1', choices=['session_1', 'session_2'],
                        help='Choose which session to analyze (default: session_1)')
    args = parser.parse_args()

    if args.session == 'session_1':
        # Session 1 code
        print("Running code for SESSION 1")
        analyze_session1()
    elif args.session == 'session_2':
        # Session 2 code
        print("Running code for SESSION 2")
        analyze_session2()
























