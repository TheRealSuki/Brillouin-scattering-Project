import os
import glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import argparse
from scipy.integrate import simpson







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
def five_lorentzian(x, x01, g1, A1, x02, g2, A2, x03, g3, A3, 
                   x04, g4, A4, x05, g5, A5, y0):
    return (lorentzian(x, x01, g1, A1, 0) +
            lorentzian(x, x02, g2, A2, 0) +
            lorentzian(x, x03, g3, A3, 0) +
            lorentzian(x, x04, g4, A4, 0) +
            lorentzian(x, x05, g5, A5, 0) +
            y0)
def dbm_to_watts(dbm):
    return 10 ** ((dbm - 30) / 10)       
        
def analyze_session1():
	folder_path = 'Measurements/Session_1_No_Polariser/'
	image_folder = 'Measurements_Analysis/Session_1/imagesLorentzians/'
	os.makedirs(folder_path, exist_ok=True)
	os.makedirs(image_folder, exist_ok=True)
	results = []
	for csv_file in glob.glob(os.path.join(folder_path, '*.csv')):
		# Get just the filename
		base_name = os.path.basename(csv_file)
		print(base_name)
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
			peak_widths = [popt[1], popt[4], popt[7]]
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
			# Define the conversion factor for clarity
			hz_to_thz = 1e12

			# Convert all frequency data to THz
			dense_x_thz = dense_x / hz_to_thz
			peak_xs_thz = peak_xs / hz_to_thz
			min_xs_thz = min_xs / hz_to_thz
			freqs_fit_thz = freqs_fit / hz_to_thz

			# Now plot using the converted THz values
			plt.figure(figsize=(8,5))
			plt.plot(dense_x_thz, fit_y, label='Triple Lorentzian Fit')
			plt.plot(peak_xs_thz, fit_y[maxima_idx], 'ro', label='Peaks')
			plt.plot(min_xs_thz, fit_y[minima_idx], 'go', label='Minima')
			plt.scatter(freqs_fit_thz, powers_fit_dbm, s=10, label='Data', alpha=0.7)
			plt.title(f'Fit for: {str(round(float(just_name[:-4])*9/1000, 2))} mW Input Power')
			plt.grid(True, which='both', linestyle='--', linewidth=0.5)

			# Update the xlabel to reflect the new unit
			plt.xlabel('Frequency (THz)')
			plt.ylabel('Power (dBm)')

			plt.legend()
			plt.tight_layout()
			save_path = os.path.join(image_folder, f'{str(round(float(just_name[:-4])*9/1000, 2))}_dBm.png')
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
			# Define the frequency conversion factor for THz
			hz_to_thz = 1e12

			# Convert the frequency data from Hz to THz
			dense_x_thz = dense_x / hz_to_thz

			# --- Dynamic Y-axis scaling logic ---
			# Check the maximum power to decide on the appropriate unit
			max_power_watt = np.max(fit_y_watt)

			if max_power_watt >= 1e-3:
				# Use milliwatts (mW)
				y_conversion_factor = 1e3
				y_unit = 'mW'
			elif max_power_watt >= 1e-6:
				# Use microwatts (µW)
				y_conversion_factor = 1e6
				y_unit = 'µW'
			elif max_power_watt >= 1e-9:
				# Use nanowatts (nW)
				y_conversion_factor = 1e9
				y_unit = 'nW'
			elif max_power_watt >= 1e-12:
				# Use picowatts (pW)
				y_conversion_factor = 1e12
				y_unit = 'pW'
			else:
				# Use femtowatts (fW)
				y_conversion_factor = 1e15
				y_unit = 'fW'


			# Apply the scaling to all y-axis data
			fit_y_scaled = fit_y_watt * y_conversion_factor
			baseline_scaled = baseline_watt * y_conversion_factor
			net_fit_y_scaled = net_fit_y * y_conversion_factor
			# --- End of dynamic scaling logic ---


			plt.figure(figsize=(10, 6))

			# Use the new 'dense_x_thz' for x-axis and the dynamically scaled values for y-axis
			plt.plot(dense_x_thz, fit_y_scaled, label=f'Fitted curve ({y_unit})', color='blue')
			plt.plot(dense_x_thz, np.full_like(dense_x_thz, baseline_scaled),
					label=f'Baseline ({y_unit})', color='gray', linestyle='--')
			plt.plot(dense_x_thz, net_fit_y_scaled, label=f'Net fit ({y_unit} - baseline)', color='red')

			# Shade integration regions using the THz values for the boundaries
			plt.axvspan(dense_x_thz[0], dense_x_thz[idx1], color='lightblue', alpha=0.3, label='Peak 1 region')
			plt.axvspan(dense_x_thz[idx1], dense_x_thz[idx2], color='lightgreen', alpha=0.3, label='Peak 2 region')
			plt.axvspan(dense_x_thz[idx2], dense_x_thz[-1], color='navajowhite', alpha=0.3, label='Peak 3 region')

			# Mark minima using the THz values
			plt.axvline(dense_x_thz[idx1], color='green', linestyle=':', label='Boundary 1 (min1)')
			plt.axvline(dense_x_thz[idx2], color='orange', linestyle=':', label='Boundary 2 (min2)')

			# Update axis labels and title to reflect the dynamic units
			plt.xlabel('Frequency (THz)')
			plt.ylabel(f'Power ({y_unit})')
			plt.title(f'Fit and Integration Regions (in {y_unit})')
			plt.grid(True, which='both', linestyle='--', linewidth=0.5)

			# Adjust legend to prevent overlapping with the plot
			plt.legend(loc='best') # 'best' will find the least obstructive location
			plt.tight_layout()

			# The filename remains the same as it's based on input power, not frequency units
			save_path = os.path.join(image_folder, f'{str(round(float(just_name[:-4])*9/1000, 2))}_Watt.png')
			plt.savefig(save_path, dpi=180)
			plt.close()
			#plt.show()
			# -------------------------------------
			area1 = simpson(net_fit_y[:idx1], dense_x[:idx1])
			area2 = simpson(net_fit_y[idx1:idx2], dense_x[idx1:idx2])
			area3 = simpson(net_fit_y[idx2:], dense_x[idx2:])
		except Exception as e:
			peak_freqs = [np.nan, np.nan, np.nan]
			peak_amps = [np.nan, np.nan, np.nan]
			baseline = np.nan
			area1 = area2 = area3 = np.nan
			print(e)

		# Save all results in a row (add areas)
		row = peak_amps + peak_freqs + [baseline, round(float(just_name[:-4])*9, 2), area1, area2, area3] + peak_widths
		results.append(row)

	# Prepare DataFrame and save to CSV
	columns = [
	    'peak1_amp', 'peak2_amp', 'peak3_amp',
	    'peak1_freq', 'peak2_freq', 'peak3_freq',
	    'baseline', 'input_power_uW',
	    'peak1_area_W_Hz', 'peak2_area_W_Hz', 'peak3_area_W_Hz',
		'peak1_width', 'peak2_width', 'peak3_width'
	]
	df = pd.DataFrame(results, columns=columns)

	output_folder = 'Measurements_Analysis/Session_1/'
	os.makedirs(output_folder, exist_ok=True)
	output_path = os.path.join(output_folder, 'all_peaks_summary.csv')
	df.to_csv(output_path, index=False)

	print("Analysis and integration complete! Results saved to:", output_path)



def analyze_session2():
	folder_path = 'Measurements/Session_2_Heterodyne_Measurement_MHzRange/'
	image_folder = 'Measurements_Analysis/Session_2/imagesLorentzians/'
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

		if float(just_name[:-14]) > 800:
			#6 peak initial guesses
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
			#Bit where the extra peak starts appearing
			center_tol = 10e6     # 1 MHz
			amp_max   = (np.max(powers_fit_dbm) - y0_guess) * 2
			width_max = np.ptp(freqs_fit)

			lower_bounds = []
			upper_bounds = []
			for f0, w0, a0 in zip(initial_freqs, initial_widths, initial_amps):
				if f0 == 375634517.766497:
					lower_bounds += [f0 - center_tol*1.1,  # freq  >= f0-1 MHz
							5e3,              # width >= 1 kHz
							a0*0.85]           # amplitude >= 
					upper_bounds += [f0 + center_tol*1.1,  # freq  <= f0+1 MHz
							width_max*3,        # width <= full span
							amp_max*3]          # amplitude <= 2×max
				else:
					lower_bounds += [f0 - center_tol,  # freq  >= f0-1 MHz
							1e3,              # width >= 1 kHz
							a0*0.3]           # amplitude >=  10% of guess (or 0)
					upper_bounds += [f0 + center_tol,  # freq  <= f0+1 MHz
							width_max,        # width <= full span
							amp_max*1.5]          # amplitude <= 2×max
			# finally, bounds for the baseline y0:
			lower_bounds.append(y0_guess - 5)      # e.g. 5 dB below guess
			upper_bounds.append(y0_guess + 5)      # e.g. 5 dB above guess
			try:

				popt, _ = curve_fit(
				    six_lorentzian, freqs_fit, powers_fit_dbm,
				    p0=p0, bounds=(lower_bounds, upper_bounds),
				    maxfev=100000
				)
				fit_vals  = six_lorentzian(freqs_fit, *popt)
				residuals = powers_fit_dbm - fit_vals
				ss_res = np.sum(residuals**2)
				ss_tot = np.sum((powers_fit_dbm - np.mean(powers_fit_dbm))**2)
				r_squared = 1 - (ss_res/ss_tot)
				#Calculation of the r^2 to see how good of a fit it
				peak_freqs = [popt[0], popt[3], popt[6], popt[9], popt[12], popt[15]]
				peak_widths = [popt[1], popt[4], popt[7], popt[10], popt[13], popt[16]]
				peak_amps = [popt[2], popt[5], popt[8], popt[11], popt[14], popt[17]]
				baseline = popt[18]


				# Dense grid for fit and analysis
				dense_x = np.linspace(freqs_fit.min(), freqs_fit.max(), 10000)

				fit_y = six_lorentzian(dense_x, *popt)


				
				maxima_idx = argrelextrema(fit_y, np.greater)[0]
				minima_idx = argrelextrema(fit_y, np.less)[0]
				peak_xs = dense_x[maxima_idx]
				min_xs = dense_x[minima_idx]
				# Define the frequency conversion factor for MHz
				hz_to_mhz = 1e6

				# Convert the frequency data from Hz to MHz
				dense_x_mhz = dense_x / hz_to_mhz


				plt.figure(figsize=(8,5))
				plt.plot(dense_x_mhz, fit_y, label='6-Lorentzian Fit')
				plt.plot(peak_xs / hz_to_mhz, fit_y[maxima_idx], 'ro', label='Peaks')
				plt.plot(min_xs / hz_to_mhz, fit_y[minima_idx], 'go', label='Minima')
				plt.scatter(freqs_fit / hz_to_mhz, powers_fit_dbm, s=10, label='Data', alpha=0.7)
				plt.title(f'Fit for: {str(round(float(just_name[:-14])*9/1000, 2))} mW Input Power')
				plt.grid(True, which='both', linestyle='--', linewidth=0.5)
				plt.xlabel('Frequency (MHz)')
				plt.ylabel('Power (dBm)')
				plt.legend()
				plt.tight_layout()
				save_path = os.path.join(image_folder, f'{str(round(float(just_name[:-14])*9/1000, 2))}_dBm.png')
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
				plt.grid(True, which='both', linestyle='--', linewidth=0.5)
				plt.legend(loc='upper right', fontsize=8)
				plt.tight_layout()
				max_display = max(np.max(fit_y_watt), np.max(net_fit_y)) * 1.1
				plt.ylim(bottom=0, top=max_display)

				save_path = os.path.join(image_folder, f'{str(round(float(just_name[:-14])*9/1000, 2))}_Watt.png')
				plt.savefig(save_path, dpi=180)
				plt.close()

				# Integrate area under each peak
				areas = []
				for i in range(6):
					area = simpson(net_fit_y[idxs[i]:idxs[i+1]], dense_x[idxs[i]:idxs[i+1]])
					areas.append(area)
				

			except Exception as e:
				peak_freqs = [np.nan]*6
				peak_amps = [np.nan]*6
				baseline = np.nan
				areas = [np.nan]*6
				print("Issue")

			row = peak_amps + peak_freqs + peak_widths + [baseline, round(float(just_name[:-14])*9, 2), r_squared] + areas
			results.append(row)

		else:
			# 5 peak initial guesses
			known_peaks = np.array([
			    69712351.95,
			    149676988.2,
			    228087986.5,
			    303891709,
			    375634517.8
			])
			indices = [np.abs(freqs_fit - pk).argmin() for pk in known_peaks]
			initial_freqs = freqs_fit[indices]
			initial_amps = powers_fit[indices] - (-121)
			initial_widths = [np.ptp(freqs_fit)/30]*5

			order = np.argsort(initial_freqs)
			initial_freqs = initial_freqs[order]
			initial_amps = initial_amps[order]
			initial_widths = np.array(initial_widths)[order]
			y0_guess = -121

			# 5 peaks: freq, width, amp for each, plus baseline
			
			p0 = [
			    initial_freqs[0], initial_widths[0], initial_amps[0],
			    initial_freqs[1], initial_widths[1], initial_amps[1],
			    initial_freqs[2], initial_widths[2], initial_amps[2],
			    initial_freqs[3], initial_widths[3], initial_amps[3],
			    initial_freqs[4], initial_widths[4], initial_amps[4],
			    y0_guess
			]
			#Bit where the extra peak does not appear
			center_tol = 10e6     # 1 MHz
			amp_max   = (np.max(powers_fit_dbm) - y0_guess) * 2
			width_max = np.ptp(freqs_fit)

			lower_bounds = []
			upper_bounds = []
			for f0, w0, a0 in zip(initial_freqs, initial_widths, initial_amps):
				if f0 == 375634517.766497:
					lower_bounds += [f0 - center_tol*1.2,  # freq  >= f0-1 MHz
							1e3,              # width >= 1 kHz
							a0]           # amplitude >= 
					upper_bounds += [f0 + center_tol*1.2,  # freq  <= f0+1 MHz
							width_max*4,        # width <= full span
							amp_max]          # amplitude <= 2×max
				else:
					lower_bounds += [f0 - center_tol*1.2,  # freq  >= f0-1 MHz
							1e3,              # width >= 1 kHz
							a0*0.4]           # amplitude >=  10% of guess (or 0)
					upper_bounds += [f0 + center_tol*1.2,  # freq  <= f0+1 MHz
							width_max,        # width <= full span
							amp_max*1.5]          # amplitude <= 2×max
			# finally, bounds for the baseline y0:
			lower_bounds.append(y0_guess - 5)      # e.g. 5 dB below guess
			upper_bounds.append(y0_guess + 5)      # e.g. 5 dB above guess
			try:

				popt, _ = curve_fit(
				    five_lorentzian, freqs_fit, powers_fit_dbm,
				    p0=p0, bounds=(lower_bounds, upper_bounds),
				    maxfev=100000
				)
				#Calculation of r^2 - see how good of a fit it is.
				fit_vals  = five_lorentzian(freqs_fit, *popt)
				residuals = powers_fit_dbm - fit_vals
				ss_res = np.sum(residuals**2)
				ss_tot = np.sum((powers_fit_dbm - np.mean(powers_fit_dbm))**2)
				r_squared = 1 - (ss_res/ss_tot)
				
				peak_freqs = [popt[0], popt[3], popt[6], popt[9], popt[12]]
				peak_widths = [popt[1], popt[4], popt[7], popt[10], popt[13]]
				peak_amps = [popt[2], popt[5], popt[8], popt[11], popt[14]]
				baseline = popt[15]


				# Dense grid for fit and analysis
				dense_x = np.linspace(freqs_fit.min(), freqs_fit.max(), 10000)

				fit_y = five_lorentzian(dense_x, *popt)


				
				maxima_idx = argrelextrema(fit_y, np.greater)[0]
				minima_idx = argrelextrema(fit_y, np.less)[0]
				peak_xs = dense_x[maxima_idx]
				min_xs = dense_x[minima_idx]
				# Define the frequency conversion factor for MHz
				hz_to_mhz = 1e6

				# Convert the frequency data from Hz to MHz
				dense_x_mhz = dense_x / hz_to_mhz

				plt.figure(figsize=(8,5))
				plt.plot(dense_x_mhz, fit_y, label='5-Lorentzian Fit')
				plt.plot(peak_xs / hz_to_mhz, fit_y[maxima_idx], 'ro', label='Peaks')
				plt.plot(min_xs / hz_to_mhz, fit_y[minima_idx], 'go', label='Minima')
				plt.scatter(freqs_fit / hz_to_mhz, powers_fit_dbm, s=10, label='Data', alpha=0.7)
				plt.title(f'Fit for: {str(round(float(just_name[:-14])*9/1000, 2))} mW Input Power')
				plt.grid(True, which='both', linestyle='--', linewidth=0.5)
				plt.xlabel('Frequency (MHz)')
				plt.ylabel('Power (dBm)')
				plt.legend()
				plt.tight_layout()
				save_path = os.path.join(image_folder, f'{str(round(float(just_name[:-14])*9/1000, 2))}_dBm.png')
				plt.savefig(save_path, dpi=180)
				plt.close()

				# Use minima to define integration regions for 5 peaks = 4 minima boundaries
				# If less than 4 minima, fill with equally spaced boundaries
				
				if len(min_xs) >= 4:
					boundaries = [dense_x[0]] + list(min_xs[:5]) + [dense_x[-1]]
				else:
					# fallback: equally spaced boundaries
					boundaries = list(np.linspace(dense_x[0], dense_x[-1], 7))
				fit_y_watt = dbm_to_watts(fit_y)
				baseline_watt = dbm_to_watts(baseline)
				net_fit_y = fit_y_watt - baseline_watt

				idxs = [np.searchsorted(dense_x, b) for b in boundaries]
				

				# Plot with integration regions for all 5
				plt.figure(figsize=(10, 6))
				plt.plot(dense_x, fit_y_watt, label='Fitted curve (Watts)', color='blue')
				plt.plot(dense_x, np.full_like(dense_x, baseline_watt), 
				     label='Baseline (Watts)', color='gray', linestyle='--')
				plt.plot(dense_x, net_fit_y, label='Net fit (Watts - baseline)', color='red')

				colors = ['lightblue', 'lightgreen', 'navajowhite', 'pink', 'plum', 'lightcoral']
				for i in range(5):
					plt.axvspan(dense_x[idxs[i]], dense_x[idxs[i+1]], color=colors[i], alpha=0.3, label=f'Peak {i+1} region')

				for i in range(1, 5):
					plt.axvline(dense_x[idxs[i]], color='gray', linestyle=':', label=f'Boundary {i}')

				plt.xlabel('Frequency (Hz)')
				plt.ylabel('Power (Watt)')
				plt.title('Fit and Integration Regions (in Watts)')
				plt.grid(True, which='both', linestyle='--', linewidth=0.5)
				plt.legend(loc='upper right', fontsize=8)
				plt.tight_layout()
				max_display = max(np.max(fit_y_watt), np.max(net_fit_y)) * 1.1
				plt.ylim(bottom=0, top=max_display)

				save_path = os.path.join(image_folder, f'{str(round(float(just_name[:-14])*9/1000, 2))}_Watt.png')
				plt.savefig(save_path, dpi=180)
				plt.close()

				# Integrate area under each peak
				areas = []
				for i in range(5):
					area = simpson(net_fit_y[idxs[i]:idxs[i+1]], dense_x[idxs[i]:idxs[i+1]])
					areas.append(area)
				

			except Exception as e:
				peak_freqs = [np.nan]*5
				peak_amps = [np.nan]*5
				baseline = np.nan
				areas = [np.nan]*5
				print("Issue")
			peak_amps.append(0)
			peak_freqs.append(0)
			areas.append(0)
			peak_widths.append(0)
			row = peak_amps + peak_freqs + peak_widths + [baseline, round(float(just_name[:-14])*9, 2), r_squared] + areas
			results.append(row)



	columns = (
	[f'peak{i+1}_amp' for i in range(6)] +
	[f'peak{i+1}_freq' for i in range(6)] +
	[f'peak{i+1}_width' for i in range(6)] +
	['baseline', 'input_power_uW', 'r_squared for dBm Fit'] +
	[f'peak{i+1}_area_W_Hz' for i in range(6)]
	)
	df = pd.DataFrame(results, columns=columns)

	output_folder = 'Measurements_Analysis/Session_2/'
	os.makedirs(output_folder, exist_ok=True)
	output_path = os.path.join(output_folder, 'all_peaks_summary.csv')
	df.to_csv(output_path, index=False)

	print("Analysis and integration complete! Results saved to:", output_path)




# --- Main execution block with command-line argument parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze photon statistics and heterodyne data from different experimental sessions.'
    )
    parser.add_argument('--session', type=str, default='session_1', 
                        choices=['session_1', 'session_2'],
                        help='Choose which session to analyze (default: session_1)')
    args = parser.parse_args()

    if args.session == 'session_1':
        print("Running code for SESSION 1")
        analyze_session1()
    elif args.session == 'session_2':
        print("Running code for SESSION 2")
        analyze_session2()
















