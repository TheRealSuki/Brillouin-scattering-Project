import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

def watts_to_dbm(watts):
    return 10 * np.log10(watts) + 30
def dbm_to_watts(dbm):
    return 10 ** ((dbm - 30) / 10)  


def analyseSession_1():
	# Load the CSV
	df = pd.read_csv('OSA_Measurements_Analysis/Session_1/all_peaks_summary.csv')

	power = df['powerOf10PercentBeamSplitter(MicroWatt)']

	# Make 'images' folder next to the CSV if it doesn't exist
	images_folder = "OSA_Measurements_Analysis/Session_1/images"

	# Plot 1: x = amplitude, y = power
	plt.figure(figsize=(10,6))
	plt.scatter(df['peak1_amp'], power, label='Stokes Amplitude')
	plt.scatter(df['peak2_amp'], power, label='Rayleigh Amplitude')
	plt.scatter(df['peak3_amp'], power, label='Anti-Stokes Amplitude')
	plt.ylabel('Power of 10% Beam Splitter (MicroWatt)')
	plt.xlabel('Peak Amplitude (dBm)')
	plt.title('Beam Splitter Power vs Peak Amplitudes')
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'amplitude_vs_power.png'), dpi=300)
	plt.show()

	# Plot 2: x = power, y = amplitude (axes swapped)
	plt.figure(figsize=(10,6))
	plt.scatter(power, df['peak1_amp'], label='Stokes Amplitude')
	plt.scatter(power, df['peak2_amp'], label='Rayleigh Amplitude')
	plt.scatter(power, df['peak3_amp'], label='Anti-Stokes Amplitude')
	plt.xlabel('Power of 10% Beam Splitter (MicroWatt)')
	plt.ylabel('Peak Amplitude (dBm)')
	plt.title('Peak Amplitudes vs Beam Splitter Power')
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'power_vs_amplitude.png'), dpi=300)
	plt.show()

	# Plot 3: x = area, y = power
	plt.figure(figsize=(10,6))
	plt.scatter(df['peak1_area_W_Hz'], power, label='Stokes Area', s=5)
	plt.scatter(df['peak2_area_W_Hz'], power, label='Rayleigh Area', s=5)
	plt.scatter(df['peak3_area_W_Hz'], power, label='Anti-Stokes Area', s=5)
	plt.ylabel('Power of 10% Beam Splitter (MicroWatt)')
	plt.xlabel('Peak Area (Watt × Hz)')
	plt.title('Beam Splitter Power vs Peak Areas')
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'area_vs_power.png'), dpi=300)
	# plt.show()  # Commented for fast batch processing

	# Plot 4: x = power, y = area (axes swapped)
	plt.figure(figsize=(10,6))
	plt.scatter(power, df['peak1_area_W_Hz'], label='Stokes Area', s=5)
	plt.scatter(power, df['peak2_area_W_Hz'], label='Rayleigh Area', s=5)
	plt.scatter(power, df['peak3_area_W_Hz'], label='Anti-Stokes Area', s=5)
	plt.xlabel('Power of 10% Beam Splitter (MicroWatt)')
	plt.ylabel('Peak Area (Watt × Hz)')
	plt.title('Peak Areas vs Beam Splitter Power')
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'power_vs_area.png'), dpi=300)
	# plt.show()  # Commented for fast batch processing

	# Plot 5: x = area, y = power
	plt.figure(figsize=(10,6))
	plt.scatter(watts_to_dbm(df['peak1_area_W_Hz']), power, label='Stokes Area', s=5)
	plt.scatter(watts_to_dbm(df['peak2_area_W_Hz']), power, label='Rayleigh Area', s=5)
	plt.scatter(watts_to_dbm(df['peak3_area_W_Hz']), power, label='Anti-Stokes Area', s=5)
	plt.ylabel('Power of 10% Beam Splitter (MicroWatt)')
	plt.xlabel('Peak Area (dBm)')
	plt.title('Beam Splitter Power vs Peak Areas')
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'areadBm_vs_power.png'), dpi=300)
	# plt.show()  # Commented for fast batch processing

	# Plot 6: x = power, y = area (axes swapped)
	plt.figure(figsize=(10,6))
	plt.scatter(power, watts_to_dbm(df['peak1_area_W_Hz']), label='Stokes Area', s=5)
	plt.scatter(power, watts_to_dbm(df['peak2_area_W_Hz']), label='Rayleigh Area', s=5)
	plt.scatter(power, watts_to_dbm(df['peak3_area_W_Hz']), label='Anti-Stokes Area', s=5)
	plt.xlabel('Power of 10% Beam Splitter (MicroWatt)')
	plt.ylabel('Peak Area (dBm)')
	plt.title('Peak Areas vs Beam Splitter Power')
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'power_vs_areadBm.png'), dpi=300)
	# plt.show()  # Commented for fast batch processing

def analyseSession_2():
	# Load the CSV
	df = pd.read_csv('OSA_Measurements_Analysis/Session_2/all_peaks_summary.csv')

	power = df['powerOf10PercentBeamSplitter(MicroWatt)']

	# Make 'images' folder next to the CSV if it doesn't exist
	images_folder = "OSA_Measurements_Analysis/Session_2/images"
	'''
	Not useful data
	# Plot 1: x = amplitude, y = power
	plt.figure(figsize=(10,6))
	plt.scatter(df['peak1_amp'], power, label='1st Peak')
	plt.scatter(df['peak2_amp'], power, label='2nd Peak')
	plt.scatter(df['peak3_amp'], power, label='3rd Peak')
	plt.scatter(df['peak4_amp'], power, label='4th Peak')
	plt.scatter(df['peak5_amp'], power, label='5th/Main Peak')
	plt.scatter(df['peak6_amp'], power, label='6th/Extra Peak')
	plt.ylabel('Power of 10% Beam Splitter (MicroWatt)')
	plt.xlabel('Peak Amplitude (dBm)')
	plt.title('Beam Splitter Power vs Peak Amplitudes')
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'amplitude_vs_power.png'), dpi=300)
	#plt.show() # Commented for fast batch processing

	# Plot 2: x = power, y = amplitude (axes swapped)
	plt.figure(figsize=(10,6))
	plt.scatter(power, df['peak1_amp'], label='1st Peak')
	plt.scatter(power, df['peak2_amp'], label='2nd Peak')
	plt.scatter(power, df['peak3_amp'], label='3rd Peak')
	plt.scatter(power, df['peak4_amp'], label='4th Peak')
	plt.scatter(power, df['peak5_amp'], label='5th/Main Peak')
	plt.scatter(power, df['peak6_amp'], label='6th/Extra Peak')
	plt.xlabel('Power of 10% Beam Splitter (MicroWatt)')
	plt.ylabel('Peak Amplitude (dBm)')
	plt.title('Peak Amplitudes vs Beam Splitter Power')
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'power_vs_amplitude.png'), dpi=300)
	#plt.show() # Commented for fast batch processing
	'''
	# Plot 3: x = area, y = power
	plt.figure(figsize=(10,6))
	plt.scatter(df['peak1_area_W_Hz'], power, label='1st Area', s=5)
	plt.scatter(df['peak2_area_W_Hz'], power, label='2nd Area', s=5)
	plt.scatter(df['peak3_area_W_Hz'], power, label='3rd Area', s=5)
	plt.scatter(df['peak4_area_W_Hz'], power, label='4th Area', s=5)
	plt.scatter(df['peak5_area_W_Hz'], power, label='5th/Main Area', s=5)
	plt.scatter(df['peak6_area_W_Hz'], power, label='6th/Extra Area', s=5)
	plt.ylabel('Power of 10% Beam Splitter (MicroWatt)')
	plt.xlabel('Peak Area (Watt × Hz)')
	plt.title('Beam Splitter Power vs Peak Areas')
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'area_vs_power.png'), dpi=300)
	# plt.show()  # Commented for fast batch processing

	# Plot 4: x = power, y = area (axes swapped)
	plt.figure(figsize=(10,6))
	plt.scatter(power, df['peak1_area_W_Hz'], label='1st Area', s=5)
	plt.scatter(power, df['peak2_area_W_Hz'], label='2nd Area', s=5)
	plt.scatter(power, df['peak3_area_W_Hz'], label='3rd Area', s=5)
	plt.scatter(power, df['peak4_area_W_Hz'], label='4th Area', s=5)
	plt.scatter(power, df['peak5_area_W_Hz'], label='5th/Main Area', s=5)
	plt.scatter(power, df['peak6_area_W_Hz'], label='6th/Extra Area', s=5)
	plt.xlabel('Power of 10% Beam Splitter (MicroWatt)')
	plt.ylabel('Peak Area (Watt × Hz)')
	plt.title('Peak Areas vs Beam Splitter Power')
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'power_vs_area.png'), dpi=300)
	# plt.show()  # Commented for fast batch processing

	# Plot 5: x = area, y = power
	plt.figure(figsize=(10,6))
	plt.scatter(watts_to_dbm(df['peak1_area_W_Hz']), power, label='1st Area', s=5)
	plt.scatter(watts_to_dbm(df['peak2_area_W_Hz']), power, label='2nd Area', s=5)
	plt.scatter(watts_to_dbm(df['peak3_area_W_Hz']), power, label='3rd Area', s=5)
	plt.scatter(watts_to_dbm(df['peak4_area_W_Hz']), power, label='4th Area', s=5)
	plt.scatter(watts_to_dbm(df['peak5_area_W_Hz']), power, label='5th/Main Area', s=5)
	plt.scatter(watts_to_dbm(df['peak6_area_W_Hz']), power, label='6th/Extra Area', s=5)
	plt.ylabel('Power of 10% Beam Splitter (MicroWatt)')
	plt.xlabel('Peak Area (dBm)')
	plt.title('Beam Splitter Power vs Peak Areas')
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'areadBm_vs_power.png'), dpi=300)
	# plt.show()  # Commented for fast batch processing

	# Plot 6: x = power, y = area (axes swapped)
	plt.figure(figsize=(10,6))
	plt.scatter(power, watts_to_dbm(df['peak1_area_W_Hz']), label='1st Area', s=5)
	plt.scatter(power, watts_to_dbm(df['peak2_area_W_Hz']), label='2nd Area', s=5)
	plt.scatter(power, watts_to_dbm(df['peak3_area_W_Hz']), label='3rd Area', s=5)
	plt.scatter(power, watts_to_dbm(df['peak4_area_W_Hz']), label='4th Area', s=5)
	plt.scatter(power, watts_to_dbm(df['peak5_area_W_Hz']), label='5th/Main Area', s=5)
	plt.scatter(power, watts_to_dbm(df['peak6_area_W_Hz']), label='6th/Extra Area', s=5)
	plt.xlabel('Power of 10% Beam Splitter (MicroWatt)')
	plt.ylabel('Peak Area (dBm)')
	plt.title('Peak Areas vs Beam Splitter Power')
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'power_vs_areadBm.png'), dpi=300)
	# plt.show()  # Commented for fast batch processing
	
def analyseSessionExtra_2():
	# Load the CSV
	df = pd.read_csv('OSA_Measurements_Analysis/Session_2/all_peaks_summary.csv')

	power = df['powerOf10PercentBeamSplitter(MicroWatt)']

	# Make 'images' folder next to the CSV if it doesn't exist
	images_folder = "OSA_Measurements_Analysis/Session_2/images"
	# Extract each peak-frequency column into its own list
	peak_lists = [df[f'peak{i}_freq'].tolist() for i in range(1, 7)]
	peak_widths = [df[f'peak{i}_width'].tolist() for i in range(1, 7)]
	peak_amplitudes = [df[f'peak{i}_amp'].tolist() for i in range(1, 7)]
	baselines = [df[f'baseline'].tolist()]
	while 0.0 in peak_lists[len(peak_lists)-1]:
		peak_lists[len(peak_lists)-1].remove(0.0)
	while 0.0 in peak_widths[len(peak_widths)-1]:
		peak_widths[len(peak_widths)-1].remove(0.0)
	while 0.0 in peak_amplitudes[len(peak_amplitudes)-1]:
		peak_amplitudes[len(peak_amplitudes)-1].remove(0.0)
	# Compute mean & SEM for each
	means = [np.mean(lst) for lst in peak_lists]
	sems  = [np.std(lst, ddof=1)/np.sqrt(len(lst)) for lst in peak_lists]

	# Prepare x axis as the peak number (1–6)
	x = np.arange(1, 7)

	# Plot with error bars
	plt.figure(figsize=(8,5))
	plt.errorbar(x, means, yerr=sems, marker='o', linestyle='None')
	# 1) compute the adjacent differences
	diffs = np.diff(means)   # length 5, Δ1 = mean2-mean1, etc.

	# 2) draw & annotate each Δ
	for i, d in enumerate(diffs, start=1):
		# x‐position exactly between peak i and i+1
		x_mid = i + 0.5           
		# y‐positions at the two means
		y_low  = means[i-1]
		y_high = means[i]
		# draw a dashed vertical line
		plt.vlines(x_mid, ymin=y_low, ymax=y_high, linestyles='--', color='gray')
		# annotate the gap
		plt.text(
			x_mid, 
			(y_low+y_high)/2, 
			f'{d:.2e} Hz', 
			ha='center', 
			va='bottom',
			fontsize=8,
			rotation=0
		)
	plt.xlabel('Peak Number')
	plt.ylabel('Mean Peak Frequency (Hz)')
	plt.title('Mean Lorentzian Peak Frequencies ± SEM')
	plt.tight_layout()

	# Save out the figure
	plt.savefig(os.path.join(images_folder, 'mean_peak_frequencies.png'), dpi=300)
	plt.close()
	
	'''
		Not exactly sure what the peaks we are looking for just yet. This is the next step in the project.
	'''
	print(peak_widths[4][0])
	print("-----------------")
	print(peak_amplitudes[4][0])
	print("-----------------")
	print(baselines[0][0])
	print("-----------------")
	'''
	gamma   = peak_widths[4][0]     # Linewidht for the 5th Peak 
	amp     = peak_amplitudes[4][0] + baselines[0][0]    # Amplitude for the 5th Peak
	baseline= dbm_to_watts(baselines[0][0])    # your y0

	# compute gain contrast
	G = amp / baseline

	# then the Brillouin linewidth (in Hz) is
	delta_nu_B = gamma * np.sqrt( G/np.log((np.exp(G)+1)/2) - 1 )

	print(f"Brillouin linewidth = {delta_nu_B:.2f} Hz")
	'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=str, default='session_1', choices=['session_1', 'session_2', 'session_2_freq'],
                        help='Choose which session to analyze (default: session_1)')
    args = parser.parse_args()

    if args.session == 'session_1':
        # Session 1 code
        print("Running code for SESSION 1")
        analyseSession_1()
    elif args.session == 'session_2':
        # Session 2 code
        print("Running code for SESSION 2")
        analyseSession_2()
    elif args.session == 'session_2_freq':
    	# Extra session 2
    	print("Running code for Extra Session 2")
    	analyseSessionExtra_2()











