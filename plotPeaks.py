import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV
df = pd.read_csv('OSA_Measurements_Analysis/Session_1/all_peaks_summary.csv')

power = df['powerOf10PercentBeamSplitter(MicroWatt)']

# Make 'images' folder next to the CSV if it doesn't exist
images_folder = "OSA_Measurements_Analysis/Session_1/images"

# Plot 1: x = amplitude, y = power
plt.figure(figsize=(10,6))
plt.scatter(df['peak1_amp'], power, label='Peak 1 Amplitude')
plt.scatter(df['peak2_amp'], power, label='Peak 2 Amplitude')
plt.scatter(df['peak3_amp'], power, label='Peak 3 Amplitude')
plt.ylabel('Power of 10% Beam Splitter (MicroWatt)')
plt.xlabel('Peak Amplitude (dBm)')
plt.title('Beam Splitter Power vs Peak Amplitudes')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(images_folder, 'amplitude_vs_power.png'), dpi=300)
plt.show()

# Plot 2: x = power, y = amplitude (axes swapped)
plt.figure(figsize=(10,6))
plt.scatter(power, df['peak1_amp'], label='Peak 1 Amplitude')
plt.scatter(power, df['peak2_amp'], label='Peak 2 Amplitude')
plt.scatter(power, df['peak3_amp'], label='Peak 3 Amplitude')
plt.xlabel('Power of 10% Beam Splitter (MicroWatt)')
plt.ylabel('Peak Amplitude (dBm)')
plt.title('Peak Amplitudes vs Beam Splitter Power')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(images_folder, 'power_vs_amplitude.png'), dpi=300)
plt.show()
