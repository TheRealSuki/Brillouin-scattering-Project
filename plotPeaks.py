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
