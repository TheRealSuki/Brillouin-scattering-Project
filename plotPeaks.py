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
	df = pd.read_csv('Measurements_Analysis/Session_1/all_peaks_summary.csv')

	power = df['input_power_uW']

	for i in range(0,len(power)):
		power[i] = power[i] / 1000  # Convert to mW

	# Make 'images' folder next to the CSV if it doesn't exist
	images_folder = "Measurements_Analysis/Session_1/images"

	# Plot 1: x = amplitude, y = power
	plt.figure(figsize=(10,6))
	plt.scatter(df['peak1_amp'], power, label='Stokes Amplitude')
	plt.scatter(df['peak2_amp'], power, label='Rayleigh Amplitude')
	plt.scatter(df['peak3_amp'], power, label='Anti-Stokes Amplitude')
	plt.ylabel('Input power (mW)')
	plt.xlabel('Peak Amplitude (dBm)')
	plt.title('Input Power vs Peak Amplitudes')
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'power_vs_amplitude.png'), dpi=300)
	plt.show()

	# Plot 2: x = power, y = amplitude (axes swapped)
	plt.figure(figsize=(10,6))
	plt.scatter(power, df['peak1_amp'], label='Stokes Amplitude')
	plt.scatter(power, df['peak2_amp'], label='Rayleigh Amplitude')
	plt.scatter(power, df['peak3_amp'], label='Anti-Stokes Amplitude')
	plt.xlabel('Input power (mW)')
	plt.ylabel('Peak Amplitude (dBm)')
	plt.title('Peak Amplitudes vs Input Power')
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'amplitude_vs_power.png'), dpi=300)
	plt.show()

	# Plot 3: x = area, y = power
	plt.figure(figsize=(10,6))
	plt.scatter(dbm_to_watts(watts_to_dbm(df['peak1_area_W_Hz']) + df['baseline']) / 1e-3, power, label='Stokes Area', s=5)
	plt.scatter(dbm_to_watts(watts_to_dbm(df['peak2_area_W_Hz']) + df['baseline']) / 1e-3, power, label='Rayleigh Area', s=5)
	plt.scatter(dbm_to_watts(watts_to_dbm(df['peak3_area_W_Hz']) + df['baseline']) / 1e-3, power, label='Anti-Stokes Area', s=5)
	plt.ylabel('Input power (mW)')
	plt.xlabel('Peak Area (mW)')
	plt.title('Input Power vs Peak Areas')
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'power_vs_area.png'), dpi=300)
	# plt.show()  # Commented for fast batch processing

	# Plot 4: x = power, y = area (axes swapped)
	plt.figure(figsize=(10,6))
	plt.scatter(power, dbm_to_watts(watts_to_dbm(df['peak1_area_W_Hz']) + df['baseline']) / 1e-3, label='Stokes Area', s=5)
	plt.scatter(power, dbm_to_watts(watts_to_dbm(df['peak2_area_W_Hz']) + df['baseline']) / 1e-3, label='Rayleigh Area', s=5)
	plt.scatter(power, dbm_to_watts(watts_to_dbm(df['peak3_area_W_Hz']) + df['baseline']) / 1e-3, label='Anti-Stokes Area', s=5)
	plt.xlabel('Input power (mW)')
	plt.ylabel('Peak Area (mW)')
	plt.title('Peak Areas vs Input Power')
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'area_vs_power.png'), dpi=300)
	# plt.show()  # Commented for fast batch processing

	# Plot 5: x = area, y = power
	plt.figure(figsize=(10,6))
	plt.scatter(watts_to_dbm(df['peak1_area_W_Hz']), power, label='Stokes Area', s=5)
	plt.scatter(watts_to_dbm(df['peak2_area_W_Hz']), power, label='Rayleigh Area', s=5)
	plt.scatter(watts_to_dbm(df['peak3_area_W_Hz']), power, label='Anti-Stokes Area', s=5)
	plt.ylabel('Input power (mW)')
	plt.xlabel('Peak Area (dBm)')
	plt.title('Input Power vs Peak Areas')
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'power_vs_areadBm.png'), dpi=300)
	# plt.show()  # Commented for fast batch processing

	# Plot 6: x = power, y = area (axes swapped)
	plt.figure(figsize=(10,6))
	plt.scatter(power, watts_to_dbm(df['peak1_area_W_Hz']) + df['baseline'], label='Stokes Area', s=5)
	plt.scatter(power, watts_to_dbm(df['peak2_area_W_Hz']) + df['baseline'], label='Rayleigh Area', s=5)
	plt.scatter(power, watts_to_dbm(df['peak3_area_W_Hz']) + df['baseline'], label='Anti-Stokes Area', s=5)
	plt.xlabel('Input power (mW)')
	plt.ylabel('Peak Area (dBm)')
	plt.title('Peak Areas vs Input Power')
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'areadBm_vs_power.png'), dpi=300)
	# plt.show()  # Commented for fast batch processing

def analyseSession_2():
	# Load the CSV
	df = pd.read_csv('Measurements_Analysis/Session_2/all_peaks_summary.csv')

	power = df['input_power_uW']

	for i in range(0,len(power)):
		power[i] = power[i] / 1000  # Convert to mW

	# Make 'images' folder next to the CSV if it doesn't exist
	images_folder = "Measurements_Analysis/Session_2/images"
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
	plt.scatter(df['peak1_area_W_Hz'] / 1e-6, power, label='1st Area', s=5)
	plt.scatter(df['peak2_area_W_Hz'] / 1e-6, power, label='2nd Area', s=5)
	plt.scatter(df['peak3_area_W_Hz'] / 1e-6, power, label='3rd Area', s=5)
	plt.scatter(df['peak4_area_W_Hz'] / 1e-6, power, label='4th Area', s=5)
	plt.scatter(df['peak5_area_W_Hz'] / 1e-6, power, label='5th/Main Area', s=5)
	#plt.scatter(df['peak6_area_W_Hz'], power, label='6th/Extra Area', s=5) # Ignored as this is the Phantom peak
	plt.ylabel('Input Power (mW)')
	plt.xlabel(r'Peak Area ($\mu$W)')
	plt.title('Input Power vs Peak Areas')
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'power_vs_area.png'), dpi=300)
	# plt.show()  # Commented for fast batch processing

	# Plot 4: x = power, y = area (axes swapped)
	plt.figure(figsize=(10,6))
	plt.scatter(power, df['peak1_area_W_Hz'] / 1e-6, label='1st Area', s=5)
	plt.scatter(power, df['peak2_area_W_Hz'] / 1e-6, label='2nd Area', s=5)
	plt.scatter(power, df['peak3_area_W_Hz'] / 1e-6, label='3rd Area', s=5)
	plt.scatter(power, df['peak4_area_W_Hz'] / 1e-6, label='4th Area', s=5)
	plt.scatter(power, df['peak5_area_W_Hz'] / 1e-6, label='5th/Main Area', s=5)
	#plt.scatter(power, df['peak6_area_W_Hz'], label='6th/Extra Area', s=5) # Ignored as this is the Phantom peak
	plt.xlabel('Input Power (mW)')
	plt.ylabel(r'Peak Area ($\mu$W)')
	plt.title('Peak Areas vs Input Power')
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'area_vs_power.png'), dpi=300)
	# plt.show()  # Commented for fast batch processing

	# Plot 5: x = area, y = power
	plt.figure(figsize=(10,6))
	plt.scatter(watts_to_dbm(df['peak1_area_W_Hz']), power, label='1st Area', s=5)
	plt.scatter(watts_to_dbm(df['peak2_area_W_Hz']), power, label='2nd Area', s=5)
	plt.scatter(watts_to_dbm(df['peak3_area_W_Hz']), power, label='3rd Area', s=5)
	plt.scatter(watts_to_dbm(df['peak4_area_W_Hz']), power, label='4th Area', s=5)
	plt.scatter(watts_to_dbm(df['peak5_area_W_Hz']), power, label='5th/Main Area', s=5)
	#plt.scatter(watts_to_dbm(df['peak6_area_W_Hz']), power, label='6th/Extra Area', s=5) # Ignored as this is the Phantom peak
	plt.ylabel('Input Power (mW)')
	plt.xlabel('Peak Area (dBm)')
	plt.title('Input Power vs Peak Areas')
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'power_vs_areadBm.png'), dpi=300)
	# plt.show()  # Commented for fast batch processing

	# Plot 6: x = power, y = area (axes swapped)
	plt.figure(figsize=(10,6))
	plt.scatter(power, watts_to_dbm(df['peak1_area_W_Hz']), label='1st Area', s=5)
	plt.scatter(power, watts_to_dbm(df['peak2_area_W_Hz']), label='2nd Area', s=5)
	plt.scatter(power, watts_to_dbm(df['peak3_area_W_Hz']), label='3rd Area', s=5)
	plt.scatter(power, watts_to_dbm(df['peak4_area_W_Hz']), label='4th Area', s=5)
	plt.scatter(power, watts_to_dbm(df['peak5_area_W_Hz']), label='5th/Main Area', s=5)
	#plt.scatter(power, watts_to_dbm(df['peak6_area_W_Hz']), label='6th/Extra Area', s=5) # Ignored as this is the Phantom peak
	plt.xlabel('Input Power (mW)')
	plt.ylabel('Peak Area (dBm)')
	plt.title('Peak Areas vs Input Power')
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(images_folder, 'areadBm_vs_power.png'), dpi=300)
	# plt.show()  # Commented for fast batch processing
	
def analyseSessionExtra_2():
    """
    Analyzes peak frequency data from a CSV file, converts frequencies to MHz,
    and plots the mean peak frequencies with error bars and annotations.
    """
    # Define the path to the CSV file
    csv_path = 'Measurements_Analysis/Session_2/all_peaks_summary.csv'

    # Check if the file exists before trying to read it
    if not os.path.exists(csv_path):
        print(f"Error: The file was not found at {csv_path}")
        # Create a dummy dataframe and file to allow the script to run without error
        print("Creating a dummy 'all_peaks_summary.csv' to demonstrate functionality.")
        dummy_data = {
            'input_power_uW': np.linspace(10, 100, 20),
            'baseline': np.random.rand(20) * 1e7
        }
        for i in range(1, 6):
            # Create plausible frequency peaks in the GHz range (e.g., around 2.4 GHz)
            # with some noise and drift related to power.
            dummy_data[f'peak{i}_freq'] = (2.4e9 + (i-1)*5e7) + np.random.randn(20) * 1e6 + dummy_data['input_power_uW'] * 1e3
            dummy_data[f'peak{i}_width'] = np.random.rand(20) * 1e5
            dummy_data[f'peak{i}_amp'] = np.random.rand(20) * -50 + -10

        df = pd.DataFrame(dummy_data)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
    else:
        # Load the CSV
        df = pd.read_csv(csv_path)

    # Make an 'images' folder next to the CSV if it doesn't exist
    images_folder = "Measurements_Analysis/Session_2/images"
    os.makedirs(images_folder, exist_ok=True)

    # Extract each peak-frequency column into its own list
    peak_lists = [df[f'peak{i}_freq'].tolist() for i in range(1, 6)]
    
    # The original code for removing zeros from other lists is maintained here,
    # although it's not used in this specific plotting function.
    peak_widths = [df[f'peak{i}_width'].tolist() for i in range(1, 6)]
    peak_amplitudes = [df[f'peak{i}_amp'].tolist() for i in range(1, 6)]
    
    while 0.0 in peak_lists[-1]:
        peak_lists[-1].remove(0.0)
    while 0.0 in peak_widths[-1]:
        peak_widths[-1].remove(0.0)
    while 0.0 in peak_amplitudes[-1]:
        peak_amplitudes[-1].remove(0.0)

    # Compute mean & SEM for each list (values are in Hz)
    means_hz = [np.mean(lst) for lst in peak_lists]
    sems_hz = [np.std(lst, ddof=1) / np.sqrt(len(lst)) for lst in peak_lists]

    # --- Convert all values from Hz to MHz for plotting ---
    conversion_factor = 1e6
    means_mhz = [m / conversion_factor for m in means_hz]
    sems_mhz = [s / conversion_factor for s in sems_hz]
    
    # Prepare x-axis as the peak number (1-5)
    x = np.arange(1, 6)

    # Plot with error bars using the MHz values
    plt.figure(figsize=(10, 6))
    plt.errorbar(x, means_mhz, yerr=sems_mhz, marker='o', linestyle='None', capsize=5, label='Mean Frequency')
    
    # 1) Compute the adjacent differences in MHz
    diffs_mhz = np.diff(means_mhz)

    # 2) Draw & annotate each difference (Δ)
    for i, d in enumerate(diffs_mhz, start=1):
        # x-position exactly between peak i and i+1
        x_mid = i + 0.5
        # y-positions at the two means (in MHz)
        y_low = means_mhz[i-1]
        y_high = means_mhz[i]
        
        # Draw a dashed vertical line connecting the means
        plt.vlines(x_mid, ymin=y_low, ymax=y_high, linestyles='--', color='gray')
        
        # Annotate the gap with the difference in MHz
        plt.text(
            x=x_mid,
            y=(y_low + y_high) / 2,
            s=f'Δ = {d:.2f} MHz', # Updated label and format for MHz
            ha='center',
            va='bottom',
            fontsize=9,
            rotation=0,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
        )
        
    # --- Final plot styling ---
    plt.xlabel('Peak Number')
    plt.ylabel('Mean Peak Frequency (MHz)')
    plt.title('Mean Lorentzian Peak Frequencies')
    plt.xticks(x) # Ensure x-axis ticks are integers for peak numbers
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(images_folder, 'mean_peak_frequencies_MHz.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    	
def plotg20comparison():
    """
    Analyzes multiple folders of g^(2)(τ) data to find the highest g^(2)(0)
    achieved with each filter setup.
    """
    # =========================================================================
    # 1. Define the Folder Paths and Names for Each Filter Setup
    # =========================================================================
    
    base_path = 'Measurements/Session_3_Photon_Statistics_And_Heterodyne/'
    
    filter_setups = {
        '2GHz Filter': os.path.join(base_path, '3_2GHz_187microWatt_slightly_lasing/data_photon_stat'),
        'Cavity Filter': os.path.join(base_path, '2_Cavity_Filter_focusing_on_one_mode/data_photon_stat'),
        'Cavity + 2GHz Filters': os.path.join(base_path, '1_Cavity_and_2GHz_Filters/data_photon_stat'),
        '2GHz + 10GHz + 4GHz Filters' : '/home/apolloin/Desktop/Brillouin_Scattering/Brillouin-scattering-Project/Measurements/Session_4_Photon_Statistics_More_Filters_No_Heterodyne/2GHz10GHzAOS4GHz',
        '2GHz + 4GHz + 10GHz Filters' : '/home/apolloin/Desktop/Brillouin_Scattering/Brillouin-scattering-Project/Measurements/Session_4_Photon_Statistics_More_Filters_No_Heterodyne/2GH4GHzAOS10GHz'
    }
    
    # Dictionary to store the best result for each filter
    best_results = {}

    # =========================================================================
    # 2. Loop Through Each Folder and Analyze All CSV Files
    # =========================================================================
    
    for filter_name, folder_path in filter_setups.items():
        print(f"\n--- Analyzing Folder: {filter_name} ---")
        
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder not found at {folder_path}. Skipping.")
            continue
            
        highest_g20 = 0
        best_file = None
        
        # Get a list of all csv files in the directory
        try:
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            if not csv_files:
                print("No CSV files found in this directory.")
                continue
        except FileNotFoundError:
            print(f"Error: Could not access folder path: {folder_path}")
            continue

        for file_name in csv_files:
            full_path = os.path.join(folder_path, file_name)
            
            try:
                # Load the data using pandas
                data = pd.read_csv(full_path, header=0)
                data.columns = ['Time_ns', 'Counts']
                
                # --- Normalize the data to get g^(2)(τ) ---
                # Find the baseline by averaging the last 25% of the data
                baseline_window = int(len(data) * 0.25)
                if baseline_window == 0: continue # Skip tiny files
                
                baseline_counts = data['Counts'][-baseline_window:].mean()
                if baseline_counts == 0: continue # Avoid division by zero
                
                g2_experimental = data['Counts'] / baseline_counts
                
                # The peak g^(2)(0) is the maximum value of the normalized data
                current_g20 = g2_experimental.max()
                
                # Check if this is the best result for this folder so far
                if current_g20 > highest_g20:
                    highest_g20 = current_g20
                    best_file = file_name
                    
            except Exception as e:
                print(f"Could not process file {file_name}. Error: {e}")
        
        # Store the best result for this filter setup
        if best_file:
            best_results[filter_name] = {'g20': highest_g20, 'file': best_file}
            print(f"Highest g^(2)(0) found: {highest_g20:.3f} in file: {best_file}")
        else:
            print("No valid data files could be processed in this folder.")

    # =========================================================================
    # 3. Manually Add Additional Results
    # =========================================================================
    print("\nAdding manual data points...")
    best_results['2GHz and 4GHz Filter'] = {'g20': 1.51, 'file': 'Manual Entry'}
    best_results['AOS Filter (10GHz and 30 GHz Combinations)'] = {'g20': 1.0, 'file': 'Manual Entry'}


    # =========================================================================
    # 4. Print Summary and Plot the Results
    # =========================================================================
    
    print("\n\n--- Final Summary ---")
    if not best_results:
        print("No results found.")
        return
        
    for filter_name, result in best_results.items():
        print(f"Filter: {filter_name:<35} | Highest g^(2)(0): {result['g20']:.4f} | File: {result['file']}")
    
    # --- Create a Bar Chart for Visual Comparison ---
    names = list(best_results.keys())
    values = [result['g20'] for result in best_results.values()]
    
    #Adding manual texts
    manual_texts = ['1.683mW', '0.63mW', '~1.683mW', '~1.683mW', '~1.683mW', 'Any power']  # Add more texts as needed

    # Add more colors for the new bars
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'orchid']

    plt.figure(figsize=(12, 7))
    bars = plt.bar(names, values, color=colors[:len(names)])
    
    # Add the g^(2)(0) value on top of each bar
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval / 2, manual_texts[i], va='center', ha='center', color='white', fontsize=12, fontweight='bold')
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.3f}', va='bottom', ha='center')

    plt.title('Maximum Achieved $g^{(2)}(0)$ by Filter Configuration', fontsize=16)
    plt.ylabel('$g^{(2)}(0)$ Value', fontsize=12)
    plt.xticks(rotation=15, ha="right") # Rotate labels slightly to prevent overlap
    plt.ylim(top=2.1) # Set y-axis limit slightly above the theoretical max of 2
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def create_g2_plots():
    """
    Finds g2(tau) measurement .csv files in specified directories,
    normalizes the data, creates an individual plot for each, and saves it to a
    dedicated analysis folder.
    """
    # --- Configuration ---
    # NOTE: You might need to adjust these paths to match your computer's file structure.
    base_path_session3 = 'Measurements/Session_3_Photon_Statistics_And_Heterodyne/'
    base_path_session4 = 'Measurements/Session_4_Photon_Statistics_More_Filters_No_Heterodyne/'

    # A dictionary to hold the descriptive names and paths for each measurement setup.
    filter_setups = {
        '2GHz Filter': os.path.join(base_path_session3, '3_2GHz_187microWatt_slightly_lasing/data_photon_stat'),
        'Cavity Filter': os.path.join(base_path_session3, '2_Cavity_Filter_focusing_on_one_mode/data_photon_stat'),
        'Cavity + 2GHz Filters': os.path.join(base_path_session3, '1_Cavity_and_2GHz_Filters/data_photon_stat'),
        '2GHz + 10GHz + 4GHz Filters': os.path.join(base_path_session4, '2GHz10GHzAOS4GHz'),
        '2GHz + 4GHz + 10GHz Filters': os.path.join(base_path_session4, '2GH4GHzAOS10GHz')
    }

    # --- Output Directory ---
    # Define the directory where all the plots will be saved.
    output_directory = 'Measurements_Analysis/Session_3'
    # Create the directory if it doesn't exist. exist_ok=True prevents an error if it's already there.
    os.makedirs(output_directory, exist_ok=True)
    print(f"Plots will be saved to: {os.path.abspath(output_directory)}")

    # --- Data Processing and Plotting Loop ---
    # Iterate over each setup defined in the dictionary.
    for name, path in filter_setups.items():
        try:
            # Find all files in the directory that end with .csv.
            csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
            
            if not csv_files:
                print(f"Warning: No CSV files found in directory: {path}")
                continue
            
            # This loop will process every CSV file found, not just the first one.
            for csv_file in csv_files:
                file_path = os.path.join(path, csv_file)
                print(f"Processing: {file_path}")

                # Read the CSV file using pandas.
                # skiprows=1 tells pandas to skip the first line (the header).
                data = pd.read_csv(file_path, skiprows=1, names=['tau', 'g2'])

                # --- Normalization ---
                tail_start_index = int(len(data['g2']) * 0.75)
                normalization_factor = data['g2'].iloc[tail_start_index:].mean()
                
                if normalization_factor == 0:
                    print(f"Warning: Normalization factor is zero for {name}. Skipping this file.")
                    continue

                g2_normalized = data['g2'] / normalization_factor

                # --- Plotting Setup (New plot for each file) ---
                plt.style.use('seaborn-v0_8-whitegrid')
                fig, ax = plt.subplots(figsize=(12, 7))

                # --- Plot the Data ---
                ax.plot(data['tau'], g2_normalized, label=name, linewidth=2, color='royalblue')

                # --- Plot Customization ---
                title_name = name.replace(' + ', ' & ') # Make title cleaner
                ax.set_title(f'Second-Order Autocorrelation: {title_name}', fontsize=18, fontweight='bold')
                ax.set_xlabel('Time Delay $\\tau$ (ns)', fontsize=14)
                ax.set_ylabel('Normalized Correlation $g^{(2)}(\\tau)$', fontsize=14)
                ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='Uncorrelated (g$^{(2)}$=1)')
                ax.set_ylim(0.5, 2.1)
                ax.legend(fontsize=12)
                ax.tick_params(axis='both', which='major', labelsize=12)
                plt.tight_layout()

                # --- File Saving ---
                # Create a safe and descriptive filename.
                # 1. Sanitize the setup name.
                sanitized_name = name.replace(' ', '_').replace('+', '').replace('/', '_')
                # 2. Get the original filename without the .csv extension.
                csv_basename = os.path.splitext(csv_file)[0]
                # 3. Combine them for the final filename.
                output_filename = f"{sanitized_name}_{csv_basename}.png"
                
                # Construct the full path for saving the file.
                output_path = os.path.join(output_directory, output_filename)
                
                # Save the figure with high resolution (dpi).
                plt.savefig(output_path, dpi=300)
                print(f"  -> Saved plot to: {output_path}")

                # Close the figure to free up memory before the next loop iteration.
                plt.close(fig)

        except FileNotFoundError:
            print(f"Error: The directory was not found: {path}")
        except Exception as e:
            print(f"An error occurred while processing '{name}': {e}")

def create_linewidth_session_1():
    #THIS FILE IS JUST USED TO CREATE THE LINEWIDTH VS POWER SCATTER PLOT FOR SESSION 1


    # --- Configuration ---
    # Path to the input CSV file for Session 1.
    input_filepath = 'Measurements_Analysis/Session_1/all_peaks_summary_sorted.csv'
    # Directory where the output plot image will be saved.
    output_dir = 'Measurements_Analysis/Session_1/images'
    # Filename for the saved plot.
    output_filename = 'linewidth_vs_power_scatter_MHz.png'

    # --- Column Names ---
    # Column for the x-axis data.
    x_column = 'powerOf10PercentBeamSplitter(MicroWatt)'
    # Columns for the y-axis data (the three peak widths).
    y_columns = {
        'peak1_width': 'Stokes',
        'peak2_width': 'Rayleigh',
        'peak3_width': 'Anti-Stokes'
    }

    # --- Script ---
    try:
        # --- 1. Load and Prepare Data ---

        # Read the entire CSV file without skipping rows to ensure the header is read correctly.
        df = pd.read_csv(input_filepath)

        # Skip the first 3 data points by slicing the DataFrame.
        # .iloc[3:] selects all rows from the 4th row (index 3) onwards.
        df_filtered = df.iloc[3:].reset_index(drop=True)


        # Extract the x and y data from the filtered DataFrame.
        x_data = df_filtered[x_column]
        
        # --- 2. Create the Plot ---

        # Initialize a new plot. `figsize` makes the plot wider for better readability.
        plt.figure(figsize=(10, 6))

        # Loop through the peak width columns and plot each one as a scatter plot.
        for col, label in y_columns.items():
            if col in df_filtered.columns:
                # Convert y-axis data from Hz to MHz by dividing by 1e6.
                y_data_mhz = 2*df_filtered[col] / 1e9 # Factor of 2 to convert HWHM to FWHM (linewidth)
                x_data_mW = x_data*9/1000 # Convert to mW
                plt.scatter(x_data_mW, y_data_mhz, label=label)
            else:
                print(f"Warning: Column '{col}' not found in the file. Skipping.")

        # --- 3. Style the Plot ---

        # Add labels to the axes and a title to the plot for clarity.
        plt.xlabel('Power (mW)')
        plt.ylabel('Linewidth (GHz)')
        plt.title('Linewidth(FWHM) vs. Input Power')
        
        # Add a legend to distinguish between the different peaks.
        plt.legend()
        
        # Add a grid for easier reading of values.
        plt.grid(True)
        
        # --- 4. Save the Plot ---

        # Check if the output directory exists. If not, create it.
        # `exist_ok=True` prevents an error if the directory already exists.
        os.makedirs(output_dir, exist_ok=True)

        # Construct the full path for the output file.
        output_filepath = os.path.join(output_dir, output_filename)

        # Save the plot to the specified file.
        # `dpi=300` saves it in high resolution.
        plt.savefig(output_filepath, dpi=300)

        print(f"Scatter plot successfully created and saved to: {output_filepath}")

    except FileNotFoundError:
        print(f"Error: The file was not found at '{input_filepath}'")
        print("Please make sure the file path and name are correct.")
    except KeyError as e:
        print(f"Error: A required column was not found in the CSV file: {e}")
        print("Please check the column names in your configuration.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Choices list
    parser.add_argument('--session', type=str, default='session_1', 
                        choices=['session_1', 'session_2', 'session_2_freq', 'plotg20comparison', 'create_g2_plots', 'create_linewidth_session_1'],
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
    elif args.session == 'plotg20comparison':
        # Plot g2 comparison
        print("Running code for g2 comparison")
        plotg20comparison()
    elif args.session == 'create_g2_plots':
        # Create g2 plots
        print("Running code to create g2 plots")
        create_g2_plots()
    elif args.session == 'create_linewidth_session_1':
        # Create linewidth plots for session 1
        print("Running code to create linewidth plots for SESSION 1")
        create_linewidth_session_1()


