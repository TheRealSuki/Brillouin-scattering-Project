import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Assuming this file analyses the weird data from Session 4 photon counting. (1GHz4GHz10GHzAOS Filter ones.)


def analyze_photon_data(filename):
    """
    Loads photon counting data, calculates the g2(τ) correlation function,
    determines the baseline and g2(0), and plots the results.

    Args:
        filename (str): The path to the CSV data file.
    """
    try:
        # 1. Load the data using pandas
        # We use `usecols=[0, 1]` because the source CSV has a trailing comma,
        # which can be misinterpreted as a third, empty column. This ensures
        # we only load the first two columns of data.
        data = pd.read_csv(filename, usecols=[0, 1])
        
        # Rename columns for clarity and consistency within the script.
        data.columns = ['Time_ns', 'Counts']

        time_tau = data['Time_ns'].to_numpy()
        counts = data['Counts'].to_numpy()

        print(f"Successfully loaded data from '{filename}'.")
        print(f"Found {len(data)} data points.")

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        print("Please make sure the script is in the correct directory or provide the full path.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # 2. Calculate the baseline count for normalization
    # We assume the correlation function settles to its baseline value at long
    # delay times. Here, we'll use the last 25% of the data to calculate this.
    baseline_region_start_index = int(len(counts) * 0.75)
    baseline_counts = counts[baseline_region_start_index:]
    
    # The baseline is the average of the counts in this region.
    baseline_mean = np.mean(baseline_counts)
    
    # The number of data points used to calculate the baseline.
    n_baseline = len(baseline_counts)

    if baseline_mean == 0 or n_baseline == 0:
        print("Error: Could not calculate a valid baseline. Check your data.")
        return
        
    print(f"\n--- Calculations ---")
    print(f"Baseline count (average of last 25% of data): {baseline_mean:.2f}")

    # 3. Normalize the data to get g2(τ)
    g2_tau = counts / baseline_mean

    # 4. Find g2(0), the highest value of the g2 function
    # This is often the point of maximum correlation (or anti-correlation).
    g2_zero_index = np.argmax(g2_tau)
    g2_zero_value = g2_tau[g2_zero_index]
    time_at_g2_zero = time_tau[g2_zero_index]

    print(f"g2(0) [Peak Value]: {g2_zero_value:.3f} at τ = {time_at_g2_zero} ns")
    
    # 5. Plot the results
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot g2(τ) as a scatter plot (without error bars)
    ax.plot(time_tau, g2_tau, 'o', markersize=4, alpha=0.7, 
            label='Experimental Data', color='royalblue')

    # Highlight the g2(0) point
    ax.plot(time_at_g2_zero, g2_zero_value, 'o', color='crimson', markersize=8,
            label=f'$g^2(0)$ = {g2_zero_value:.3f}')

    # Draw a line for the normalized baseline (g2 = 1)
    ax.axhline(y=1, color='gray', linestyle='--', label='Baseline (g² = 1)')

    # --- Formatting the plot ---
    ax.set_title('Second-Order Correlation Function $g^2(\\tau)$', fontsize=16)
    ax.set_xlabel('Time Delay $\\tau$ (ns)', fontsize=12)
    ax.set_ylabel('Normalized Coincidence Counts $g^2(\\tau)$', fontsize=12)
    ax.legend(fontsize=10)
    
    # Add text box with key information
    info_text = (
        f"Baseline Count: {baseline_mean:.2f}\n"
        f"$g^2(0)$: {g2_zero_value:.3f}"
    )
    ax.text(0.95, 0.95, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


# --- Main execution ---
if __name__ == '__main__':
    # The filename you provided.
    # IMPORTANT: Make sure this file is in the same directory as the script,
    # or provide the full, absolute path to the file.
    filename = 'Measurements/Session_4_Photon_Statistics_More_Filters_No_Heterodyne/1GH4GHzAOS10GHz/1GHz4Ghz10GHzAOS_0.1nFilter.csv'
    
    # Run the analysis
    analyze_photon_data(filename)
