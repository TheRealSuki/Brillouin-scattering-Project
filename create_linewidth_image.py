import pandas as pd
import matplotlib.pyplot as plt
import os

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
            plt.scatter(x_data, y_data_mhz, label=label)
        else:
            print(f"Warning: Column '{col}' not found in the file. Skipping.")

    # --- 3. Style the Plot ---

    # Add labels to the axes and a title to the plot for clarity.
    plt.xlabel('Power (μW)')
    plt.ylabel('Linewidth (GHz)')
    plt.title('Linewidth(FWHM) vs. Input Power (10% Beam Splitter Port)')
    
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

