#This will just order the all_peaks.csv files in the first two sessions so it is easier to compare them.

import pandas as pd
import os

# --- Configuration ---
# A list of the full paths to the CSV files you want to sort.
input_filepaths = [
    'Measurements_Analysis/Session_1/all_peaks_summary.csv',
    'Measurements_Analysis/Session_2/all_peaks_summary.csv'
]
# The name of the column you want to sort by.
sort_column = 'powerOf10PercentBeamSplitter(MicroWatt)'

# --- Script ---
# Loop through each file path provided in the list.
for input_filepath in input_filepaths:
    print(f"--- Processing file: {input_filepath} ---")
    
    # Automatically create the output filename by adding '_sorted' before the extension.
    # For example, '.../file.csv' becomes '.../file_sorted.csv'.
    path_without_extension, extension = os.path.splitext(input_filepath)
    output_filepath = f"{path_without_extension}_sorted{extension}"

    try:
        # Read the CSV file into a pandas DataFrame.
        # A DataFrame is like a table, making it easy to work with structured data.
        df = pd.read_csv(input_filepath)

        # Sort the DataFrame by the specified column in ascending order.
        # `inplace=False` means it returns a new sorted DataFrame, leaving the original unchanged.
        df_sorted = df.sort_values(by=sort_column, ascending=True)

        # Save the sorted DataFrame to a new CSV file.
        # `index=False` prevents pandas from writing the DataFrame index as a column in the new file.
        df_sorted.to_csv(output_filepath, index=False)

        print(f"Successfully sorted the file!")
        print(f"Sorted data saved to: {output_filepath}\n")

    except FileNotFoundError:
        print(f"Error: The file was not found at '{input_filepath}'")
        print("Please make sure the file path and name are correct.\n")
    except KeyError:
        print(f"Error: The column '{sort_column}' was not found in the CSV file.")
        print("Please check the column name for typos or case sensitivity.\n")
    except Exception as e:
        print(f"An unexpected error occurred: {e}\n")


