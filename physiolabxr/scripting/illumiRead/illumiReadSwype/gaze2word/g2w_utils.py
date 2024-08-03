import numpy as np
import pandas as pd


def load_trace_file(file_path):
    trace = pd.read_csv(file_path)

    # Initialize the list to hold arrays
    trace_list = []

    # Temporary list to hold current array
    current_array = []

    # Iterate over the rows of the dataframe
    for index, row in trace.iterrows():
        if pd.isna(row['KeyBoardLocalY']):
            # If the row contains only one element, start a new array
            if current_array:
                trace_list.append(np.array(current_array).astype(float))
                current_array = []
        else:
            # Append the row to the current array
            current_array.append(row.tolist())

    return trace_list