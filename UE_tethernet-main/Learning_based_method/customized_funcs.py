import os
import re
import pandas as pd
import numpy as np

# List all filenames in the folder
def get_sorted_filenames(folder_path):
    # folder_path = "./datasets/CNN_MLP_img/"
    filenames = os.listdir(folder_path)

    # Filter for .png files
    png_files = [f for f in filenames if f.endswith(".png")]

    # Define a custom key function
    def sort_key(name):
        # for xx_yy.png, extract xx and yy as integers
        # match = re.match(r"(\d+)_(\d+)\.png", name)
        # if match:
        #     return int(match.group(1)), int(match.group(2))
        # else:
        #     return float('inf'), float('inf')  # put non-matching files at the end

        # for xx_yy.png, extract just xx as integer         
        match = re.match(r"(\d+)_", name)
        if match:
            return int(match.group(1))   # just the first number
        else:
            return float('inf')

    # Sort the list
    sorted_files = sorted(png_files, key=sort_key)
    sorted_files_with_folder = [os.path.join(folder_path, f) for f in sorted_files]

    # Optionally print
    for f in sorted_files_with_folder:
        print(f)

    return sorted_files_with_folder

def normalize_dataframe(df: pd.DataFrame, lower_bounds, upper_bounds):
    """
    Normalize each column in the dataframe using provided lower and upper bounds.
    Normalization: (x - lower) / (upper - lower)
    Args:
        df: pandas DataFrame
        lower_bounds: list or array of lower bounds (length = number of columns)
        upper_bounds: list or array of upper bounds (length = number of columns)
    Returns:
        pandas DataFrame with normalized values
    """
    df_norm = df.copy()
    for i, col in enumerate(df.columns):
        lb = lower_bounds[i]
        ub = upper_bounds[i]
        df_norm[col] = (df[col] - lb) / (ub - lb)
    return df_norm

def denormalize_dataframe(df_norm: pd.DataFrame, lower_bounds, upper_bounds):
    """
    Denormalize each column in the dataframe using provided lower and upper bounds.
    Denormalization: x * (upper - lower) + lower
    Args:
        df_norm: pandas DataFrame with normalized values
        lower_bounds: list or array of lower bounds (length = number of columns)
        upper_bounds: list or array of upper bounds (length = number of columns)
    Returns:
        pandas DataFrame with denormalized values
    """
    df_denorm = df_norm.copy()
    for i, col in enumerate(df_norm.columns):
        lb = lower_bounds[i]
        ub = upper_bounds[i]
        df_denorm[col] = df_norm[col] * (ub - lb) + lb
    return df_denorm

def percept_confidence_MinMaxScaler(original_arr, data_type="input", operation="normalize"):
    """
    Normalize and trucation
    """
    if data_type == "input":
        # X, Y, Z, qw, qx, qy, qz
        l_bounds=np.array([-3.2, -10.00,-60.00,-1.0,-1.0,-1.0,-1.0])
        u_bounds=np.array([3.5, 10.00,-34.00,1.0,1.0,1.0,1.0])

    elif data_type == "output":
        # log_err from URSO-Net predictions of df_log_err_train_6k.pkl
        l_bounds=np.array([0.0])
        u_bounds=np.array([2.8])

    if operation == "normalize":
        result_arr = (original_arr - l_bounds) / (u_bounds - l_bounds)

    elif operation == "denormalize":
        result_arr = original_arr * (u_bounds - l_bounds) + l_bounds

    return result_arr