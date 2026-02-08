import os
import re

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