"""
Author: Feng Liu
Date: August 2025

This script trains a Multi-Layer Perceptron (MLP) regressor using PyTorch with detailed logs
"""


# import scipy.io
import os
import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.metrics import confusion_matrix, f1_score
import torch
from torch import nn
# from torch.utils.data import DataLoader, TensorDataset, Dataset
# from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_sequence
import torch.optim as optim
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from architectures import MLP_Regressor
import pickle
import joblib
import time
from matplotlib import pyplot as plt

model_id = 1
model_folder = f"./MLP_training/train_log_{model_id}"

num_sample_per_scenario = 10
est_pos_cols = [f"Est_PX{i}" for i in range(1,num_sample_per_scenario+1)] + \
            [f"Est_PY{i}" for i in range(1,num_sample_per_scenario+1)] + \
            [f"Est_PZ{i}" for i in range(1,num_sample_per_scenario+1)]
est_ori_cols = [f"Est_OX{i}" for i in range(1,num_sample_per_scenario+1)] + \
            [f"Est_OY{i}" for i in range(1,num_sample_per_scenario+1)] + \
            [f"Est_OZ{i}" for i in range(1,num_sample_per_scenario+1)]
input_cols = est_pos_cols + est_ori_cols

output_cols = ["Gt_PX", "Gt_PY", "Gt_PZ"]

def run_MLmodel(model_inputs):
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            # print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    # epoch_id = 1500
    
    model_dict_savename_best_roll = f"{model_folder}/ICRA2026_MLP_dict{model_id}_best_roll.pth"
    # model_inputs = np.array(model_inputs).reshape(1, 33)    

    # hd_set = [64,32]
    hd_set = [256,128,64,32,16]    
    # model = CNNBinaryClassifier(input_size=len(input_cols), num_filters=8, kernel_size=3, hidden_size=256*2).to(device)
    # model = MLPBinaryClassifier(input_size=len(input_cols), hidden_size=hd_set, output_size=len(output_cols)).to(device)
    model = MLP_Regressor(input_size=len(input_cols), hidden_size=hd_set, output_size=len(output_cols)).to(device)


    # state_dict_path = f"./{model_folder}/ICRA2025_MLP_CAP_dict{model_id}.pth"
    state_dict_path = model_dict_savename_best_roll
    model.load_state_dict(torch.load(state_dict_path, weights_only=True))

    # model = torch.load(f"./{model_folder}/ICRA2025_MLP_CAP{model_id}.pth").to(device)
    input_scaler = joblib.load(f"./{model_folder}/ICRA2026_MLP_input_scaler{model_id}.pkl")
    output_scaler = joblib.load(f"./{model_folder}/ICRA2026_MLP_output_scaler{model_id}.pkl")
    model_inputs = pd.DataFrame(model_inputs, columns=input_cols)
    model_inputs_scaled = input_scaler.transform(model_inputs)

    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(model_inputs_scaled, dtype=torch.float32).to(device))
    # return torch.round(outputs.cpu()).numpy().ravel()
    outputs = output_scaler.inverse_transform(outputs.cpu())
    
    return outputs

with open(f"./{model_folder}/df_train_and_df_val.pkl", "rb") as f:
    _, df_val = pickle.load(f)


model_in = df_val[input_cols]
model_in = model_in.values
out = run_MLmodel(model_in)
# out = np.expm1(out)  # Inverse of log1p transformation
df_out = df_val[output_cols]
# df_out = np.expm1(df_out)  # Inverse of log1p transformation
out_gt = df_out.values

# 3D Plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(out_gt[:,0], out_gt[:,1], out_gt[:,2], c='blue', label='Ground Truth Position', alpha=0.5)
ax.scatter3D(out[:,0], out[:,1], out[:,2], c='red', label='Predicted Position', alpha=0.5)
plt.show()

# 2D Plot


# plt.figure(figsize=(8, 6))
# plt.scatter(range(len(out_gt)), out_gt, alpha=0.5, label="Ground Truth CQI", color='blue')
# plt.scatter(range(len(out)), out, alpha=0.5, label="Predicted CQI", color='red')
# plt.xlabel("Sample Index")
# plt.ylabel("CQI Value")
# plt.title("Predicted and Ground Truth CQI")
# plt.legend()
# plt.grid(True)
# plt.show()

diff = out - out_gt
print("Mean abs difference:", np.abs(diff).mean())
print("Standard deviation of difference:", diff.std())