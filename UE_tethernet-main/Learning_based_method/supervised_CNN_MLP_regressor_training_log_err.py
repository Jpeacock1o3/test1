"""
Author: Feng Liu
Date: August 2025

This script trains a CNN combined with a Multi-Layer Perceptron (MLP) regressor using PyTorch with detailed logs
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
from architectures import MLP_Regressor, CNN_MLP_Fusion
import pickle
import joblib
import time
from customized_funcs import get_sorted_filenames, percept_confidence_MinMaxScaler
import cv2

plot_flag = 1
if plot_flag == 1:
    from matplotlib import pyplot as plt

start = time.time()

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

save_flag = 1           # set to 1 to activate saving
if save_flag == 1:
    input("Saving?")
    print("Data will be saved")
    pass

# write_temp_txt_flag = 0 # set to 1 to activate saving

# set seed
rand_seed = 1
torch.manual_seed(rand_seed)
generator = torch.Generator()
generator.manual_seed(rand_seed)

# customized log info
############# User inputs ##############
run_id = 1
img_dim = [320,180,3]
num_sample_per_scenario = 1
validation_size_preset = 500
lr_set = 0.01
num_epochs = 100
cnn_channel_nums_set = [16,32,64]
mlp_hidden_size_set = [32]
fusion_mlp_hidden_size = [128, 64, 32]


# dataset_name = "inputdata_df_sruthi.pkl"
dataset_name = "df_log_err_train_6k.pkl"

img_folder_path = "./datasets/CNN_MLP_img/"
img_done_read_file = "image_arrs_done_read_6k.pkl"
model_type = "MLP_Regressor, mse loss"
additional_info = "addintional log info here"

# log file names
model_folder = f"MLP_training/train_log_{run_id}"
model_savename = f"{model_folder}/ICRA2026_MLP_model{run_id}.pth"
model_dict_savename = f"{model_folder}/ICRA2026_MLP_dict{run_id}.pth"
loss_train_track_name = f"{model_folder}/ICRA2026_MLP_train_loss{run_id}.npy"
loss_val_track_name = f"{model_folder}/ICRA2026_MLP_val_loss{run_id}.npy"
lr_track_name = f"{model_folder}/ICRA2026_MLP_LR{run_id}.npy"
input_scaler_name = f"{model_folder}/ICRA2026_MLP_input_scaler{run_id}.pkl"
output_scaler_name = f"{model_folder}/ICRA2026_MLP_output_scaler{run_id}.pkl"


# prepare the dataset
# est_pos_cols = ["Est_PX", "Est_PY", "Est_PZ"]
# est_ori_cols = ["Est_OX", "Est_OY", "Est_OZ"]
# input_cols = est_pos_cols + est_ori_cols

est_pos_cols = [f"Est_PX{i}" for i in range(1,num_sample_per_scenario+1)] + \
            [f"Est_PY{i}" for i in range(1,num_sample_per_scenario+1)] + \
            [f"Est_PZ{i}" for i in range(1,num_sample_per_scenario+1)]
est_ori_cols = [f"Est_OW{i}" for i in range(1,num_sample_per_scenario+1)] + \
            [f"Est_OX{i}" for i in range(1,num_sample_per_scenario+1)] + \
            [f"Est_OY{i}" for i in range(1,num_sample_per_scenario+1)] + \
            [f"Est_OZ{i}" for i in range(1,num_sample_per_scenario+1)]

# img_filenames_sorted = get_sorted_filenames(img_folder_path)

# img_arrs = []
# for img_filename in img_filenames_sorted:
#     img = cv2.imread(img_filename)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_rgb_resized = cv2.resize(img_rgb, (img_dim[1], img_dim[0]))  # Resize to (width, height)
#     img_rgb_resized_norm = img_rgb_resized.astype(np.float32) / 255.0
#     img_arrs.append(img_rgb_resized_norm)

# with open(f"./datasets/image_arrs_done_read.pkl", "rb") as f:
with open(f"./datasets/{img_done_read_file}", "rb") as f:
    img_arrs = pickle.load(f)

if all((img >= 0).all() and (img <= 1).all() for img in img_arrs):
    print("All values in img_arrs are between 0 and 1 (inclusive).")
else:
    print("Warning: Some values in img_arrs are outside [0, 1].")

# for img in img_arrs:
for i in range(len(img_arrs)):
    img = img_arrs[i]
    img = cv2.resize(img, (img_dim[1], img_dim[0]))  # Resize to (width, height)
    img_arrs[i] = img.astype(np.float32)

df = pd.read_pickle(dataset_name)
# df = df.iloc[:5, :] # DEBUG
df = df.astype('float32')

# check if the number of image filenames matches the number of dataframe rows
# if len(img_filenames_sorted) != len(df):
#     raise ValueError(f"Number of image filenames ({len(img_filenames_sorted)}) does not match number of dataframe rows ({len(df)}).")

# Add images to dataframe as a new column
df["image"] = img_arrs

est_vec = est_pos_cols + est_ori_cols
input_cols = est_vec + ["image"]

# output_cols = ["Gt_PX", "Gt_PY", "Gt_PZ"]
# output_cols = ["Confidence"]   # 1/(Gt_Pos - Ursonet_Pos), scalar
output_cols = ["log_err"] # log of position error, scalar 

# Calculate Euclidean distance between ground truth and estimated position
# df.loc[:, output_cols] = 1 / np.linalg.norm(
#     df[["Gt_PX", "Gt_PY", "Gt_PZ"]].values - df[["Est_PX1", "Est_PY1", "Est_PZ1"]].values,
#     axis=1
# )
# remove outliers
# idx_max = []
# df = df.drop(idx_max).reset_index(drop=True)

df0 = df.copy()

# shuffle and get success and failed data evenly
df = df.sample(frac=1, random_state=rand_seed).reset_index(drop=True)
# df_success = df[df["Successflag"]==1]
# df_fail = df[df["Successflag"]==0].iloc[0:len(df_success)*3, :]
# df = pd.concat([df_success, df_fail], ignore_index=True)
df = df.sample(frac=1, random_state=rand_seed).reset_index(drop=True)


df_val = df.iloc[-validation_size_preset :, :].copy()
df_train = df.iloc[:-validation_size_preset, :].copy()


batch_size_set = 100
train_size = len(df_train)
val_size = len(df_val)


# hd_set = 256
# hd_set = 128


if not os.path.exists(model_folder):
    # Create the directory
    os.makedirs(model_folder)
    print(f"{model_folder} created successfully!")

# with open(f"./{model_folder}/df_train_and_df_val.pkl", "wb") as f:
#     pickle.dump((df_train, df_val), f)

print("RUNNING id: ", run_id)

df_train.loc[:, est_vec]  = percept_confidence_MinMaxScaler(df_train.loc[:, est_vec].values, 
                                                            data_type="input", 
                                                            operation="normalize")
df_val.loc[:, est_vec]  = percept_confidence_MinMaxScaler(df_val.loc[:, est_vec].values, 
                                                            data_type="input", 
                                                            operation="normalize")

df_train.loc[:, output_cols]  = percept_confidence_MinMaxScaler(df_train.loc[:, output_cols].values, 
                                                            data_type="output",
                                                            operation="normalize")
df_val.loc[:, output_cols]  = percept_confidence_MinMaxScaler(df_val.loc[:, output_cols].values, 
                                                            data_type="output", 
                                                            operation="normalize")

if (df_train.loc[:, est_vec+output_cols].values >= 0).all() and (df_train.loc[:, est_vec+output_cols].values <= 1).all() and \
    (df_val.loc[:, est_vec+output_cols].values >= 0).all() and (df_val.loc[:, est_vec+output_cols].values <= 1).all():
        print("All df_train and df_val values are between 0 and 1 (inclusive).")
else:
        print("Warning: Some values in df_train or df_val are outside [0, 1].")

# train_tensor = torch.tensor(df_train[input_cols+output_cols].values)
# val_tensor = torch.tensor(df_val[input_cols+output_cols].values)

input_len = len(input_cols)
output_len = len(output_cols)

train_input_vec = torch.tensor(df_train[est_vec].values, dtype=torch.float32)
train_input_img = torch.stack([torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) for img in df_train['image']])

val_input_vec = torch.tensor(df_val[est_vec].values, dtype=torch.float32)
val_input_img = torch.stack([torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) for img in df_val['image']])


# train_input_dataset = torch.tensor(df_train[input_cols].values, dtype=torch.float32)
# val_input_dataset = torch.tensor(df_val[input_cols].values, dtype=torch.float32)

if len(output_cols) > 1:
    train_output_dataset = torch.tensor(df_train[output_cols].values, dtype=torch.float32)
    val_output_dataset = torch.tensor(df_val[output_cols].values, dtype=torch.float32)
else:
    train_output_dataset = torch.tensor(df_train[output_cols].values.ravel(), dtype=torch.float32).unsqueeze(1)
    val_output_dataset = torch.tensor(df_val[output_cols].values.ravel(), dtype=torch.float32).unsqueeze(1)

train_dataset = torch.utils.data.TensorDataset(train_input_vec, train_input_img, train_output_dataset)
val_dataset = torch.utils.data.TensorDataset(val_input_vec, val_input_img, val_output_dataset)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = batch_size_set,
    shuffle = True,
    generator=generator
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size = batch_size_set,
    shuffle = True,
    generator=generator
)

# model = CNNBinaryClassifier(input_size=len(input_cols), num_filters=flt_set, kernel_size=kn_set, hidden_size=hd_set).to(device)
# model = MLPBinaryClassifier(input_size=len(input_cols), hidden_size=hd_set, output_size=len(output_cols)).to(device)
# model = MLP_Regressor(input_size=len(input_cols), hidden_size=hd_set, output_size=len(output_cols)).to(device)
model = CNN_MLP_Fusion(input_channels=img_dim[2], 
                       img_size=[img_dim[0], img_dim[1]], 
                       cnn_channel_nums=cnn_channel_nums_set, 
                       vec_size=len(est_vec), 
                       mlp_hidden_size=mlp_hidden_size_set, 
                       fusion_mlp_hidden_size=fusion_mlp_hidden_size, 
                       output_size=len(output_cols)).to(device)

criterion = nn.MSELoss().to(device)
# criterion = nn.SmoothL1Loss().to(device)
# criterion = nn.BCELoss().to(device)
# pos_weight = torch.tensor([0.8]).to(device)
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
# criterion = FocalLoss(alpha=0.25, gamma=8.0)

# optimizer = AdamW(model.parameters(), lr=lr_set)
optimizer = Adam(model.parameters(), lr=lr_set)
# optimizer = Adam(model.parameters(), lr=lr_set, weight_decay=1e-4)
# optimizer = Adam(model.parameters(), lr=lr_set, weight_decay=1e-5)
# optimizer = Adam(model.parameters(), lr=lr_set)

from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, threshold=1e-5, threshold_mode='abs')


loss_train_track = np.zeros(num_epochs)
loss_val_track = np.zeros(num_epochs)
# loss_val_denorm_track = np.zeros(num_epochs)
lr_track = np.zeros(num_epochs)

# if write_temp_txt_flag==1:
#     f = open(f"train_log_temp{run_id}.txt", "w")
#     f.write("CCR training log \n")
#     f.close()


best_val_loss = 1000 # dummy initial large value
model_savename_best = f"{model_folder}/ICRA2026_MLP_round{run_id}_best_at_initial.pth"
model_dict_savename_best = f"{model_folder}/ICRA2026_MLP_dict{run_id}_best_at_initial.pth"

prev_lr = optimizer.param_groups[0]['lr']
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0
    for input_vec_feat, img_feat, output_feat in train_dataloader:
        input_vec_feat = input_vec_feat.to(device)
        output_feat = output_feat.to(device)
        img_feat = img_feat.to(device)
        optimizer.zero_grad()
        # outputs = model(input_vec_feat, output_feat)
        outputs = model(img_feat, input_vec_feat)
        loss = criterion(outputs, output_feat)
        # add weight to false positive
        # weights = torch.ones_like(output_feat)
        # weights[output_feat == 0] = 10
        # weighted_loss = (loss * weights).mean()
        # weighted_loss.backward()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    val_denorm_loss = 0
    with torch.no_grad():  # No gradients needed for validation
        for input_vec_feat, img_feat, output_feat in val_dataloader:
            input_vec_feat = input_vec_feat.to(device)
            output_feat = output_feat.to(device)
            img_feat = img_feat.to(device)
            # outputs = model(input_vec_feat, output_feat)
            outputs = model(img_feat, input_vec_feat)
            loss = criterion(outputs, output_feat)
            val_loss += loss.item()
            # val_denorm_loss += np.sqrt(loss_denorm.item())

    loss_train_track[epoch] = epoch_loss/len(train_dataloader)
    loss_val_track[epoch] = val_loss/len(val_dataloader)

    current_lr = optimizer.param_groups[0]['lr']

    ########## For Scheudler ##########
    '''Comment out if not using scheduler or saving'''
    scheduler.step(val_loss)
    # Detect learning rate drop
    if current_lr < prev_lr:
        print(f"LR dropped from {prev_lr:.6f} to {current_lr:.6f}, reloading best model from: {model_dict_savename_best_roll}")
        model.load_state_dict(torch.load(model_dict_savename_best_roll))
        model.to(device)  # ensure model is on correct device
    prev_lr = current_lr  # update tracker
    ########## For Scheudler END ##########

    lr_track[epoch] = current_lr
    temp_end = time.time()
    # print and save log
    train_epoch_loss = epoch_loss/len(train_dataloader)
    val_epoch_loss = val_loss/len(val_dataloader)
    log_text = f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_epoch_loss:.6f}, Val Loss: {val_epoch_loss:.6f}, LR: {current_lr:.4f}, Elapsed Time: {temp_end - start:.4f} \n'
    print(log_text)
    if (save_flag==1) and ((epoch+1) % 50 == 0):
        model_dict_savename = f"{model_folder}/ICRA2026_MLP_dict{run_id}_ep{epoch+1}.pth"
        torch.save(model.state_dict(), model_dict_savename)
    if (save_flag==1) and (val_epoch_loss < best_val_loss):
        # remove old best model to save space
        if os.path.exists(model_savename_best):
            os.remove(model_savename_best)
            os.remove(model_dict_savename_best)
        model_savename_best = f"{model_folder}/ICRA2026_MLP_round{run_id}_best_at{epoch+1}.pth"
        model_dict_savename_best = f"{model_folder}/ICRA2026_MLP_dict{run_id}_best_at{epoch+1}.pth"
        model_dict_savename_best_roll = f"{model_folder}/ICRA2026_MLP_dict{run_id}_best_roll.pth"
        torch.save(model, model_savename_best)
        torch.save(model.state_dict(), model_dict_savename_best)
        torch.save(model.state_dict(), model_dict_savename_best_roll)
        np.save(loss_train_track_name, loss_train_track)
        np.save(loss_val_track_name, loss_val_track)
        best_val_loss = val_epoch_loss


end = time.time()
print("Total elapsed time: ", end - start)


# conf_matrix = confusion_matrix(output_feat.cpu().numpy().ravel(), torch.round(outputs.cpu()).numpy().ravel())
# # f1 = f1_score(output_feat.cpu().numpy().ravel(), torch.round(outputs.cpu()).numpy().ravel(), labels=[0,1])
# print(conf_matrix)
# print(f1)

# accuracy = accuracy_score(output_feat.cpu().numpy().ravel(), torch.round(outputs.cpu()).numpy().ravel())
# print("Accuracy on validation set:", accuracy)

if save_flag==1:
    # if not os.path.exists(model_folder):
    #     # Create the directory
    #     os.makedirs(model_folder)
    #     print(f"{model_folder} created successfully!")

    # log_text = f""" {model_type}
    # dataset_num_success = {len(df_success)}
    # dataset_num_fail = {len(df_fail)}
    # training_dataset_size = {train_size}
    # validation_dataset_size = {val_size}
    # batch_size_set = {batch_size_set}
    # lr {lr_set}
    # num_epochs = {num_epochs}
    # hidden_size = {hd_set}
    # filter_num = {flt_set}
    # kernal_size = {kn_set}
    # final_val_loss = {best_val_loss}
    # accuracy = {accuracy_best}
    # conf_matrix:
    # {conf_matrix_best}
    # note: {additional_info}
    # """

    log_text = f""" 
    ######################################
    {model_type}
    run_id = {run_id}
    rand_id = {rand_seed}
    output_cols = {output_cols}
    loss_type = {criterion}
    training_dataset_size = {train_size}
    validation_dataset_size = {val_size}
    batch_size_set = {batch_size_set}
    img_dim = {img_dim}
    num_sample_per_scenario = {num_sample_per_scenario}
    lr {lr_set}
    num_epochs = {num_epochs}
    cnn_channel_nums = {cnn_channel_nums_set}
    mlp_hidden_size = {mlp_hidden_size_set}
    fusion_hidden_size = {fusion_mlp_hidden_size}
    final_best_val_loss = {best_val_loss}
    note: {additional_info}
    """

    print(log_text)
    with open(f"{model_folder}/log_Info.txt", "a") as f:
        f.write(log_text)

    torch.save(model, model_savename)
    torch.save(model.state_dict(), model_dict_savename)
    np.save(loss_train_track_name, loss_train_track)
    np.save(loss_val_track_name, loss_val_track)
    # np.save(loss_val_denorm_track_name, loss_val_denorm_track)
    np.save(lr_track_name, lr_track)
    # joblib.dump(input_scaler, input_scaler_name)
    # joblib.dump(output_scaler, output_scaler_name)

if plot_flag == 1:
    # plot loss history
    ymin = 0
    ymax = 1.0
    plotx = np.arange(0,num_epochs)
    plt.plot(plotx, loss_train_track)
    plt.title("Training Loss Convergence History")
    plt.xlabel("# of Epochs")
    plt.ylabel("Loss")
    # plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.show()

    plt.plot(plotx, loss_val_track)
    plt.title("Validation Loss Convergence History")
    plt.xlabel("# of Epochs")
    plt.ylabel("Loss")
    plt.ylim([ymin, ymax])
    plt.show()


