import pandas as pd
import numpy as np
import pickle

num_sample = 5
num_sample_per_scenario = 1

est_pos_cols = [f"Est_PX{i}" for i in range(1,num_sample_per_scenario+1)] + \
            [f"Est_PY{i}" for i in range(1,num_sample_per_scenario+1)] + \
            [f"Est_PZ{i}" for i in range(1,num_sample_per_scenario+1)]
est_ori_cols = [f"Est_OW{i}" for i in range(1,num_sample_per_scenario+1)] + \
            [f"Est_OX{i}" for i in range(1,num_sample_per_scenario+1)] + \
            [f"Est_OY{i}" for i in range(1,num_sample_per_scenario+1)] + \
            [f"Est_OZ{i}" for i in range(1,num_sample_per_scenario+1)]
input_cols = est_pos_cols + est_ori_cols

output_cols = ["Gt_PX", "Gt_PY", "Gt_PZ"]

add_cols = ["Error", "Confidence"]

df_new = pd.read_csv(".\datasets\MLP_input.csv")
df_new = df_new[input_cols + output_cols]

df_add = pd.read_pickle("inputdata_df_sruthi.pkl")
df_add = df_add[input_cols + output_cols]
df_add = df_add.iloc[-1000:, :] # Unseen data from old dataset

df_fin = pd.concat([df_new, df_add], ignore_index=True)
df_fin = df_fin.astype('float32')

df_fin["Error"] = np.linalg.norm(
    df_fin[["Gt_PX", "Gt_PY", "Gt_PZ"]].values - df_fin[["Est_PX1", "Est_PY1", "Est_PZ1"]].values,
    axis=1
)

df_fin["log_err"] = np.log1p(df_fin["Error"])
df_fin.to_pickle("df_log_err_train_6k.pkl")


with open(f"./datasets/5k_images_data.pkl", "rb") as f:
    img_arrs_new = pickle.load(f)

with open(f"./datasets/image_arrs_done_read.pkl", "rb") as f:
    img_arrs_add = pickle.load(f)
    img_arrs_add = img_arrs_add[-1000:]  # Unseen data from old dataset

img_arrs_fin = img_arrs_new + img_arrs_add

with open("6k_with_images.pkl", "wb") as f:
    pickle.dump(img_arrs_fin, f)
    
# df = pd.DataFrame(
#     np.random.randn(num_sample, len(input_cols + output_cols)),
#     columns=input_cols + output_cols
# )

# with open("example_df.pkl", "wb") as f:
#     pd.to_pickle(df, f)