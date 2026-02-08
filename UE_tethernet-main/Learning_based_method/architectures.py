import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_Regressor(nn.Module):
    def __init__(self, input_size, hidden_size=[64, 32], dropout=0.5, output_size=1):
        super(MLP_Regressor, self).__init__()
        layers = []
        in_features = input_size
        for h in hidden_size:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_features = h
        layers.append(nn.Linear(in_features, output_size))
        # layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        # x = torch.sigmoid(x)
        return x
    

class CNN_MLP_Fusion(nn.Module):
    def __init__(self, input_channels=3, img_size=(64, 64), cnn_channel_nums=[16, 32], vec_size=7, mlp_hidden_size=[32], fusion_mlp_hidden_size=[64, 32], dropout=0.5, output_size=1):
        super(CNN_MLP_Fusion, self).__init__()

        # 1. CNN Branch for image
        layers = []
        in_channels = input_channels
        conv_layer_num = len(cnn_channel_nums)

        for out_channels in cnn_channel_nums:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        conv_out_h, conv_out_w = img_size[0] // (2**conv_layer_num), img_size[1] // (2**conv_layer_num)
        cnn_out_features = cnn_channel_nums[-1] * conv_out_h * conv_out_w

        # 2. MLP Branch for 7-D vector
        layers = []
        in_features = vec_size
        for h in mlp_hidden_size:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = h
        layers.append(nn.Linear(in_features, mlp_hidden_size[-1]))
        self.vec_mlp = nn.Sequential(*layers)
        vec_mlp_out_features = mlp_hidden_size[-1]
        # self.vec_mlp = nn.Sequential(
        #     nn.Linear(vec_size, mlp_hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(dropout)
        # )

        # 3. Fusion MLP
        fusion_input_size = cnn_out_features + vec_mlp_out_features
        layers = []
        in_features = fusion_input_size
        for h in fusion_mlp_hidden_size:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = h
        layers.append(nn.Linear(in_features, output_size))
        self.fusion_mlp = nn.Sequential(*layers)

    def forward(self, img, vec):
        # img: [B, C, H, W], vec: [B, 7]
        img_feat = self.cnn(img)
        img_feat = img_feat.view(img_feat.size(0), -1)  # flatten

        vec_feat = self.vec_mlp(vec)

        combined = torch.cat([img_feat, vec_feat], dim=1)
        output = self.fusion_mlp(combined)
        output = torch.tanh(output)
        return output