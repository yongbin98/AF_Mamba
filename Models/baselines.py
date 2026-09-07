import torch
import torch.nn as nn
from .AF_mamba import ResidualTCNBlock

class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
            dropout=0.1
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.attn_block(x)
        return x.permute(0, 2, 1)

class TCN_LSTMModel(nn.Module):
    def __init__(self, input_channels=1, out_channels = 32, num_classes=2):
        super().__init__()
        self.tcn = nn.Sequential(
            ResidualTCNBlock(input_channels, out_channels, dilation=1),
            ResidualTCNBlock(out_channels, out_channels, dilation=2),
            ResidualTCNBlock(out_channels, out_channels, dilation=4),
        )

        self.fc = nn.Sequential(
            nn.Linear(out_channels*2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        x = self.tcn(x)

        x = x.permute(0, 2, 1) 
        x, _ = self.lstm(x)   
        x = x[:, -1, :]         

        return self.fc(x)

class TCN_OnlyModel(nn.Module):
    def __init__(self, input_channels=1, out_channels = 32, num_classes=2):
        super().__init__()
        self.tcn = nn.Sequential(
            ResidualTCNBlock(input_channels, out_channels, dilation=1),
            ResidualTCNBlock(out_channels, out_channels, dilation=2),
            ResidualTCNBlock(out_channels, out_channels, dilation=4),
        )

        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels*2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.tcn(x)

        avg = self.global_avg_pool(x) 
        max = self.global_max_pool(x) 
        x = torch.cat([avg, max], dim=1).squeeze(-1)  

        return self.fc(x)

class TCN_TransformerModel(nn.Module):
    def __init__(self, input_channels=1, out_channels = 32, num_classes=2):
        super().__init__()
        self.tcn = nn.Sequential(
            ResidualTCNBlock(input_channels, out_channels, dilation=1),
            ResidualTCNBlock(out_channels, out_channels, dilation=2),
            ResidualTCNBlock(out_channels, out_channels, dilation=4),
        )
        self.tf = nn.Sequential(
            TransformerBlock(d_model=out_channels),
            nn.Dropout(0.2)
        )

        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels*2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.tcn(x)
        x = self.tf(x)             

        avg = self.global_avg_pool(x) 
        max = self.global_max_pool(x) 
        x = torch.cat([avg, max], dim=1).squeeze(-1)  

        return self.fc(x)
