import torch
import torch.nn as nn
from mamba_ssm import Mamba

class FFNBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)
    
class MambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mamba = Mamba(d_model=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FFNBlock(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        residual = x
        x = self.mamba(x)
        x = x + residual
        x = self.norm1(x)

        residual = x
        x = x.permute(0, 2, 1)          
        x = self.ffn(x)
        x = x.permute(0, 2, 1)          
        x = x + residual

        x = self.norm2(x)
        return x.permute(0, 2, 1)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class ResidualTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.tcnet = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.tcnet2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.BatchNorm1d(out_channels)
        )
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1)
            )

    def forward(self, x):
        out = self.tcnet(x)
        out = self.tcnet2(out)
        return self.relu(out + self.shortcut(x))


class AFMamba(nn.Module):
    def __init__(self, input_channels=1, out_channels = 32, num_classes=2):
        super().__init__()
        self.tcn = nn.Sequential(
            ResidualTCNBlock(input_channels, out_channels, dilation=1),
            ResidualTCNBlock(out_channels, out_channels, dilation=2),
            ResidualTCNBlock(out_channels, out_channels, dilation=4),
        )
        self.mamba = nn.Sequential(
            MambaBlock(d_model=out_channels),
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
        x = self.mamba(x)           

        avg = self.global_avg_pool(x)  
        max = self.global_max_pool(x)  
        x = torch.cat([avg, max], dim=1).squeeze(-1)  

        return self.fc(x)
