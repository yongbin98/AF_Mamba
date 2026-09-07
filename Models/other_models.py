import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class CNNBiLSTM(nn.Module):
    def __init__(self):
        super(CNNBiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, bidirectional=True, batch_first=True)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256, 32)  
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  

        x = x.permute(0, 2, 1)  
        x, _ = self.lstm(x)    
        x = x.permute(0, 2, 1) 

        x = self.global_max_pool(x).squeeze(-1) 
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

class CNNBiGRU(nn.Module):
    def __init__(self):
        super(CNNBiGRU, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3, stride=1) 
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, stride=1) 
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) 

        self.bigru = nn.GRU(input_size=100, hidden_size=100, bidirectional=True, batch_first=True) 

        self.fc = nn.Linear(200, 2) 

    def forward(self, x):

        x = F.relu(self.conv1(x))   
        x = F.relu(self.conv2(x))  
        x = self.pool(x)          

        x = x.permute(0, 2, 1)
        _, h_n = self.bigru(x)

        h_n = torch.cat((h_n[0], h_n[1]), dim=1)

        out = self.fc(h_n)

        return out

class MambaLayer(nn.Module):
    def __init__(self, d_model, d_ff=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff * d_model),
            nn.SiLU(),
            nn.Linear(d_ff * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.mamba(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x

class VanillaMamba(nn.Module):
    def __init__(self, seq_len=3000, d_model=32, num_layers=4, d_ff=4):
        super().__init__()

        self.embed = nn.Conv1d(1, d_model, kernel_size=3, padding=1)

        self.layers = nn.ModuleList([
            MambaLayer(d_model=d_model, d_ff=d_ff)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.embed(x)                
        x = x.transpose(1, 2)            

        for layer in self.layers:
            x = layer(x)                 

        x = x.mean(dim=1)               

        return self.fc(x)