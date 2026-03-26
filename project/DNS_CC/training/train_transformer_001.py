import os
import platform
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from tqdm import tqdm
import math

# --- 1. Versioning & Pathing ---
MODEL_NUMBER = 3  # Transformer Version
load_dotenv()
current_os = platform.system()
base_dir = os.getenv("BASE_DIR_WIN") if current_os == "Windows" else os.getenv("BASE_DIR_LIN")

data_path = os.path.join(base_dir, "output/DNS_CC_Re150_Mazi/lstm_ready_data.h5")
model_dir = os.path.join(base_dir, f"output/DNS_CC_Re150_Mazi/Model_T_{MODEL_NUMBER}")
os.makedirs(model_dir, exist_ok=True)

ckpt_latest = os.path.join(model_dir, "latest_checkpoint.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. Dataset: Centered Window ---
class CenteredSupersamplingDataset(Dataset):
    def __init__(self, ss_data, ff_data, seq_len=100, stride=1):
        self.ss_data = torch.FloatTensor(ss_data)
        self.ff_data = torch.FloatTensor(ff_data)
        self.half_len = seq_len // 2
        # Valid indices ensure window fits on both sides
        self.valid_indices = np.arange(self.half_len, len(ff_data) - self.half_len, stride)

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        x = self.ss_data[t - self.half_len : t + self.half_len, :]
        y = self.ff_data[t, :]
        return x, y

# --- 3. Architecture: Transformer ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class SupersamplingTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.out_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Global average pooling over the sequence dimension
        return self.out_layer(torch.mean(x, dim=1))

# --- 4. Training Pipeline ---
def train_model():
    # --- HYPERPARAMETERS ---
    config = {
        'input_dim': 0,     # Will be set dynamically
        'output_dim': 0,    # Will be set dynamically
        'seq_len': 100,
        'stride': 1,        # Using full temporal resolution
        'd_model': 128,
        'nhead': 4,
        'num_layers': 4,
        'dim_feedforward': 256,
        'dropout': 0.1
    }

    with h5py.File(data_path, 'r') as f:
        a_ff_train, a_ss_train = f['train/a_ff'][:], f['train/a_ss'][:]
        a_ff_test, a_ss_test = f['test/a_ff'][:], f['test/a_ss'][:]

    config['input_dim'] = a_ss_train.shape[1]
    config['output_dim'] = a_ff_train.shape[1]

    scaler_ss = StandardScaler().fit(a_ss_train)
    scaler_ff = StandardScaler().fit(a_ff_train)
    
    train_ds = CenteredSupersamplingDataset(scaler_ss.transform(a_ss_train), 
                                            scaler_ff.transform(a_ff_train), 
                                            config['seq_len'], config['stride'])
    val_ds = CenteredSupersamplingDataset(scaler_ss.transform(a_ss_test), 
                                          scaler_ff.transform(a_ff_test), 
                                          config['seq_len'], config['stride'])
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = SupersamplingTransformer(
        input_dim=config['input_dim'], 
        output_dim=config['output_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=25, factor=0.5)
    criterion = nn.MSELoss()

    # --- Resume Logic ---
    start_epoch = 0
    history = {'train': [], 'val': []}
    if os.path.exists(ckpt_latest):
        print(f"Resuming Transformer Model {MODEL_NUMBER}...")
        checkpoint = torch.load(ckpt_latest, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('history', history)

    # --- Training Loop ---
    for epoch in range(start_epoch, 2000):
        model.train()
        t_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            # Optional: Add small noise during training for robustness
            bx = bx + torch.randn_like(bx) * 0.005 
            
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        
        avg_train = t_loss / len(train_loader)
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_loss += criterion(model(vx), vy).item()
        
        avg_val = v_loss / len(val_loader)
        scheduler.step(avg_val)
        
        history['train'].append(avg_train)
        history['val'].append(avg_val)

        if (epoch + 1) % 10 == 0:
            print(f"Mod{MODEL_NUMBER} Ep{epoch+1:03d} | T: {avg_train:.2e} | V: {avg_val:.2e} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # --- Save Checkpoints ---
        save_dict = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'history': history,
            'scaler_ss': scaler_ss,
            'scaler_ff': scaler_ff,
            'config': config
        }
        torch.save(save_dict, ckpt_latest)
        if (epoch + 1) % 500 == 0:
            torch.save(save_dict, os.path.join(model_dir, f"ckpt_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    train_model()