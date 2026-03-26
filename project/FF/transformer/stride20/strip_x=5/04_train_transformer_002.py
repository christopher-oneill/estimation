import os
import h5py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import math

# --- 1. CONFIGURATION ---
MODEL_NUM = 5
output_dir = "/media/chris-remote/Projects/ONeill/estimation/output/kevin_2cylinder"
model_dir = os.path.join(output_dir, f"transformer_{MODEL_NUM}_FF") # New folder for sanity check
os.makedirs(model_dir, exist_ok=True)

data_path = os.path.join(output_dir, "lstm_ready_data_FF_nearwake.h5")
ckpt_path = os.path.join(model_dir, "latest_checkpoint.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. DATASET (Modified to accept slice counts) ---
class SparseLabelDataset(Dataset):
    def __init__(self, ss_data, ff_data, seq_len=100, stride=20):
        self.ss_data = torch.FloatTensor(ss_data)
        self.ff_data = torch.FloatTensor(ff_data)
        self.half_len = seq_len // 2
        self.stride = stride
        full_range = np.arange(self.half_len, len(ff_data) - self.half_len)
        self.valid_indices = full_range[full_range % self.stride == 0]

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        x = self.ss_data[t - self.half_len : t + self.half_len, :]
        y = self.ff_data[t, :]
        return x, y

# --- 3. ARCHITECTURE (Same as before) ---
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
    def __init__(self, input_dim, output_dim, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.3):
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
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, output_dim)
        )
    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.out_layer(torch.mean(x, dim=1))


# --- 4. TRAINING ENGINE ---
def train():

    # Define hyperparameters here so they can be saved in the checkpoint
    config = {
        'input_dim': 40, 
        'output_dim': 20, 
        'd_model': 128,        # Increased for higher mode resolution
        'nhead': 4,            # More heads to track different vortex structures
        'num_layers': 8,       # Deeper for chaotic flip-flop transitions
        'dim_feedforward': 128, 
        'dropout': 0.2,        # Reduced since we have a larger task
        'seq_len': 200
    }

    model = SupersamplingTransformer(
        input_dim=config['input_dim'], 
        output_dim=config['output_dim'], 
        d_model=config['d_model'], 
        nhead=config['nhead'], 
        num_layers=config['num_layers'], 
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(device)




    print(f"Loading Flip-Flop Dataset...")
    with h5py.File(data_path, 'r') as f:
        a_ff_train = f['train/a_ff'][:, :config['output_dim']]
        a_ss_train = f['train/a_ss'][:, :config['input_dim']]
        a_ff_test = f['test/a_ff'][:, :config['output_dim']]
        a_ss_test = f['test/a_ss'][:, :config['input_dim']]

    scaler_ss = StandardScaler().fit(a_ss_train)
    scaler_ff = StandardScaler().fit(a_ff_train)
    
    train_ds = SparseLabelDataset(scaler_ss.transform(a_ss_train), 
                                  scaler_ff.transform(a_ff_train), 100, 20)
    val_ds = SparseLabelDataset(scaler_ss.transform(a_ss_test), 
                                scaler_ff.transform(a_ff_test), 100, 20)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)


    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=150, factor=0.5)
    criterion = nn.MSELoss()

    # --- SAFE RESUME LOGIC ---
    start_epoch = 0
    if os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        
        # Safely load optimizer
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("--> Optimizer state restored.")
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"--> Starting at Epoch {start_epoch}")

    print(f"Sanity Check: 20 SS Modes -> 10 FF Modes")
    for epoch in range(start_epoch, start_epoch+4000):
        # --- NOISE SCHEDULER ---
        initial_noise = 0.02
        final_noise = 0.001  # A tiny bit of "regularization floor"
        anneal_epochs = 4000   # How long to take to reach the floor

        # Cosine decay calculation
        if epoch < anneal_epochs:
            noise_std = final_noise + 0.5 * (initial_noise - final_noise) * \
                        (1 + math.cos(math.pi * epoch / anneal_epochs))
        else:
            noise_std = final_noise

        model.train()
        t_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            bx = bx + torch.randn_like(bx) * noise_std
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        
        model.eval()
        v_loss = 0
        t_loss_clean = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                v_loss += criterion(model(vx.to(device)), vy.to(device)).item()
            for bx, by in train_loader:
                t_loss_clean += criterion(model(bx.to(device)), by.to(device)).item()
        
        avg_train = t_loss / len(train_loader)
        avg_train_clean = t_loss_clean / len(train_loader)
        avg_val = v_loss / len(val_loader)
        scheduler.step(avg_val)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train: {avg_train:.2e} | Train (Clean): {avg_train_clean:.2e} | Val: {avg_val:.2e} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(), # Properly saved dictionary
            'scaler_ss': scaler_ss,
            'scaler_ff': scaler_ff,
            'config': config,  # Using the local config dictionary
            'model_type': 'Transformer'
        }, ckpt_path)


if __name__ == "__main__":
    train()