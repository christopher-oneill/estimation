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
MODEL_NUMBER = 3 # Unique ID for this FF run
base_dir = "/media/chris-remote/Projects/ONeill/estimation/output/kevin_2cylinder/FF/LSTM/strip10_stride1"
master_pod_path = os.path.join(base_dir, "ff_pod_master.h5")
model_dir = os.path.join(base_dir, f"Model_BiLSTM_{MODEL_NUMBER}")
os.makedirs(model_dir, exist_ok=True)

ckpt_latest = os.path.join(model_dir, "latest_checkpoint.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters for the chaotic Flip-Flop regime
config = {
    'n_modes_ss': 50,   # Sensor input (Far-wake x=10 needs higher mode count)
    'n_modes_ff': 10,   # Field output (Targeting ~2-3% velocity error)
    'seq_len': 200,      # Longer window to capture the slow 'flip' cycle
    'hidden_dim': 1024,  
    'num_layers': 2,
    'batch_size': 350,
    'lr': 1e-3,          # Slightly lower LR for deeper stack stability
    'initial_noise': 0.00,
    'final_noise': 0.0
}

# --- 2. DATASET ---
class MasterPODDataset(Dataset):
    def __init__(self, ss_coeffs, ff_coeffs, seq_len=100):
        self.ss_data = torch.FloatTensor(ss_coeffs)
        self.ff_data = torch.FloatTensor(ff_coeffs)
        self.half_len = seq_len // 2
        self.valid_indices = np.arange(self.half_len, len(ff_coeffs) - self.half_len)

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        x = self.ss_data[t - self.half_len : t + self.half_len, :]
        y = self.ff_data[t, :]
        return x, y

# --- 3. ARCHITECTURE ---
class SupersamplingBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, 
                            batch_first=True, dropout=0.1, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(), 
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 4. TRAINING ENGINE ---
def train():
    print(f"Loading FF Master POD: {config['n_modes_ss']} SS -> {config['n_modes_ff']} FF")
    
    with h5py.File(master_pod_path, 'r') as f:
        a_ff_tr = f['train/a_ff'][:, :config['n_modes_ff']]
        a_ss_tr = f['train/a_ss'][:, :config['n_modes_ss']]
        a_ff_ts = f['test/a_ff'][:, :config['n_modes_ff']]
        a_ss_ts = f['test/a_ss'][:, :config['n_modes_ss']]

    scaler_ss = StandardScaler().fit(a_ss_tr)
    scaler_ff = StandardScaler().fit(a_ff_tr)

    train_ds = MasterPODDataset(scaler_ss.transform(a_ss_tr), 
                               scaler_ff.transform(a_ff_tr), config['seq_len'])
    val_ds = MasterPODDataset(scaler_ss.transform(a_ss_ts), 
                             scaler_ff.transform(a_ff_ts), config['seq_len'])
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

    model = SupersamplingBiLSTM(config['n_modes_ss'], config['hidden_dim'], 
                                config['n_modes_ff'], config['num_layers']).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)
    criterion = nn.MSELoss()

    # Resume Logic
    start_epoch = 0
    if os.path.exists(ckpt_latest):
        print(f"Resuming from {ckpt_latest}")
        checkpoint = torch.load(ckpt_latest, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1

    print(f"Starting Flip-Flop Training on {device}...")
    for epoch in range(start_epoch, 2000):
        # Noise Annealing logic
        if epoch < 500:
            noise_std = config['final_noise'] + 0.5 * (config['initial_noise'] - config['final_noise']) * \
                        (1 + math.cos(math.pi * epoch / 500))
        else:
            noise_std = config['final_noise']

        model.train()
        t_loss = 0
        for bx, by in tqdm(train_loader, leave=False, desc=f"Ep {epoch}"):
            bx, by = bx.to(device), by.to(device)
            # Inject noise into sensor coefficients
            bx = bx + torch.randn_like(bx) * noise_std
            
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

        if (epoch + 1) % 5 == 0:
            print(f"Ep{epoch+1:03d} | Noise: {noise_std:.4f} | T: {avg_train:.2e} | V: {avg_val:.2e} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_ss': scaler_ss,
            'scaler_ff': scaler_ff,
            'config': config
        }, ckpt_latest)

if __name__ == "__main__":
    train()