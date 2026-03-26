import os
import h5py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- 1. CONFIGURATION ---
MODEL_NUMBER = 2
base_dir = "/media/chris-remote/Projects/ONeill/estimation/output/kevin_2cylinder/MP"
master_pod_path = os.path.join(base_dir, "mp_pod_master.h5") # Your pre-computed file
model_dir = os.path.join(base_dir, f"Model_BiLSTM_{MODEL_NUMBER}")
os.makedirs(model_dir, exist_ok=True)

ckpt_latest = os.path.join(model_dir, "latest_checkpoint.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
config = {
    'n_modes_ss': 240,   # Input: First 33 sensor modes (Observable limit)
    'n_modes_ff': 123,   # Output: First 10 field modes (~10% vel error)
    'seq_len': 100,     # 100 snapshot window (centered)
    'hidden_dim': 1024,  # Large enough to track modulation frequencies
    'num_layers': 6,
    'batch_size': 512,
    'lr': 1e-3
}

# --- 2. DATASET LOGIC ---
class MasterPODDataset(Dataset):
    """Loads and slices pre-computed POD coefficients from the Master file."""
    def __init__(self, ss_coeffs, ff_coeffs, seq_len=100):
        self.ss_data = torch.FloatTensor(ss_coeffs)
        self.ff_data = torch.FloatTensor(ff_coeffs)
        self.half_len = seq_len // 2
        # Indices for centered window [t-50 : t+50]
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
        # Context from the bidirectional "meeting point" at the end of the sequence
        return self.fc(out[:, -1, :])

# --- 4. TRAINING ENGINE ---
def train():
    print(f"Loading Master POD data: {config['n_modes_ss']} SS -> {config['n_modes_ff']} FF")
    
    with h5py.File(master_pod_path, 'r') as f:
        # Slicing directly from the 484-mode master file
        a_ff_tr = f['train/a_ff'][:, :config['n_modes_ff']]
        a_ss_tr = f['train/a_ss'][:, :config['n_modes_ss']]
        a_ff_ts = f['test/a_ff'][:, :config['n_modes_ff']]
        a_ss_ts = f['test/a_ss'][:, :config['n_modes_ss']]

    # Standardizing is still required even for POD coefficients
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    criterion = nn.MSELoss()

    # --- MAIN LOOP ---
    print("Starting Bi-LSTM Training for Modulated Periodic Wake...")
    for epoch in range(1000):
        model.train()
        t_loss = 0
        for bx, by in tqdm(train_loader, leave=False, desc=f"Epoch {epoch}"):
            bx, by = bx.to(device), by.to(device)
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
            print(f"Ep{epoch+1:03d} | T: {avg_train:.2e} | V: {avg_val:.2e} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        # --- SAVE CHECKPOINT ---
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