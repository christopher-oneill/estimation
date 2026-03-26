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

# --- 1. Versioning & Pathing ---
MODEL_NUMBER = 2
load_dotenv()
current_os = platform.system()
base_dir = os.getenv("BASE_DIR_WIN") if current_os == "Windows" else os.getenv("BASE_DIR_LIN")

# Derived Data Path
data_path = os.path.join(base_dir, "output/DNS_CC_Re150_Mazi/lstm_ready_data.h5")

# Model Specific Output Directory
model_dir = os.path.join(base_dir, f"output/DNS_CC_Re150_Mazi/Model_{MODEL_NUMBER}")
os.makedirs(model_dir, exist_ok=True)

ckpt_latest = os.path.join(model_dir, "latest_checkpoint.pt")

# GPU Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True 

# --- 2. Dataset Logic: Centered Window (Post-Processing) ---
class CenteredSupersamplingDataset(Dataset):
    def __init__(self, ss_data, ff_data, seq_len=100, stride=100):
        self.ss_data = torch.FloatTensor(ss_data)
        self.ff_data = torch.FloatTensor(ff_data)
        self.half_len = seq_len // 2
        
        # Valid indices must have room for the half-window on both sides
        # This ensures we have 'future' sensor data for every target point
        self.valid_indices = np.arange(self.half_len, len(ff_data) - self.half_len, stride)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        # X: Sensor window centered at t -> [t-50 : t+50]
        x = self.ss_data[t - self.half_len : t + self.half_len, :]
        # Y: Full field at t
        y = self.ff_data[t, :]
        return x, y

# --- 3. Architecture: Bidirectional LSTM ---
class SupersamplingBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, 
                            batch_first=True, dropout=0.1, bidirectional=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(), 
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        # For bidirectional, the last index of the output contains
        # the concatenated forward/backward context for the sequence
        last_hidden = out[:, -1, :] 
        return self.fc(last_hidden)

# --- 4. Training Pipeline ---
def train_model():
    print(f"--- Training Model {MODEL_NUMBER} ---")
    
    with h5py.File(data_path, 'r') as f:
        a_ff_train, a_ss_train = f['train/a_ff'][:], f['train/a_ss'][:]
        a_ff_test, a_ss_test = f['test/a_ff'][:], f['test/a_ss'][:]

    scaler_ss = StandardScaler().fit(a_ss_train)
    scaler_ff = StandardScaler().fit(a_ff_train)
    
    SEQ_LEN = 100 
    STRIDE_TRAIN = 100 
    HIDDEN_DIM = 512

    train_ds = CenteredSupersamplingDataset(scaler_ss.transform(a_ss_train), 
                                            scaler_ff.transform(a_ff_train), SEQ_LEN, STRIDE_TRAIN)
    # Validate on every step to monitor gap-filling performance
    val_ds = CenteredSupersamplingDataset(scaler_ss.transform(a_ss_test), 
                                          scaler_ff.transform(a_ff_test), SEQ_LEN, stride=1)
    
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    model = SupersamplingBiLSTM(a_ss_train.shape[1], HIDDEN_DIM, a_ff_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)
    criterion = nn.MSELoss()

    start_epoch = 0
    history = {'train': [], 'val': []}

    # Resume Logic
    if os.path.exists(ckpt_latest):
        print(f"Loading previous checkpoint for Model {MODEL_NUMBER}...")
        checkpoint = torch.load(ckpt_latest, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('history', history)

    # Main Loop
    for epoch in range(start_epoch, 2000):
        model.train()
        t_loss = 0
        for bx, by in train_loader:
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
        
        history['train'].append(avg_train)
        history['val'].append(avg_val)

        if (epoch + 1) % 10 == 0:
            print(f"Mod{MODEL_NUMBER} Ep{epoch+1} | T: {avg_train:.2e} | V: {avg_val:.2e} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # --- Save Checkpoints ---
        save_dict = {
            'model_num': MODEL_NUMBER,
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'history': history,
            'scaler_ss': scaler_ss,
            'scaler_ff': scaler_ff,
            'config': {'hidden_dim': HIDDEN_DIM, 'seq_len': SEQ_LEN}
        }
        
        # Save the "latest" in the specific model folder
        torch.save(save_dict, ckpt_latest)
        
        # Save historical snapshots
        if (epoch + 1) % 200 == 0:
            hist_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(save_dict, hist_path)

if __name__ == "__main__":
    train_model()