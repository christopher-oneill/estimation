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
MODEL_NUM = 1
output_dir = "/media/chris-remote/Projects/ONeill/estimation/output/kevin_2cylinder"
model_dir = os.path.join(output_dir, f"transformer_{MODEL_NUM}_FF")
os.makedirs(model_dir, exist_ok=True)

data_path = os.path.join(output_dir, "lstm_ready_data_FF.h5")
ckpt_path = os.path.join(model_dir, "latest_checkpoint.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. DATASET: SPARSE LABELS / DENSE INPUTS ---
class SparseLabelDataset(Dataset):
    """
    Inputs (ss): Continuous sequence at full temporal resolution.
    Labels (ff): Only available every 'stride' timesteps.
    """
    def __init__(self, ss_data, ff_data, seq_len=100, stride=20):
        self.ss_data = torch.FloatTensor(ss_data)
        self.ff_data = torch.FloatTensor(ff_data)
        self.half_len = seq_len // 2
        self.stride = stride
        
        # Valid indices are multiples of stride that allow for the lookback/lookahead window
        full_range = np.arange(self.half_len, len(ff_data) - self.half_len)
        self.valid_indices = full_range[full_range % self.stride == 0]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        # The Bi-LSTM sees the full dense sequence around the sparse anchor
        x = self.ss_data[t - self.half_len : t + self.half_len, :]
        y = self.ff_data[t, :]
        return x, y

# --- 3. ARCHITECTURE ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SupersamplingTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        
        # Project input modes to d_model space
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Map the final attended representation to the 178 FF modes
        self.out_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Input_Dim)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Attention across the sequence
        x = self.transformer_encoder(x)
        
        # We take the mean across the sequence dimension (Global Average Pooling)
        # or just the last token. For fluid dynamics, the mean is often more stable.
        x = torch.mean(x, dim=1) 
        
        return self.out_layer(x)
    

# --- 4. TRAINING ENGINE ---
def train():
    print(f"Loading Flip-Flop Dataset: {data_path}")
    with h5py.File(data_path, 'r') as f:
        a_ff_train, a_ss_train = f['train/a_ff'][:], f['train/a_ss'][:]
        a_ff_test, a_ss_test = f['test/a_ff'][:], f['test/a_ss'][:]

    # Normalization is critical for POD coefficients
    scaler_ss = StandardScaler().fit(a_ss_train)
    scaler_ff = StandardScaler().fit(a_ff_train)
    
    # 20x Stride for both Train and Val to mimic experimental PIV constraints
    train_ds = SparseLabelDataset(scaler_ss.transform(a_ss_train), 
                                  scaler_ff.transform(a_ff_train), seq_len=100, stride=20)
    val_ds = SparseLabelDataset(scaler_ss.transform(a_ss_test), 
                                scaler_ff.transform(a_ff_test), seq_len=100, stride=20)
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    input_dim = a_ss_train.shape[1]  # Should be 267
    output_dim = a_ff_train.shape[1] # Should be 178
    HIDDEN_DIM = 1024

    model = SupersamplingTransformer(
        input_dim=267, 
        output_dim=178, 
        d_model=256, 
        nhead=4,         # Fewer heads for small data
        num_layers=3,      # Shallower to prevent memorization
        dropout=0.3        # Higher dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)
    criterion = nn.MSELoss()

    # Resume Logic
    start_epoch = 0
    if os.path.exists(ckpt_path):
        print("Resuming from checkpoint...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1

    print(f"Starting Training: {input_dim} SS Modes -> {output_dim} FF Modes")
    for epoch in range(start_epoch, 1000):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            
            # Jitter: Add 1% white noise to sensor data
            bx = bx + torch.randn_like(bx) * 0.01 
            
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                val_loss += criterion(model(vx.to(device)), vy.to(device)).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Train: {avg_train:.2e} | Val: {avg_val:.2e} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_ss': scaler_ss,
            'scaler_ff': scaler_ff,
            'config': {'hidden_dim': HIDDEN_DIM, 'seq_len': 100, 'input_dim': input_dim, 'output_dim': output_dim},
            'model_type': 'Bi-LSTM'
        }, ckpt_path)

if __name__ == "__main__":
    train()