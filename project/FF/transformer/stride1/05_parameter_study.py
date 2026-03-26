import os
import h5py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import math
import gc

# --- 1. CONFIGURATION & PATHS ---
base_dir = "/media/chris-remote/Projects/ONeill/estimation/output/kevin_2cylinder"
data_path = os.path.join(base_dir, "lstm_ready_data_FF_nearwake.h5")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the sweep parameters
strides = [1, 5, 10, 20]
model_sizes = {
    'small':  {'d_model': 64,  'nhead': 2, 'num_layers': 4, 'dim_feedforward': 128},
    'medium': {'d_model': 128, 'nhead': 4, 'num_layers': 6, 'dim_feedforward': 256},
    'large':  {'d_model': 256, 'nhead': 8, 'num_layers': 8, 'dim_feedforward': 512}
}

# Generate the full list of configurations
sweep_configs = []
for s in strides:
    for size_name, params in model_sizes.items():
        c = {
            'run_name': f"stride{s}_{size_name}",
            'stride': s,
            'input_dim': 40,
            'output_dim': 20,
            'dropout': 0.2,
            'seq_len': 200,
            'target_epochs': 10000 * s,  # Scales total epochs by stride
            'chunk_size': 1000 * s,      # Scales save interval by stride
            **params
        }
        sweep_configs.append(c)


# --- 2. DATASET & ARCHITECTURE ---
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
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
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


# --- 3. CHUNKED TRAINING ENGINE ---
def train_chunk(c):
    """Trains the given config for one chunk. Returns True if overall target reached."""
    
    model_dir = os.path.join(base_dir, "sweep_results", c['run_name'])
    os.makedirs(model_dir, exist_ok=True)
    latest_ckpt_path = os.path.join(model_dir, "latest_checkpoint.pt")

    # --- Load Data & Init Scalers ---
    with h5py.File(data_path, 'r') as f:
        a_ff_train = f['train/a_ff'][:, :c['output_dim']]
        a_ss_train = f['train/a_ss'][:, :c['input_dim']]
        a_ff_test = f['test/a_ff'][:, :c['output_dim']]
        a_ss_test = f['test/a_ss'][:, :c['input_dim']]

    scaler_ss = StandardScaler().fit(a_ss_train)
    scaler_ff = StandardScaler().fit(a_ff_train)
    
    train_ds = SparseLabelDataset(scaler_ss.transform(a_ss_train), 
                                  scaler_ff.transform(a_ff_train), c['seq_len'], c['stride'])
    val_ds = SparseLabelDataset(scaler_ss.transform(a_ss_test), 
                                scaler_ff.transform(a_ff_test), c['seq_len'], c['stride'])
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # --- Init Model & Optimizers ---
    model = SupersamplingTransformer(
        input_dim=c['input_dim'], output_dim=c['output_dim'], 
        d_model=c['d_model'], nhead=c['nhead'], 
        num_layers=c['num_layers'], dim_feedforward=c['dim_feedforward'],
        dropout=c['dropout']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=150, factor=0.5)
    criterion = nn.MSELoss()

    # --- Safe Resume Logic ---
    start_epoch = 0
    if os.path.exists(latest_ckpt_path):
        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint.get('epoch', 0) + 1

    if start_epoch >= c['target_epochs']:
        return True

    end_epoch = min(start_epoch + c['chunk_size'], c['target_epochs'])
    
    print(f"\n[{c['run_name']}] Resuming at Epoch {start_epoch} -> Training to {end_epoch} (Target: {c['target_epochs']})")

    # --- Chunk Training Loop ---
    for epoch in range(start_epoch, end_epoch):
        initial_noise = 0.02
        final_noise = 0.0005 
        anneal_epochs = c['target_epochs'] 

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

        # Print progress less frequently for high-stride (fast-epoch) runs
        print_interval = max(10, c['chunk_size'] // 10)
        if epoch % print_interval == 0 or epoch == end_epoch - 1:
            print(f"  Epoch {epoch:06d} | Train (Noisy): {avg_train:.2e} | Train (Clean): {avg_train_clean:.2e} | Val: {avg_val:.2e} | LR: {optimizer.param_groups[0]['lr']:.1e}")

    # --- Save Checkpoints ---
    save_dict = {
        'epoch': end_epoch - 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_ss': scaler_ss,
        'scaler_ff': scaler_ff,
        'config': c,
        'model_type': 'Transformer'
    }
    
    # Save the history checkpoint for tomorrow morning's analysis
    history_ckpt_path = os.path.join(model_dir, f"checkpoint_epoch_{end_epoch}.pt")
    torch.save(save_dict, history_ckpt_path)
    
    # Overwrite latest for smooth resuming
    torch.save(save_dict, latest_ckpt_path)

    # Aggressively clear memory before next model loads
    del model, optimizer, train_loader, val_loader, save_dict
    torch.cuda.empty_cache()
    gc.collect()

    return False


# --- 4. EXECUTION LOOP ---
if __name__ == "__main__":
    print(f"Starting Sweep for {len(sweep_configs)} Configurations...")
    
    all_done = False
    round_num = 1
    
    while not all_done:
        print(f"\n{'='*50}\nStarting Round {round_num}\n{'='*50}")
        all_done = True 
        
        for config in sweep_configs:
            is_finished = train_chunk(config)
            if not is_finished:
                all_done = False
                
        round_num += 1

    print("\nAll sweep configurations have reached their target epochs. Sweep complete!")