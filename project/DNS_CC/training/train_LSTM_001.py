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

# 1. Environment & Device Setup
load_dotenv()
current_os = platform.system()
base_dir = os.getenv("BASE_DIR_WIN") if current_os == "Windows" else os.getenv("BASE_DIR_LIN")
data_path = os.path.join(base_dir, "data/DNS_CC_Re150_Mazi/lstm_ready_data.h5")

# Leverage the 4090
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True 

# 2. Custom Dataset for Sparse Labels
class SparseSupersamplingDataset(Dataset):
    def __init__(self, ss_data, ff_data, seq_len=100, stride=100):
        self.ss_data = torch.FloatTensor(ss_data)
        self.ff_data = torch.FloatTensor(ff_data)
        self.seq_len = seq_len
        
        # We only train on indices where we have a Full Field "Anchor"
        # We start at 'seq_len' so the first window has enough history
        self.valid_indices = np.arange(seq_len, len(ff_data), stride)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        # Input: History of the sensor strip [t-seq_len : t]
        x = self.ss_data[t - self.seq_len : t, :]
        # Target: The single full-field state at time t
        y = self.ff_data[t, :]
        return x, y

# 3. Model Architecture
class SupersamplingLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # x: (Batch, Seq_len, Input_dim)
        out, (hn, cn) = self.lstm(x)
        # We only need the hidden state of the last timestep
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)

# 4. Main Training Pipeline
def train_model():
    print(f"Loading data from {data_path}...")
    with h5py.File(data_path, 'r') as f:
        a_ff_train = f['train/a_ff'][:]
        a_ss_train = f['train/a_ss'][:]
        
        # We use the 'test' group for real validation (the original DNS)
        a_ff_test = f['test/a_ff'][:]
        a_ss_test = f['test/a_ss'][:]

    # Normalization (Crucial for LSTMs)
    # Scale based on the training set's statistics
    scaler_ss = StandardScaler().fit(a_ss_train)
    scaler_ff = StandardScaler().fit(a_ff_train)
    
    a_ss_train_norm = scaler_ss.transform(a_ss_train)
    a_ff_train_norm = scaler_ff.transform(a_ff_train)
    a_ss_test_norm = scaler_ss.transform(a_ss_test)
    a_ff_test_norm = scaler_ff.transform(a_ff_test)

    # Hyperparameters
    SEQ_LEN = 100
    STRIDE = 100 # Only "see" a PIV frame every 100 steps
    BATCH_SIZE = 256 # 4090 can handle much larger, but 256 is good for 1000 samples
    HIDDEN_DIM = 256
    EPOCHS = 200
    LR = 1e-3

    train_ds = SparseSupersamplingDataset(a_ss_train_norm, a_ff_train_norm, SEQ_LEN, STRIDE)
    # For validation, we can use a stride of 1 to see how it performs on every step
    val_ds = SparseSupersamplingDataset(a_ss_test_norm, a_ff_test_norm, SEQ_LEN, stride=1)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SupersamplingLSTM(a_ss_train.shape[1], HIDDEN_DIM, a_ff_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    print(f"Training on {len(train_ds)} sparse samples...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_pred = model(vx)
                val_loss += criterion(v_pred, vy).item()
        
        avg_val = val_loss/len(val_loader)
        scheduler.step(avg_val)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {avg_val:.6f}")

    # Save Model and Scalers
    torch.save({
        'model_state': model.state_dict(),
        'scaler_ss': scaler_ss,
        'scaler_ff': scaler_ff,
        'config': {'seq_len': SEQ_LEN, 'hidden_dim': HIDDEN_DIM}
    }, "supersampling_lstm.pt")
    print("Model saved as supersampling_lstm.pt")

if __name__ == "__main__":
    train_model()