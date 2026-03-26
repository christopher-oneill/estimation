import os
import platform
import h5py
import torch
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

# 1. Environment & Pathing
load_dotenv()
current_os = platform.system()
base_dir = os.getenv("BASE_DIR_WIN") if current_os == "Windows" else os.getenv("BASE_DIR_LIN")
data_path = os.path.join(base_dir, "data/DNS_CC_Re150_Mazi/lstm_ready_data.h5")
model_path = "supersampling_lstm.pt"
output_path = os.path.join(base_dir, "data/DNS_CC_Re150_Mazi/inference_results.h5")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Re-define Model Class (Must match Training)
class SupersamplingLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def run_inference():
    # 3. Load Checkpoint and Metadata
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    scaler_ss = checkpoint['scaler_ss']
    scaler_ff = checkpoint['scaler_ff']
    config = checkpoint['config']
    seq_len = config['seq_len']

    with h5py.File(data_path, 'r') as f:
        # Load test coefficients
        a_ff_test = f['test/a_ff'][:]
        a_ss_test = f['test/a_ss'][:]
        test_time = f['test/time'][:]
            
        # Load spatial modes 
        phi_ff = f['phi_ff'][:]
            
        # FIX: Load the unified mean and slice it in half
        mean_ff_all = f['mean_ff'][:]
        n_points = mean_ff_all.shape[0] // 2
        mean_ff_ux = mean_ff_all[:n_points]
        mean_ff_uy = mean_ff_all[n_points:]
            
        n_ff = a_ff_test.shape[1]
        n_points = mean_ff_ux.shape[0]

    # Initialize model
    model = SupersamplingLSTM(a_ss_test.shape[1], config['hidden_dim'], n_ff).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # 4. Prepare Test Data Sequences
    print("Normalizing test sequences...")
    a_ss_test_norm = scaler_ss.transform(a_ss_test)
    
    # We will predict for every possible step after the initial seq_len
    # Test set: 4100. If seq_len=100, we predict from index 100 to 4099 (4000 steps)
    indices = np.arange(seq_len, len(a_ss_test))
    
    a_ff_pred_norm = []
    
    print(f"Running inference on {len(indices)} test snapshots...")
    with torch.no_grad():
        # We can batch the inference for the 4090
        batch_size = 512
        for i in tqdm(range(0, len(indices), batch_size)):
            batch_indices = indices[i:i+batch_size]
            
            # Create sliding window batch
            x_batch = []
            for idx in batch_indices:
                x_batch.append(a_ss_test_norm[idx - seq_len : idx, :])
            
            x_tensor = torch.FloatTensor(np.array(x_batch)).to(device)
            preds = model(x_tensor)
            a_ff_pred_norm.append(preds.cpu().numpy())

    # Concatenate and inverse transform coefficients
    a_ff_pred_norm = np.vstack(a_ff_pred_norm)
    a_ff_pred = scaler_ff.inverse_transform(a_ff_pred_norm)

    # 5. Physical Reconstruction
    print("Reconstructing physical velocity fields...")
    # Predicted Field = Mean + Pred_Coeffs * Modes
    # Note: phi_ff shape is (n_modes, 2 * n_points)
    # ux is first half, uy is second half
    ux_pred = mean_ff_ux + a_ff_pred @ phi_ff[:, :n_points]
    uy_pred = mean_ff_uy + a_ff_pred @ phi_ff[:, n_points:]
    
    # Also reconstruct True Field for direct comparison (at the same indices)
    a_ff_true = a_ff_test[indices]
    ux_true = mean_ff_ux + a_ff_true @ phi_ff[:, :n_points]
    uy_true = mean_ff_uy + a_ff_true @ phi_ff[:, n_points:]

    # 6. Save Results
    print(f"Saving results to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        # Save time vector matching the predictions
        f.create_dataset('time', data=test_time[indices])
        
        # Save predicted and ground truth fields
        # Shape: (4000, N_points)
        f.create_dataset('ux_pred', data=ux_pred.astype(np.float32))
        f.create_dataset('uy_pred', data=uy_pred.astype(np.float32))
        f.create_dataset('ux_true', data=ux_true.astype(np.float32))
        f.create_dataset('uy_true', data=uy_true.astype(np.float32))
        
        # Save error metrics for quick access later
        mse_ux = np.mean((ux_pred - ux_true)**2)
        mse_uy = np.mean((uy_pred - uy_true)**2)
        f.attrs['mse_ux'] = mse_ux
        f.attrs['mse_uy'] = mse_uy

    print(f"Inference complete. Global MSE: {mse_ux + mse_uy:.6e}")

if __name__ == "__main__":
    run_inference()