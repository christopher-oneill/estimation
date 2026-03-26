import os
import h5py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import imageio_ffmpeg
from tqdm import tqdm
import math

# --- 1. CONFIGURATION ---
MODEL_NUM = 2 # Ensure this matches your directory
base_dir = "/media/chris-remote/Projects/ONeill/estimation/output/kevin_2cylinder/MP"
unified_h5 = os.path.join(base_dir, "MP_velocity_unified.h5")
master_pod_path = os.path.join(base_dir,"strip10_stride1_field_only", "mp_pod_master_raw_sensor.h5")

model_dir = os.path.join(base_dir, "strip10_stride1_field_only",f"Model_BiLSTM_{MODEL_NUM}")
ckpt_path = os.path.join(model_dir, "latest_checkpoint.pt")
video_out = os.path.join(model_dir, f"MP_Reconstruction_LSTM_M{MODEL_NUM}.mp4")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. ARCHITECTURE DEFINITION ---
class SupersamplingBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        # Context from the bidirectional meeting point at the end of the sequence
        return self.fc(out[:, -1, :])

# --- 3. LOAD CHECKPOINT & MAP KEYS ---
print("Loading Checkpoint...")
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
c = checkpoint['config']

# Mapping your specific keys to the model initialization
# Using .get() ensures we don't crash if a name changes later
N_SS = c.get('n_modes_ss')
N_FF = c.get('n_modes_ff')
H_DIM = c.get('hidden_dim')
N_LAY = c.get('num_layers')
S_LEN = c.get('seq_len')

print(f"Model Specs: {N_SS} SS -> {N_FF} FF | Hidden: {H_DIM} | Layers: {N_LAY}")

model = SupersamplingBiLSTM(
    input_dim=N_SS, 
    hidden_dim=H_DIM, 
    output_dim=N_FF,
    n_layers=N_LAY
).to(device)

model.load_state_dict(checkpoint['model_state'])
model.eval()

scaler_ss = checkpoint['scaler_ss']
scaler_ff = checkpoint['scaler_ff']

# --- 4. LOAD PHYSICAL DATA ---
with h5py.File(master_pod_path, 'r') as f:
    XI, YI = f['XI'][:], f['YI'][:]
    grid_shape = tuple(f['grid_shape'][:])
    phi_ff = f['phi_ff'][:N_FF, :] 
    mean_ff_vec = f['mean_ff'][:]
    
    a_ss_tr, a_ss_ts = f['train/s_raw'][:, :N_SS], f['test/s_raw'][:, :N_SS]
    a_ff_tr, a_ff_ts = f['train/a_ff'][:, :N_FF], f['test/a_ff'][:, :N_FF]

a_ss_all = np.vstack([a_ss_tr, a_ss_ts])
a_ff_true = np.vstack([a_ff_tr, a_ff_ts])
n_train = len(a_ss_tr)

with h5py.File(unified_h5, 'r') as f:
    ux_raw_test = f['test/ux'][:]

# --- 5. INFERENCE ---
print("Running Inference...")
half_len = S_LEN // 2
a_ss_norm = scaler_ss.transform(a_ss_all)
a_ff_est_norm = np.zeros((len(a_ss_all), N_FF))

with torch.no_grad():
    for i in tqdm(range(half_len, len(a_ss_all) - half_len)):
        window = a_ss_norm[i - half_len : i + half_len, :]
        batch = torch.FloatTensor(window).unsqueeze(0).to(device)
        a_ff_est_norm[i] = model(batch).cpu().numpy()

a_ff_est = scaler_ff.inverse_transform(a_ff_est_norm)
valid_idx = np.arange(half_len, len(a_ss_all) - half_len)
a_ff_true_valid = a_ff_true[valid_idx]
a_ff_est_valid = a_ff_est[valid_idx]

# --- 6. PLOT: TRACKING & ERROR ---
print("Generating Diagnostic Plots...")
# Top 4 modes tracking
fig, axes = plt.subplots(2, 2, figsize=(15, 8))
for i, ax in enumerate(axes.flatten()):
    ax.plot(valid_idx, a_ff_true_valid[:, i], 'k', alpha=0.3, label='DNS')
    ax.plot(valid_idx, a_ff_est_valid[:, i], 'r--', label='LSTM')
    ax.axvline(n_train, color='b', linestyle=':')
    ax.set_title(f"POD Mode {i+1}")
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "mode_tracking.png"))
plt.show()

# Integrated field error
phi_u = phi_ff[:, :phi_ff.shape[1]//2]
u_true_p = a_ff_true_valid @ phi_u
u_est_p = a_ff_est_valid @ phi_u
u_err = np.sqrt(np.mean((u_true_p - u_est_p)**2, axis=1))

plt.figure(figsize=(12, 4))
plt.plot(valid_idx, u_err, color='blue')
plt.axvline(n_train, color='red', linestyle='--')
plt.title("Full-Field RMSE (Velocity Scale)")
plt.savefig(os.path.join(model_dir, "field_rmse.png"))
plt.show()

# --- 7. VIDEO GENERATION (ENTIRE TIMESERIES) ---
print(f"Rendering Complete Dataset Video (Train + Test) to {video_out}...")

# 1. Unify Raw DNS data for the background
with h5py.File(unified_h5, 'r') as f:
    ux_raw_tr = f['train/ux'][:]
    ux_raw_ts = f['test/ux'][:]
ux_raw_all = np.vstack([ux_raw_tr, ux_raw_ts])

# 2. Reconstruct Full Timeseries (no masking)
u_mean_2d = mean_ff_vec[:len(mean_ff_vec)//2].reshape(grid_shape, order='C')

# These use the full length of valid_idx (Training + Testing segments)
up_lom_all = (a_ff_true_valid @ phi_u)
up_est_all = (a_ff_est_valid @ phi_u)

# 3. Dynamic range from a sample of the whole dataset
u_sample = ux_raw_all[len(ux_raw_all)//2].reshape(grid_shape, order='C')
u_min, u_max = np.percentile(u_sample, 1), np.percentile(u_sample, 99)
levs = np.linspace(u_min, u_max, 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
width, height = fig.canvas.get_width_height()

cmd = [
    imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
    '-s', f'{width}x{height}', '-pix_fmt', 'rgba', '-r', '30',
    '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-preset', 'fast', '-crf', '18', video_out
]
process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

try:
    # Iterate over every valid index in the entire history
    for t in tqdm(range(len(up_est_all)), desc="Encoding Full Series"):
        # Match current index to the stacked raw data
        current_global_idx = valid_idx[t]
        
        u_raw = ux_raw_all[current_global_idx].reshape(grid_shape, order='C')
        u_lom = u_mean_2d + up_lom_all[t].reshape(grid_shape, order='C')
        u_est = u_mean_2d + up_est_all[t].reshape(grid_shape, order='C')
        
        u_err = (u_est - u_lom) / max(abs(u_max), 1e-3)
        err_lim = max(np.percentile(np.abs(u_err), 99), 0.01)

        for ax in axes.flatten(): 
            ax.clear()
        
        # Determine if we are in the Training or Testing segment
        mode_text = "TRAINING" if current_global_idx < n_train else "TESTING (UNSEEN)"
        mode_color = "blue" if current_global_idx < n_train else "red"
        
        axes[0,0].contourf(XI, YI, u_raw, levels=levs, cmap='RdBu_r', extend='both')
        axes[0,0].set_title('DNS Ground Truth')
        
        axes[0,1].contourf(XI, YI, u_lom, levels=levs, cmap='RdBu_r', extend='both')
        axes[0,1].set_title(f'LOM Target ({N_FF} Modes)')
        
        axes[1,0].contourf(XI, YI, u_est, levels=levs, cmap='RdBu_r', extend='both')
        axes[1,0].set_title('Bi-LSTM Reconstruction')
        
        axes[1,1].contourf(XI, YI, u_err, levels=np.linspace(-err_lim, err_lim, 50), cmap='PuOr', extend='both')
        axes[1,1].set_title(f'Rel. Error | Max: {err_lim:.3f}')
        
        fig.suptitle(f"Full Timeseries | {mode_text} | Frame {t}/{len(up_est_all)}", 
                     fontsize=16, color=mode_color, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        fig.canvas.draw()
        process.stdin.write(fig.canvas.buffer_rgba())

except Exception as e:
    print(f"Render Error at frame {t}: {e}")
finally:
    process.stdin.close()
    process.wait()
    plt.close(fig)

print(f"Entire timeseries video saved to: {video_out}")

print("Inference Complete.")