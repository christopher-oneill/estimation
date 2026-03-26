import os
import h5py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import math

# --- 1. CONFIGURATION ---
MODEL_NUM = 5
base_dir = "/media/chris-remote/Projects/ONeill/estimation"
output_dir = os.path.join(base_dir, "output/kevin_2cylinder")
raw_data_dir = os.path.join(base_dir, "data/kevin_2cylinder/LD0_TD2")

unified_h5 = os.path.join(output_dir, "flipflop_velocity_unified.h5")
pod_data_path = os.path.join(output_dir, "lstm_ready_data_FF_nearwake.h5")
grid_info_path = os.path.join(raw_data_dir, "GridInfo_FF.mat")

model_dir = os.path.join(output_dir, f"transformer_{MODEL_NUM}_FF")
ckpt_path = os.path.join(model_dir, "latest_checkpoint.pt")
video_out = os.path.join(model_dir, "FF_Reconstruction_vorticity.mp4")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. ARCHITECTURE DEFINITION ---
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
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.2):
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
        return self.out_layer(torch.mean(x, dim=1))

# --- 3. LOAD CHECKPOINT & CONFIG ---
print("Loading Checkpoint...")
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
c = checkpoint['config']  # DYNAMIC CONFIG!

model = SupersamplingTransformer(
    input_dim=c['input_dim'], output_dim=c['output_dim'], 
    d_model=c['d_model'], nhead=c['nhead'], 
    num_layers=c['num_layers'], dim_feedforward=c['dim_feedforward'],
    dropout=c['dropout']
).to(device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

scaler_ss = checkpoint['scaler_ss']
scaler_ff = checkpoint['scaler_ff']

# --- 4. LOAD DATA ---
print(f"Loading Grid & Data (Target: {c['output_dim']} modes)...")
with h5py.File(grid_info_path, 'r') as f:
    # .T handles MATLAB v7.3 reading orientation
    XI = np.array(f['XI'])
    YI = np.array(f['YI'])
    grid_shape = XI.shape

with h5py.File(pod_data_path, 'r') as f:
    phi_ff = f['phi_ff'][:c['output_dim'], :] 
    mean_ff_vec = f['mean_ff'][:]
    a_ss_train = f['train/a_ss'][:, :c['input_dim']]
    a_ss_test = f['test/a_ss'][:, :c['input_dim']]
    a_ff_train = f['train/a_ff'][:, :c['output_dim']]
    a_ff_test = f['test/a_ff'][:, :c['output_dim']]

a_ss_all = np.vstack([a_ss_train, a_ss_test])
a_ff_true = np.vstack([a_ff_train, a_ff_test])
n_train = len(a_ss_train)

# Load Raw Test Data for Video Comparison
with h5py.File(unified_h5, 'r') as f:
    ux_raw_test = f['test/ux'][:]
    uy_raw_test = f['test/uy'][:]

# --- 5. INFERENCE OVER FULL TIME SERIES ---
print("Running Inference...")
seq_len = c['seq_len']
half_len = seq_len // 2
n_total = len(a_ss_all)

a_ss_norm = scaler_ss.transform(a_ss_all)
a_ff_est_norm = np.zeros((n_total, c['output_dim']))

with torch.no_grad():
    for i in tqdm(range(half_len, n_total - half_len)):
        window = a_ss_norm[i - half_len : i + half_len, :]
        batch = torch.FloatTensor(window).unsqueeze(0).to(device)
        a_ff_est_norm[i] = model(batch).cpu().numpy()

a_ff_est = scaler_ff.inverse_transform(a_ff_est_norm)

# Align valid predictions (trim the unpredicted edges)
valid_idx = np.arange(half_len, n_total - half_len)
a_ff_true_valid = a_ff_true[valid_idx]
a_ff_est_valid = a_ff_est[valid_idx]

# --- 6. PLOT: TEMPORAL COEFFICIENTS ---
print("Plotting Temporal Coefficients...")
cols = 4
rows = math.ceil(c['output_dim'] / cols)
fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows), sharex=True)
axes = axes.flatten()

for i in range(c['output_dim']):
    axes[i].plot(valid_idx, a_ff_true_valid[:, i], 'k', alpha=0.5, label='True')
    axes[i].plot(valid_idx, a_ff_est_valid[:, i], 'r--', alpha=0.8, label='Estimated')
    axes[i].axvline(n_train, color='b', linestyle=':', alpha=0.5, label='Test Split')
    axes[i].set_title(f"Mode {i+1}")
    if i == 0: axes[i].legend(loc='upper right')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j]) # Clean up empty subplots
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "coefficient_tracking.png"))
plt.show()

# --- 7. PLOT: INTEGRATED u' & v' ERRORS ---
print("Plotting Integrated Errors...")
phi_u = phi_ff[:, :phi_ff.shape[1]//2]
phi_v = phi_ff[:, phi_ff.shape[1]//2:]

u_true_prime = a_ff_true_valid @ phi_u
v_true_prime = a_ff_true_valid @ phi_v
u_est_prime = a_ff_est_valid @ phi_u
v_est_prime = a_ff_est_valid @ phi_v

u_err_t = np.sqrt(np.mean((u_true_prime - u_est_prime)**2, axis=1))
v_err_t = np.sqrt(np.mean((v_true_prime - v_est_prime)**2, axis=1))

plt.figure(figsize=(12, 5))
plt.plot(valid_idx, u_err_t, label="u' RMSE", color='blue')
plt.plot(valid_idx, v_err_t, label="v' RMSE", color='green')
plt.axvline(n_train, color='red', linestyle='--', label='Train/Test Split')
plt.title("Full-Field Integrated Temporal Error (RMSE)")
plt.ylabel("RMSE Magnitude")
plt.xlabel("Time Step")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(model_dir, "integrated_error.png"))
plt.show()




# --- 8. MEAN FIELD & REYNOLDS STRESSES (TEST SET ONLY) ---
print("Computing Test-Set Statistics...")
# We evaluate stats strictly on the unseen test set
test_valid_idx = valid_idx >= n_train
u_est_test = u_est_prime[test_valid_idx]
v_est_test = v_est_prime[test_valid_idx]

# When reshaping, explicitly use order='F' to match MATLAB's flattening
u_mean = mean_ff_vec[:len(mean_ff_vec)//2].reshape(grid_shape, order='C')
v_mean = mean_ff_vec[len(mean_ff_vec)//2:].reshape(grid_shape, order='C')
uu = np.mean(u_est_test**2, axis=0).reshape(grid_shape, order='C')
vv = np.mean(v_est_test**2, axis=0).reshape(grid_shape, order='C')
uv = np.mean(u_est_test * v_est_test, axis=0).reshape(grid_shape, order='C')

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
stats = [u_mean, v_mean, uu, uv, vv]
titles = ['Mean U', 'Mean V', "u'u'", "u'v'", "v'v'"]

for i, ax in enumerate(axes.flatten()[:5]):
    im = ax.contourf(XI, YI, stats[i], levels=50, cmap='RdBu_r' if i < 2 else 'viridis')
    ax.set_title(titles[i])
    plt.colorbar(im, ax=ax)
fig.delaxes(axes[1,2])
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "reynolds_stresses.png"))
plt.show()

import imageio_ffmpeg
import subprocess


# --- 7. VIDEO GENERATION (U-Velocity) ---
print(f"Rendering U-Velocity Video to {video_out}...")

# 1. Ensure we have the mean field to add back to the fluctuations
u_mean = mean_ff_vec[:len(mean_ff_vec)//2].reshape(grid_shape, order='C')

# 2. Slice valid test data
test_mask = valid_idx >= n_train
a_ff_true_test = a_ff_true_valid[test_mask]
a_ff_est_test = a_ff_est_valid[test_mask]

up_lom_all = (a_ff_true_test @ phi_ff[:, :phi_ff.shape[1]//2])
up_est_all = (a_ff_est_test @ phi_ff[:, :phi_ff.shape[1]//2])

# 3. Robust Color Limits (NaN guarded)
# Take a sample of the raw U field to set fixed colorbar limits
u_sample = ux_raw_test[half_len].reshape(grid_shape, order='C')
u_sample = np.nan_to_num(u_sample, nan=0.0) # Scrub any stray NaNs

u_min, u_max = np.percentile(u_sample, 1), np.percentile(u_sample, 99)
if np.isnan(u_min) or np.isnan(u_max) or u_min == u_max:
    u_min, u_max = -1.0, 1.0 # Safety fallback

print(f"Dynamic U-Velocity Limits locked to: [{u_min:.3f}, {u_max:.3f}]")

# Lock the contour levels so they don't flash/change between frames
levels_u = np.linspace(u_min, u_max, 50)

# 4. Figure Setup
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
fig.canvas.draw()
width, height = fig.canvas.get_width_height()

ffmpeg_cmd = [
    imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
    '-s', f'{width}x{height}', '-pix_fmt', 'rgba', '-r', '20',
    '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-preset', 'fast', '-crf', '18', video_out
]
process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

try:
    num_frames = min(400, len(up_est_all))
    for t in tqdm(range(0, num_frames, 2)):
        
        # Extract and scrub data
        # Raw DNS is full velocity. LOM and Est are Fluctuations + Mean
        u_raw = ux_raw_test[t + half_len].reshape(grid_shape, order='C')
        u_lom = u_mean + up_lom_all[t].reshape(grid_shape, order='C')
        u_est = u_mean + up_est_all[t].reshape(grid_shape, order='C')
        
        # Scrub NaNs just in case
        u_raw = np.nan_to_num(u_raw, nan=0.0)
        u_lom = np.nan_to_num(u_lom, nan=0.0)
        u_est = np.nan_to_num(u_est, nan=0.0)
        
        u_err = (u_est - u_lom)/u_max
        
        # Dynamically set error limits based on current frame to ensure visibility
        err_max = max(np.max(np.abs(u_err)), 1e-4)
        levels_err = np.linspace(-err_max, err_max, 50)

        # Clear axes to prevent memory leaks and overlapping contours
        for ax in axes.flatten():
            ax.clear()
            
        # Draw contour plots
        axes[0,0].contourf(XI, YI, u_raw, levels=levels_u, cmap='RdBu_r', extend='both')
        axes[0,0].set_title('Raw DNS U-Velocity')
        
        axes[0,1].contourf(XI, YI, u_lom, levels=levels_u, cmap='RdBu_r', extend='both')
        axes[0,1].set_title(f'LOM U-Velocity ({c["output_dim"]} modes)')
        
        axes[1,0].contourf(XI, YI, u_est, levels=levels_u, cmap='RdBu_r', extend='both')
        axes[1,0].set_title('Transformer Estimated U-Velocity')
        
        axes[1,1].contourf(XI, YI, u_err, levels=levels_err, cmap='PuOr', extend='both')
        axes[1,1].set_title(f'Error (Est - LOM) | Max: {err_max:.3f}')
        
        plt.tight_layout()
        
        # Force a re-render and push to FFmpeg buffer
        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        process.stdin.write(rgba_buffer)
        
except Exception as e:
    print(f"Render Error: {e}")
finally:
    process.stdin.close()
    process.wait()
    plt.close(fig)

print(f"Successfully saved video to: {video_out}")