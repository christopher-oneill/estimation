import os
import platform
import h5py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from tqdm import tqdm
import imageio_ffmpeg
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
import math

# --- 1. CONFIGURATION & PATHS ---
MODEL_NUMBER = 3  # Transformer Model Number
load_dotenv()
current_os = platform.system()
base_dir = os.getenv("BASE_DIR_WIN") if current_os == "Windows" else os.getenv("BASE_DIR_LIN")

derived_dir = os.path.join(base_dir, "output/DNS_CC_Re150_Mazi")
model_dir = os.path.join(derived_dir, f"Model_T_{MODEL_NUMBER}")
os.makedirs(model_dir, exist_ok=True)

data_path = os.path.join(derived_dir, "lstm_ready_data.h5")
grid_h5 = os.path.join(derived_dir, "dns_regular_0.02D_2d.h5")
ckpt_path = os.path.join(model_dir, "latest_checkpoint.pt")
video_out = os.path.join(model_dir, f"vorticity_recon_T{MODEL_NUMBER}.mp4")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

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
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
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

# --- 3. LOAD DATA & MODEL ---
print("Loading Checkpoint and Metadata...")
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
c = checkpoint['config']
scaler_ss, scaler_ff = checkpoint['scaler_ss'], checkpoint['scaler_ff']
half_len = c['seq_len'] // 2

with h5py.File(data_path, 'r') as f:
    a_ff_test, a_ss_test = f['test/a_ff'][:], f['test/a_ss'][:]
    test_time, phi_ff, mean_ff = f['test/time'][:], f['phi_ff'][:], f['mean_ff'][:]

with h5py.File(grid_h5, 'r') as f_grid:
    x_coords, y_coords = f_grid['x'][:], f_grid['y'][:]

model = SupersamplingTransformer(
    input_dim=c['input_dim'], output_dim=c['output_dim'], 
    d_model=c['d_model'], nhead=c['nhead'], 
    num_layers=c['num_layers'], dim_feedforward=c['dim_feedforward'],
    dropout=c['dropout']
).to(device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# --- 4. BATCHED INFERENCE ---
print("Performing Inference...")
a_ss_norm = scaler_ss.transform(a_ss_test)
indices = np.arange(half_len, len(a_ss_test) - half_len)
a_ff_pred_norm = []

with torch.no_grad():
    for i in tqdm(range(0, len(indices), 512), desc="Predicting"):
        batch_idx = indices[i : i+512]
        x_batch = [a_ss_norm[idx - half_len : idx + half_len, :] for idx in batch_idx]
        pred = model(torch.FloatTensor(np.array(x_batch)).to(device))
        a_ff_pred_norm.append(pred.cpu().numpy())

a_ff_pred = scaler_ff.inverse_transform(np.vstack(a_ff_pred_norm))
a_ff_true = a_ff_test[indices]
time_aligned = test_time[indices]

# --- 5. PHYSICAL RECONSTRUCTION ---
print("Reconstructing Physical Fields...")
n_points = mean_ff.shape[0] // 2
Nx, Ny = len(x_coords), len(y_coords)
dx, dy = x_coords[1] - x_coords[0], y_coords[1] - y_coords[0]

def get_fields(a_coeffs):
    # Fluctuations + Mean
    ux = (mean_ff[:n_points] + a_coeffs @ phi_ff[:a_coeffs.shape[1], :n_points]).reshape(-1, Ny, Nx)
    uy = (mean_ff[n_points:] + a_coeffs @ phi_ff[:a_coeffs.shape[1], n_points:]).reshape(-1, Ny, Nx)
    # Vorticity: dv/dx - du/dy
    vort = np.gradient(uy, dx, axis=2) - np.gradient(ux, dy, axis=1)
    return ux, uy, vort

ux_p, uy_p, vort_p = get_fields(a_ff_pred)
ux_t, uy_t, vort_t = get_fields(a_ff_true)

# --- 6. STATISTICAL ERROR PLOTS ---
print("Generating Statistical Comparison Plots...")
# Calculate Statistics
u_mean_err = np.mean(ux_p, 0) - np.mean(ux_t, 0)
v_mean_err = np.mean(uy_p, 0) - np.mean(uy_t, 0)

up_p, vp_p = ux_p - np.mean(ux_p, 0), uy_p - np.mean(uy_p, 0)
up_t, vp_t = ux_t - np.mean(ux_t, 0), uy_t - np.mean(uy_t, 0)

uu_err = np.mean(up_p**2, 0) - np.mean(up_t**2, 0)
vv_err = np.mean(vp_p**2, 0) - np.mean(vp_t**2, 0)
uv_err = np.mean(up_p * vp_p, 0) - np.mean(up_t * vp_t, 0)

stats = {
    "u_mean": (u_mean_err, r"Mean $u$ Error"),
    "v_mean": (v_mean_err, r"Mean $v$ Error"),
    "uu_stress": (uu_err, r"$\overline{u'u'}$ Error"),
    "vv_stress": (vv_err, r"$\overline{v'v'}$ Error"),
    "uv_stress": (uv_err, r"$\overline{u'v'}$ Error")
}

extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
for name, (data, title) in stats.items():
    plt.figure(figsize=(10, 4))
    vlim = np.max(np.abs(data))
    im = plt.imshow(data, extent=extent, origin='lower', cmap='bwr', vmin=-vlim, vmax=vlim)
    plt.gca().add_patch(plt.Circle((0,0), 0.5, color='gray'))
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"stat_{name}.png"), dpi=200)
    plt.close()

# --- 7. VIDEO RENDERING (Manual FFmpeg Pipe) ---
print(f"Rendering Video: {video_out}")
v_max = np.percentile(np.abs(vort_t), 99)
error_map = (np.abs(vort_p - vort_t) / v_max) * 100
err_vmax = max(np.percentile(error_map, 99), 0.1)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 11), dpi=100)
ims = []

# Setup figure panels
for ax, title, cmap, vmin, vmax in zip([ax1, ax2, ax3], 
    [f'Transformer (T{MODEL_NUMBER}) Vorticity', 'Ground Truth DNS', 'Normalized Absolute Error (%)'],
    ['RdBu_r', 'RdBu_r', 'inferno'], [-v_max, -v_max, 0], [v_max, v_max, err_vmax]):
    
    ax.set_title(title)
    ax.add_patch(plt.Circle((0, 0), 0.5, color='gray', zorder=10))
    im = ax.imshow(np.zeros((Ny, Nx)), extent=extent, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, animated=True)
    plt.colorbar(im, ax=ax, pad=0.02, aspect=15)
    ims.append(im)

plt.tight_layout()
fig.canvas.draw()
w, h = fig.canvas.get_width_height()

ffmpeg_cmd = [
    ffmpeg_exe, '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{w}x{h}',
    '-pix_fmt', 'rgba', '-r', '30', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-preset', 'fast', '-crf', '18', video_out
]
process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

for t in tqdm(range(len(time_aligned)), desc="Encoding Video"):
    ims[0].set_array(vort_p[t])
    ims[1].set_array(vort_t[t])
    ims[2].set_array(error_map[t])
    fig.suptitle(f"Transformer Reconstruction | Time: {time_aligned[t]:.3f}s", fontsize=16)
    
    fig.canvas.draw()
    process.stdin.write(fig.canvas.buffer_rgba())

process.stdin.close()
process.wait()
plt.close(fig)

print(f"Reconstruction Complete. Video saved to {video_out}")