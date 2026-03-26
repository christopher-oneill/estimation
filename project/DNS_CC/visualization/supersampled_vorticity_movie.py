import os
import platform
import h5py
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from tqdm import tqdm
import imageio_ffmpeg
from dotenv import load_dotenv

# 1. Environment & Pathing
load_dotenv()
current_os = platform.system()
base_dir = os.getenv("BASE_DIR_WIN") if current_os == "Windows" else os.getenv("BASE_DIR_LIN")
data_dir = os.path.join(base_dir, "data/DNS_CC_Re150_Mazi/")

results_h5 = os.path.join(data_dir, "inference_results.h5")
grid_h5 = os.path.join(data_dir, "dns_regular_0.02D_2d.h5")
output_video = os.path.join(data_dir, "reconstruction_vorticity_comparison.mp4")

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

# 2. Load Data
print("Loading results and grid data...")
with h5py.File(grid_h5, 'r') as f_grid:
    x_coords = f_grid['x'][:]
    y_coords = f_grid['y'][:]

with h5py.File(results_h5, 'r') as f_res:
    ux_pred = f_res['ux_pred'][:]
    uy_pred = f_res['uy_pred'][:]
    ux_true = f_res['ux_true'][:]
    uy_true = f_res['uy_true'][:]
    times = f_res['time'][:]

# Reshape flattened vectors back to (Time, Ny, Nx)
Nx, Ny = len(x_coords), len(y_coords)
ux_pred = ux_pred.reshape(-1, Ny, Nx)
uy_pred = uy_pred.reshape(-1, Ny, Nx)
ux_true = ux_true.reshape(-1, Ny, Nx)
uy_true = uy_true.reshape(-1, Ny, Nx)

dx = x_coords[1] - x_coords[0]
dy = y_coords[1] - y_coords[0]

# 3. Compute Vorticity for Both Fields
def get_vorticity(u, v):
    duy_dx = np.gradient(v, dx, axis=2)
    dux_dy = np.gradient(u, dy, axis=1)
    return duy_dx - dux_dy

print("Computing vorticity fields...")
vort_pred = get_vorticity(ux_pred, uy_pred)
vort_true = get_vorticity(ux_true, uy_true)

# Compute Normalized Absolute Error (%)
# Normalized by the peak vorticity in the true field
v_max_global = np.percentile(np.abs(vort_true), 99)
error_field = (np.abs(vort_pred - vort_true) / v_max_global) * 100


# Calculate a useful vmax for the error based on actual performance
# Using the 99th percentile across all space and time to ignore outliers
error_vmax = np.percentile(error_field, 99)

# If the error is truly microscopic (e.g., < 0.01%), 
# we'll force a tiny floor so the colorbar doesn't break.
error_vmax = max(error_vmax, 0.01) 

print(f"Adjusting Error Map limit to: 0 to {error_vmax:.4f}%")


# 4. Set Up Figure (3 Panels)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), dpi=120)
extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]

# Formatting helper
def format_ax(ax, title):
    ax.set_title(title)
    ax.set_aspect('equal')
    # Add cylinder
    ax.add_patch(plt.Circle((0, 0), 0.5, color='gray', zorder=10))
    return ax.imshow(np.zeros((Ny, Nx)), extent=extent, origin='lower', animated=True)

im1 = format_ax(ax1, 'LSTM-POD Estimated Vorticity')
im1.set_cmap('RdBu_r')
im1.set_clim(-v_max_global, v_max_global)

im2 = format_ax(ax2, 'True DNS Vorticity (Ground Truth)')
im2.set_cmap('RdBu_r')
im2.set_clim(-v_max_global, v_max_global)

im3 = format_ax(ax3, 'Normalized Absolute Error (%)')
im3.set_cmap('inferno')
im3.set_clim(0, error_vmax) # Dynamically adjusted range

fig.colorbar(im1, ax=ax1, label='Vorticity')
fig.colorbar(im2, ax=ax2, label='Vorticity')
fig.colorbar(im3, ax=ax3, label='Error %')

plt.tight_layout()
fig.canvas.draw()
width, height = fig.canvas.get_width_height()

# 5. FFmpeg Pipe setup
ffmpeg_cmd = [
    ffmpeg_exe, '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
    '-s', f'{width}x{height}', '-pix_fmt', 'rgba', '-r', '30',
    '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-preset', 'fast', '-crf', '18', output_video
]
process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# 6. Render Loop
print(f"Rendering {len(times)} frames to {output_video}...")
try:
    for t in tqdm(range(len(times))):
        im1.set_array(vort_pred[t])
        im2.set_array(vort_true[t])
        im3.set_array(error_field[t])
        
        fig.suptitle(f"Temporal Supersampling Reconstruction | Time: {times[t]:.3f}s", fontsize=16)
        
        fig.canvas.draw()
        process.stdin.write(fig.canvas.buffer_rgba())
except Exception as e:
    print(f"Error during render: {e}")
finally:
    process.stdin.close()
    process.wait()
    plt.close(fig)

print("Video generation complete.")