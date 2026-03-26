import os
from dotenv import load_dotenv
import numpy as np
import h5py
import matplotlib.pyplot as plt
import subprocess
from tqdm import tqdm

import imageio_ffmpeg  # <-- Add this import

# Get the absolute path to the FFmpeg binary installed in your environment
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

# 1. Environment Setup
load_dotenv()
base_dir = os.getenv("BASE_DIR")
data_dir = os.path.join(base_dir, "data/DNS_CC_Re150_Mazi/")
input_dataset = os.path.join(data_dir, "dns_regular_0.02D_2d.h5")
output_video = os.path.join(data_dir, "vorticity_wake.mp4")

# 2. Load the Regular Grid Data
print("Loading data...")
with h5py.File(input_dataset, 'r') as f:
    x = f['x'][:]
    y = f['y'][:]
    ux = f['ux'][:]
    uy = f['uy'][:]
    times = f['time'][:]

N_time, Ny, Nx = ux.shape
dx = x[1] - x[0]
dy = y[1] - y[0]

# 3. Compute Vorticity Using Central Differences
# \omega_z = du_y/dx - du_x/dy
print("Computing vorticity field...")
duy_dx = np.gradient(uy, dx, axis=2)
dux_dy = np.gradient(ux, dy, axis=1)
vorticity = duy_dx - dux_dy

# Set color limits based on the 99th percentile to avoid washing out the wake
vmax = np.percentile(np.abs(vorticity), 99)

# 4. Set Up the Matplotlib Figure
fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

# Use imshow for maximum rendering speed on regular grids
extent = [x.min(), x.max(), y.min(), y.max()]
cax = ax.imshow(vorticity[0], extent=extent, origin='lower', 
                cmap='RdBu_r', vmin=-vmax, vmax=vmax, animated=True)

# Add the cylinder patch (Diameter = 1, Radius = 0.5)
cylinder = plt.Circle((0, 0), 0.5, color='gray', zorder=10)
ax.add_patch(cylinder)

ax.set_xlabel('x/D')
ax.set_ylabel('y/D')
title = ax.set_title(f'Vorticity Field | Time: {times[0]:.2f}')
fig.colorbar(cax, ax=ax, label='Vorticity')
plt.tight_layout()

# 5. Set Up the FFmpeg Subprocess
# Force a canvas draw to get the exact pixel dimensions
fig.canvas.draw()
width, height = fig.canvas.get_width_height()

# Update the FFmpeg command list to use the absolute path
ffmpeg_cmd = [
    ffmpeg_exe,            # <-- Use the dynamic path here instead of 'ffmpeg'
    '-y',                  # Overwrite existing file
    '-f', 'rawvideo',      # Input format
    '-vcodec', 'rawvideo',
    '-s', f'{width}x{height}', # Frame size
    '-pix_fmt', 'rgba',    # Matplotlib buffer format
    '-r', '30',            # Framerate (fps)
    '-i', '-',             # Read from stdin
    '-c:v', 'libx264',     # H.264 codec
    '-pix_fmt', 'yuv420p', # Standard pixel format for compatibility
    '-preset', 'fast',     # Encoding speed
    '-crf', '18',          # Quality (lower is better, 18 is visually lossless)
    output_video
]

print(f"Starting FFmpeg render pipeline to {output_video}...")
process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# 6. Render Loop
try:
    for t in tqdm(range(N_time), desc="Rendering frames", unit="frame"):
        # Update the data in the plot (fastest method)
        cax.set_array(vorticity[t])
        title.set_text(f'Vorticity Field | Time: {times[t]:.2f}')
        
        # Draw and write the buffer directly to FFmpeg
        fig.canvas.draw()
        process.stdin.write(fig.canvas.buffer_rgba())

except Exception as e:
    print(f"Rendering interrupted: {e}")

finally:
    # Safely close the pipeline
    process.stdin.close()
    process.wait()
    plt.close(fig)

print("Video rendering complete!")