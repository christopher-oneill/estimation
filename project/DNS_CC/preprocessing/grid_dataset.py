import os
from dotenv import load_dotenv
import numpy as np
import h5py
from scipy.spatial import cKDTree
from tqdm import tqdm

# 1. Environment Setup
load_dotenv()
base_dir = os.getenv("BASE_DIR")
data_dir = os.path.join(base_dir, "data/DNS_CC_Re150_Mazi/")

import sys
sys.path.append(base_dir) 

irregular_dataset = os.path.join(data_dir, "dns_complete_2d.h5")
irregular_dataset_file = h5py.File(irregular_dataset, 'r')

# 2. Extract Data
xy_irreg = irregular_dataset_file['xy'][:]  # Shape: (N_points, 2)
times = irregular_dataset_file['time'][:]
ux_irreg = irregular_dataset_file['ux'][:]
uy_irreg = irregular_dataset_file['uy'][:]
p_irreg = irregular_dataset_file['p'][:]

# Note: Adjust the slicing below based on the exact structure of your 'grads' array.
# Assuming grads is structured to hold [dx, dy] for velocity variables only now.
grads = irregular_dataset_file['grads'][:] 
grad_ux_x = grads[:, :, 0] # Example slice: adjust to your array's schema
grad_ux_y = grads[:, :, 1]
grad_uy_x = grads[:, :, 2]
grad_uy_y = grads[:, :, 3]

# 3. Define the Regular Grid
dx = 0.02
x_reg = np.arange(-2, 10 + dx, dx)
y_reg = np.arange(-2, 2 + dx, dx)
X_reg, Y_reg = np.meshgrid(x_reg, y_reg)

# Flatten target grid for vectorized KD-Tree querying
pts_reg = np.column_stack((X_reg.ravel(), Y_reg.ravel()))
N_target = pts_reg.shape[0]
N_time = len(times)

# 4. Build KD-Tree and compute weights
print("Building KD-Tree and querying neighbors...")
tree = cKDTree(xy_irreg)
k_neighbors = 3  # Using 3 nearest neighbors for the weighted average
distances, indices = tree.query(pts_reg, k=k_neighbors)

# Inverse distance weighting (IDW)
# Add small epsilon to prevent division by zero
weights = 1.0 / (distances**2 + 1e-12)
weights /= np.sum(weights, axis=1, keepdims=True)

# Pre-calculate spatial deltas between target points and their nearest neighbors
delta_x = pts_reg[:, 0, np.newaxis] - xy_irreg[indices, 0] # Shape: (N_target, k)
delta_y = pts_reg[:, 1, np.newaxis] - xy_irreg[indices, 1] # Shape: (N_target, k)

# 5. Initialize Output Arrays
ux_reg = np.zeros((N_time, N_target))
uy_reg = np.zeros((N_time, N_target))
p_reg = np.zeros((N_time, N_target))

# 6. Perform Interpolation Over Time
print("Interpolating fields...")
for t in tqdm(range(N_time), desc="Processing time steps", unit="step"):
    # Helper function for Taylor expansion (Velocity only)
    def taylor_interp(val_irreg, grad_x, grad_y):
        v_n = val_irreg[t, indices]       # (N_target, k)
        gx_n = grad_x[t, indices]         # (N_target, k)
        gy_n = grad_y[t, indices]         # (N_target, k)
        
        # u_target = u_i + dx * (du/dx)_i + dy * (du/dy)_i
        v_taylor = v_n + delta_x * gx_n + delta_y * gy_n
        
        return np.sum(weights * v_taylor, axis=1)

    # Apply Taylor-expanded IDW to velocities
    ux_reg[t, :] = taylor_interp(ux_irreg, grad_ux_x, grad_ux_y)
    uy_reg[t, :] = taylor_interp(uy_irreg, grad_uy_x, grad_uy_y)
    
    # Apply standard IDW to pressure (no gradients)
    p_n = p_irreg[t, indices]
    p_reg[t, :] = np.sum(weights * p_n, axis=1)

# Reshape back to 2D grid dimensions (N_time, Ny, Nx)
Ny, Nx = X_reg.shape
ux_reg = ux_reg.reshape((N_time, Ny, Nx))
uy_reg = uy_reg.reshape((N_time, Ny, Nx))
p_reg = p_reg.reshape((N_time, Ny, Nx))

irregular_dataset_file.close()

# 7. Save to new HDF5 file
output_dataset = os.path.join(data_dir, "dns_regular_0.02D_2d.h5")
print(f"Saving interpolated data to {output_dataset}...")

with h5py.File(output_dataset, 'w') as f:
    f.create_dataset('x', data=x_reg)
    f.create_dataset('y', data=y_reg)
    f.create_dataset('time', data=times)
    
    # Saving fields with chunking for efficient batch loading during neural network training
    f.create_dataset('ux', data=ux_reg, chunks=(1, Ny, Nx))
    f.create_dataset('uy', data=uy_reg, chunks=(1, Ny, Nx))
    f.create_dataset('p', data=p_reg, chunks=(1, Ny, Nx))

print("Done.")