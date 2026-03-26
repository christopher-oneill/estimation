import os
import h5py
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

# --- 1. CONFIGURATION ---
base_dir = "/media/chris-remote/Projects/ONeill/estimation"
output_dir = os.path.join(base_dir, "output/kevin_2cylinder/MP/strip5_stride1")
unified_h5 = os.path.join(output_dir, "MP_velocity_unified.h5")
save_path = os.path.join(output_dir, "mp_pod_master_500.h5")

N_MODES = 484
SENSOR_X = 5.0
TOLERANCE = 0.1

# --- 2. LOAD UNIFIED DATA ---
print(f"Loading unified data from {unified_h5}...")
with h5py.File(unified_h5, 'r') as f:
    x = f['x'][:]
    y = f['y'][:]
    grid_shape = f['grid_shape'][:]
    XI, YI = f['XI'][:], f['YI'][:]
    
    ux_tr, uy_tr = f['train/ux'][:], f['train/uy'][:]
    ux_ts, uy_ts = f['test/ux'][:], f['test/uy'][:]

# Find Sensor Indices
sensor_idx = np.where(np.abs(x - SENSOR_X) < TOLERANCE)[0]
sensor_idx = sensor_idx[np.argsort(y[sensor_idx])]

print(f"Full Field Points: {ux_tr.shape[1]} | Sensor Points: {len(sensor_idx)}")

# --- 3. FULL FIELD POD ---
print(f"Computing Full Field POD (N={N_MODES})...")
S_tr = np.hstack([ux_tr, uy_tr])
S_ts = np.hstack([ux_ts, uy_ts])

mean_ff = np.mean(S_tr, axis=0)
pca_ff = PCA(n_components=N_MODES, svd_solver='randomized', random_state=42)

# Fit on training data
a_ff_tr = pca_ff.fit_transform(S_tr - mean_ff)
# Project testing data onto training basis
a_ff_ts = pca_ff.transform(S_ts - mean_ff)

# --- 4. SENSOR STRIP POD ---
print(f"Computing Sensor Strip POD (N={N_MODES})...")
# Extract strip data from the full field
Ss_tr = np.hstack([ux_tr[:, sensor_idx], uy_tr[:, sensor_idx]])
Ss_ts = np.hstack([ux_ts[:, sensor_idx], uy_ts[:, sensor_idx]])

mean_ss = np.mean(Ss_tr, axis=0)
pca_ss = PCA(n_components=N_MODES, svd_solver='randomized', random_state=42)

# Fit on training data
a_ss_tr = pca_ss.fit_transform(Ss_tr - mean_ss)
# Project testing data
a_ss_ts = pca_ss.transform(Ss_ts - mean_ss)

# --- 5. SAVE MASTER H5 ---
print(f"Saving Master POD data to {save_path}...")
with h5py.File(save_path, 'w') as f:
    # Metadata for reconstruction
    f.create_dataset('grid_shape', data=grid_shape)
    f.create_dataset('XI', data=XI)
    f.create_dataset('YI', data=YI)
    f.create_dataset('sensor_idx', data=sensor_idx)
    
    # Full Field Basis & Means
    f.create_dataset('phi_ff', data=pca_ff.components_.astype(np.float32))
    f.create_dataset('mean_ff', data=mean_ff.astype(np.float32))
    f.create_dataset('evr_ff', data=pca_ff.explained_variance_ratio_)
    
    # Sensor Strip Basis & Means
    f.create_dataset('phi_ss', data=pca_ss.components_.astype(np.float32))
    f.create_dataset('mean_ss', data=mean_ss.astype(np.float32))
    f.create_dataset('evr_ss', data=pca_ss.explained_variance_ratio_)
    
    # Coefficients
    tr = f.create_group('train')
    tr.create_dataset('a_ff', data=a_ff_tr.astype(np.float32))
    tr.create_dataset('a_ss', data=a_ss_tr.astype(np.float32))
    
    ts = f.create_group('test')
    ts.create_dataset('a_ff', data=a_ff_ts.astype(np.float32))
    ts.create_dataset('a_ss', data=a_ss_ts.astype(np.float32))

print("Processing complete. You can now load any number of modes (1-500) from this file.")