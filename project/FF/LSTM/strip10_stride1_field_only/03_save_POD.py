import os
import h5py
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

# --- 1. CONFIGURATION ---
base_dir = "/media/chris-remote/Projects/ONeill/estimation"
output_dir = os.path.join(base_dir, "output/kevin_2cylinder/")
unified_h5 = os.path.join(output_dir, "FF/flipflop_velocity_unified.h5")
# Output path remains the same to avoid script breakage
save_path = os.path.join(output_dir,"FF/LSTM/strip10_stride1_field_only", "ff_pod_master.h5")

os.makedirs(os.path.dirname(save_path), exist_ok=True)

N_MODES_FF = 700
SENSOR_X = 10.0
TOLERANCE = 0.1

# --- 2. LOAD UNIFIED DATA ---
print(f"Loading unified FF data from {unified_h5}...")
with h5py.File(unified_h5, 'r') as f:
    x = f['x'][:]
    y = f['y'][:]
    grid_shape = f['grid_shape'][:]
    
    ux_tr, uy_tr = f['train/ux'][:], f['train/uy'][:]
    ux_ts, uy_ts = f['test/ux'][:], f['test/uy'][:]

# Identify Sensor Indices (Far-wake strip at x=10)
sensor_mask = np.abs(x - SENSOR_X) < TOLERANCE
sensor_idx = np.where(sensor_mask)[0]
sensor_idx = sensor_idx[np.argsort(y[sensor_idx])]

print(f"Full Field Points: {ux_tr.shape[1]} | Sensor Points: {len(sensor_idx)}")

# The input_dim for your LSTM will be this value:
N_RAW_FEATURES = 2 * len(sensor_idx) 

# --- 3. FULL FIELD VECTOR POD (The Target) ---
print(f"Computing Full Field POD (N={N_MODES_FF})...")
S_tr = np.hstack([ux_tr, uy_tr])
S_ts = np.hstack([ux_ts, uy_ts])

mean_ff = np.mean(S_tr, axis=0)
pca_ff = PCA(n_components=N_MODES_FF, svd_solver='randomized', random_state=42)

# Fit on training fluctuations
a_ff_tr = pca_ff.fit_transform(S_tr - mean_ff)
# Project test fluctuations onto training basis
a_ff_ts = pca_ff.transform(S_ts - mean_ff)

# --- 4. RAW SENSOR EXTRACTION (The Input) ---
print(f"Extracting Raw Sensor Data (Features: {N_RAW_FEATURES})...")
# Stacking [u, v] for each sensor point
# Calling them a_ss to keep your LSTM training script compatible
a_ss_tr_raw = np.hstack([ux_tr[:, sensor_idx], uy_tr[:, sensor_idx]])
a_ss_ts_raw = np.hstack([ux_ts[:, sensor_idx], uy_ts[:, sensor_idx]])

# --- 5. SAVE MASTER H5 ---
print(f"Saving FF Master data to {save_path}...")
with h5py.File(save_path, 'w') as f:
    # Metadata for reconstruction
    f.create_dataset('grid_shape', data=grid_shape)
    f.create_dataset('sensor_idx', data=sensor_idx)
    
    # Full Field POD Basis & Mean
    f.create_dataset('phi_ff', data=pca_ff.components_.astype(np.float32))
    f.create_dataset('mean_ff', data=mean_ff.astype(np.float32))
    f.create_dataset('evr_ff', data=pca_ff.explained_variance_ratio_)
    
    # Placeholder keys for sensor metadata (to prevent loading errors)
    # We save zeros since we are bypassing the sensor POD
    f.create_dataset('phi_ss', data=np.zeros((1, N_RAW_FEATURES), dtype=np.float32))
    f.create_dataset('mean_ss', data=np.mean(a_ss_tr_raw, axis=0).astype(np.float32))
    f.create_dataset('evr_ss', data=np.zeros(1, dtype=np.float32))
    
    # Coefficients & Raw Inputs
    tr = f.create_group('train')
    tr.create_dataset('a_ff', data=a_ff_tr.astype(np.float32))
    tr.create_dataset('a_ss', data=a_ss_tr_raw.astype(np.float32)) # RAW DATA HERE
    
    ts = f.create_group('test')
    ts.create_dataset('a_ff', data=a_ff_ts.astype(np.float32))
    ts.create_dataset('a_ss', data=a_ss_ts_raw.astype(np.float32)) # RAW DATA HERE

print(f"Complete. Master file ready. Update your config['n_modes_ss'] to {N_RAW_FEATURES}.")