import os
import h5py
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

# --- 1. CONFIGURATION ---
base_dir = "/media/chris-remote/Projects/ONeill/estimation"

# Updated output directory for the "field only" comparison
output_dir = os.path.join(base_dir, "output/kevin_2cylinder/MP/strip10_stride1_field_only")
unified_h5 = os.path.join(base_dir, "output/kevin_2cylinder/MP", "MP_velocity_unified.h5")
save_path = os.path.join(output_dir, "mp_pod_master_raw_sensor.h5")

os.makedirs(output_dir, exist_ok=True)

SENSOR_X = 10.0
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

# Find Sensor Indices at x=10
sensor_idx = np.where(np.abs(x - SENSOR_X) < TOLERANCE)[0]
sensor_idx = sensor_idx[np.argsort(y[sensor_idx])]

print(f"Full Field Points: {ux_tr.shape[1]} | Sensor Points: {len(sensor_idx)}")

# We'll calculate a generous number of modes for the Field (Target)
N_MODES_FF = 500 

# --- 3. FULL FIELD POD (The Target) ---
print(f"Computing Full Field POD (N={N_MODES_FF})...")
S_tr = np.hstack([ux_tr, uy_tr])
S_ts = np.hstack([ux_ts, uy_ts])

mean_ff = np.mean(S_tr, axis=0)
pca_ff = PCA(n_components=N_MODES_FF, svd_solver='randomized', random_state=42)

a_ff_tr = pca_ff.fit_transform(S_tr - mean_ff)
a_ff_ts = pca_ff.transform(S_ts - mean_ff)

# --- 4. RAW SENSOR DATA PREPARATION (The Input) ---
print("Preparing Raw Sensor Snapshots (no POD)...")
# ss_raw is (Time, 2*len(sensor_idx))
Ss_tr_raw = np.hstack([ux_tr[:, sensor_idx], uy_tr[:, sensor_idx]])
Ss_ts_raw = np.hstack([ux_ts[:, sensor_idx], uy_ts[:, sensor_idx]])

# --- 5. SAVE MASTER H5 ---
print(f"Saving Master data to {save_path}...")
with h5py.File(save_path, 'w') as f:
    # Metadata
    f.create_dataset('grid_shape', data=grid_shape)
    f.create_dataset('XI', data=XI)
    f.create_dataset('YI', data=YI)
    f.create_dataset('sensor_idx', data=sensor_idx)
    
    # Full Field Basis & Means (Required for Reconstruction)
    f.create_dataset('phi_ff', data=pca_ff.components_.astype(np.float32))
    f.create_dataset('mean_ff', data=mean_ff.astype(np.float32))
    f.create_dataset('evr_ff', data=pca_ff.explained_variance_ratio_)
    
    # Coefficients & Raw Data
    tr = f.create_group('train')
    tr.create_dataset('a_ff', data=a_ff_tr.astype(np.float32))
    tr.create_dataset('s_raw', data=Ss_tr_raw.astype(np.float32))
    
    ts = f.create_group('test')
    ts.create_dataset('a_ff', data=a_ff_ts.astype(np.float32))
    ts.create_dataset('s_raw', data=Ss_ts_raw.astype(np.float32))

print(f"Complete. Sensor data stored as raw velocity at {len(sensor_idx)} points.")