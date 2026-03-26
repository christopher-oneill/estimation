import os
import h5py
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

# --- 1. CONFIGURATION ---
base_dir = "/media/chris-remote/Projects/ONeill/estimation"
output_dir = os.path.join(base_dir, "output/kevin_2cylinder/")
unified_h5 = os.path.join(output_dir, "FF/flipflop_velocity_unified.h5")
save_path = os.path.join(output_dir,"FF/LSTM/strip10_stride1", "ff_pod_master.h5")

os.makedirs(os.path.dirname(save_path), exist_ok=True)

N_MODES_FF = 700
# Sensor rank is limited to 242 points * 2 components = 484

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

# Identify Sensor Indices (Near-wake strip)
sensor_mask = np.abs(x - SENSOR_X) < TOLERANCE
sensor_idx = np.where(sensor_mask)[0]
sensor_idx = sensor_idx[np.argsort(y[sensor_idx])]

print(f"Full Field Points: {ux_tr.shape[1]} | Sensor Strip Points: {len(sensor_idx)}")

N_MODES_SS = 2*len(sensor_idx) # Sensor modes (u and v) 

# --- 3. FULL FIELD VECTOR POD ---
print(f"Computing Full Field POD (N={N_MODES_FF})...")
S_tr = np.hstack([ux_tr, uy_tr])
S_ts = np.hstack([ux_ts, uy_ts])

mean_ff = np.mean(S_tr, axis=0)
# Randomized solver is essential for the 20k+ snapshots in the FF case
pca_ff = PCA(n_components=N_MODES_FF, svd_solver='randomized', random_state=42)

# Fit on training fluctuations
a_ff_tr = pca_ff.fit_transform(S_tr - mean_ff)
# Project test fluctuations onto training basis
a_ff_ts = pca_ff.transform(S_ts - mean_ff)

# --- 4. SENSOR STRIP VECTOR POD ---
print(f"Computing Sensor Strip POD (N={N_MODES_SS})...")
# Extract only the sensor points (u and v)
Ss_tr = np.hstack([ux_tr[:, sensor_idx], uy_tr[:, sensor_idx]])
Ss_ts = np.hstack([ux_ts[:, sensor_idx], uy_ts[:, sensor_idx]])

mean_ss = np.mean(Ss_tr, axis=0)
pca_ss = PCA(n_components=N_MODES_SS, svd_solver='randomized', random_state=42)

# Fit on training sensor fluctuations
a_ss_tr = pca_ss.fit_transform(Ss_tr - mean_ss)
# Project test sensor data
a_ss_ts = pca_ss.transform(Ss_ts - mean_ss)

# --- 5. SAVE MASTER H5 ---
print(f"Saving FF Master POD data to {save_path}...")
with h5py.File(save_path, 'w') as f:
    # Metadata for plotting/reconstruction
    f.create_dataset('grid_shape', data=grid_shape)
    f.create_dataset('sensor_idx', data=sensor_idx)
    
    # Full Field Basis (Modes + Mean)
    f.create_dataset('phi_ff', data=pca_ff.components_.astype(np.float32))
    f.create_dataset('mean_ff', data=mean_ff.astype(np.float32))
    f.create_dataset('evr_ff', data=pca_ff.explained_variance_ratio_)
    
    # Sensor Strip Basis (Modes + Mean)
    f.create_dataset('phi_ss', data=pca_ss.components_.astype(np.float32))
    f.create_dataset('mean_ss', data=mean_ss.astype(np.float32))
    f.create_dataset('evr_ss', data=pca_ss.explained_variance_ratio_)
    
    # Coefficients (Saves both segments)
    tr = f.create_group('train')
    tr.create_dataset('a_ff', data=a_ff_tr.astype(np.float32))
    tr.create_dataset('a_ss', data=a_ss_tr.astype(np.float32))
    
    ts = f.create_group('test')
    ts.create_dataset('a_ff', data=a_ff_ts.astype(np.float32))
    ts.create_dataset('a_ss', data=a_ss_ts.astype(np.float32))

print("Complete. Master FF POD file is ready for LSTM/Transformer training.")