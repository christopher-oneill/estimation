import os
import h5py
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

# --- 1. Paths & Configuration ---
output_dir = "/media/chris-remote/Projects/ONeill/estimation/output/kevin_2cylinder"
unified_h5 = os.path.join(output_dir, "flipflop_velocity_unified.h5")
save_path = os.path.join(output_dir, "lstm_ready_data_FF.h5")

SENSOR_X = 12.0
TOLERANCE = 0.1  # Slightly wider to ensure we capture enough points at x=12
N_MODES_FF = 178 
N_MODES_SS = int(N_MODES_FF * 1.5) # 267 modes

# --- 2. Load and Align Grid ---
print("Loading data...")
with h5py.File(unified_h5, 'r') as f:
    # Ensure grid matches the actual data points saved in snapshots
    n_data_pts = f['train/ux'].shape[1]
    x = f['x'][:n_data_pts]
    y = f['y'][:n_data_pts]
    
    ux_train = f['train/ux'][:]
    uy_train = f['train/uy'][:]
    ux_test = f['test/ux'][:]
    uy_test = f['test/uy'][:]

# Find Sensor Indices at x/D = 12
# This now uses the truncated/aligned grid to prevent IndexError
sensor_mask = np.abs(x - SENSOR_X) < TOLERANCE
sensor_idx = np.where(sensor_mask)[0]
sensor_idx = sensor_idx[np.argsort(y[sensor_idx])] # Sorted vertically

print(f"Extracted sensor strip at x/D={SENSOR_X} with {len(sensor_idx)} points.")

# Safety check: Can't extract more modes than points
if len(sensor_idx) * 2 <= N_MODES_SS:
    print(f"Warning: Sensor strip too sparse ({len(sensor_idx)} pts). Reducing sensor modes.")
    N_MODES_SS = (len(sensor_idx) * 2) - 1

# --- 3. Full-Field POD ---
print(f"Computing Full Field POD for {N_MODES_FF} modes...")
S_train = np.hstack([ux_train, uy_train])
mean_ff = np.mean(S_train, axis=0)
pca_ff = PCA(n_components=N_MODES_FF, svd_solver='randomized', random_state=42)
a_ff_train = pca_ff.fit_transform(S_train - mean_ff)
phi_ff = pca_ff.components_

# --- 4. Sensor Strip POD ---
print(f"Computing Sensor Strip POD for {N_MODES_SS} modes...")
# Stack u and v for the sensor strip: (N_time, 2 * N_sensor_points)
ss_train = np.hstack([ux_train[:, sensor_idx], uy_train[:, sensor_idx]])
mean_ss = np.mean(ss_train, axis=0)
pca_ss = PCA(n_components=N_MODES_SS, svd_solver='randomized', random_state=42)
a_ss_train = pca_ss.fit_transform(ss_train - mean_ss)
phi_ss = pca_ss.components_

# --- 5. Project Test Data ---
print("Projecting test snapshots...")
S_test_prime = np.hstack([ux_test, uy_test]) - mean_ff
a_ff_test = pca_ff.transform(S_test_prime)

ss_test_prime = np.hstack([ux_test[:, sensor_idx], uy_test[:, sensor_idx]]) - mean_ss
a_ss_test = pca_ss.transform(ss_test_prime)

# --- 6. Save ---
print(f"Saving LSTM-ready data to {save_path}...")
with h5py.File(save_path, 'w') as f:
    f.create_dataset('phi_ff', data=phi_ff)
    f.create_dataset('mean_ff', data=mean_ff)
    f.create_dataset('phi_ss', data=phi_ss)
    f.create_dataset('mean_ss', data=mean_ss)
    f.create_dataset('sensor_idx', data=sensor_idx)
    
    tr = f.create_group('train')
    tr.create_dataset('a_ff', data=a_ff_train)
    tr.create_dataset('a_ss', data=a_ss_train)
    
    ts = f.create_group('test')
    ts.create_dataset('a_ff', data=a_ff_test)
    ts.create_dataset('a_ss', data=a_ss_test)

print("Done.")