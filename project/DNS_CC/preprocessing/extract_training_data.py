import os
import platform
import h5py
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
from dotenv import load_dotenv

# 1. Environment Setup
load_dotenv()
current_os = platform.system()
base_dir = os.getenv("BASE_DIR_WIN") if current_os == "Windows" else os.getenv("BASE_DIR_LIN")

if base_dir is None:
    raise ValueError(f"Base directory for {current_os} not found in environment variables.")

data_root = os.path.join(base_dir, "data/DNS_CC_Re150_Mazi/")
spectral_h5 = os.path.join(data_root, "dns_spectral_model_unified.h5")
regular_h5 = os.path.join(data_root, "dns_regular_0.02D_2d.h5")
output_h5 = os.path.join(data_root, "lstm_ready_data.h5")

# 2. Setup Interpolation (Irregular CFD -> Regular 0.02D Grid)
print("Loading coordinates and building KD-Tree...")
with h5py.File(spectral_h5, 'r') as f_spec, h5py.File(regular_h5, 'r') as f_reg:
    xy_irreg = f_spec['xy'][:]  # Original CFD Mesh (e.g., 175,077 points)
    
    # Regular Grid Setup
    x_reg_coords = f_reg['x'][:]
    y_reg_coords = f_reg['y'][:]
    X_reg, Y_reg = np.meshgrid(x_reg_coords, y_reg_coords)
    pts_reg = np.column_stack((X_reg.ravel(), Y_reg.ravel())) # Target (120,801 points)
    
    # Load Spectral Data
    freqs = f_spec['unified_freqs'][:]
    ux_c_irreg = f_spec['ux/coeffs'][:] # (N_modes, N_points_irreg)
    uy_c_irreg = f_spec['uy/coeffs'][:]
    grads_c_raw = f_spec['grads/coeffs'][:] # (N_modes, N_points_irreg * 4)
    
    # Reshape Gradients to (N_modes, N_points_irreg, 4)
    N_modes, N_pts_irreg = ux_c_irreg.shape
    grads_c_irreg = grads_c_raw.reshape(N_modes, N_pts_irreg, 4)
    
    # Time parameters
    t_orig = f_reg['time'][:]
    dt = np.mean(np.diff(t_orig))
    print(f"Detected dt: {dt:.6f} s")

# Build KD-Tree for spatial interpolation
tree = cKDTree(xy_irreg)
k_neighbors = 3
distances, indices = tree.query(pts_reg, k=k_neighbors)
weights = 1.0 / (distances**2 + 1e-12)
weights /= np.sum(weights, axis=1, keepdims=True)

# Pre-calculate Taylor deltas
delta_x = pts_reg[:, 0, np.newaxis] - xy_irreg[indices, 0]
delta_y = pts_reg[:, 1, np.newaxis] - xy_irreg[indices, 1]

# 3. Interpolation Function
def interpolate_spectral_coeffs(c_irreg, gx_irreg, gy_irreg):
    """Interpolates complex coefficients using Taylor-expanded IDW"""
    N_modes = c_irreg.shape[0]
    N_target = pts_reg.shape[0]
    c_reg = np.zeros((N_modes, N_target), dtype=np.complex128)
    
    for k in range(N_modes):
        v_n = c_irreg[k, indices]
        gx_n = gx_irreg[k, indices]
        gy_n = gy_irreg[k, indices]
        
        # Taylor: C_target = C_i + dC/dx * dx + dC/dy * dy
        v_taylor = v_n + delta_x * gx_n + delta_y * gy_n
        c_reg[k, :] = np.sum(weights * v_taylor, axis=1)
    return c_reg

print("Interpolating spectral coefficients to regular grid...")
ux_c = interpolate_spectral_coeffs(ux_c_irreg, grads_c_irreg[..., 0], grads_c_irreg[..., 1])
uy_c = interpolate_spectral_coeffs(uy_c_irreg, grads_c_irreg[..., 2], grads_c_irreg[..., 3])

# 4. Sensor Strip Identification (on 120,801 grid)
target_x_1 = x_reg_coords[np.argmin(np.abs(x_reg_coords - 6.0))]
target_idx_x = np.where(x_reg_coords == target_x_1)[0][0]
strip_x_indices = [target_idx_x, target_idx_x + 1]

# Map to flattened indices
Ny, Nx = X_reg.shape
strip_indices = []
for ix in strip_x_indices:
    # Adding vertical columns based on meshgrid flattening (row-major)
    strip_indices.extend(np.arange(ix, Ny * Nx, Nx))
strip_indices = np.sort(np.array(strip_indices))

# 5. Reconstruction & POD
def reconstruct(c_ux, c_uy, times):
    # Field(t) = C_0 + sum( 2 * Re( C_k * exp(i * 2pi * f_k * t) ) )
    exp_term = np.exp(1j * 2 * np.pi * np.outer(times, freqs[1:]))
    u = np.real(c_ux[0]) + 2 * np.real(exp_term @ c_ux[1:])
    v = np.real(c_uy[0]) + 2 * np.real(exp_term @ c_uy[1:])
    return u, v

print("Generating snapshots for POD basis...")
t_pod = np.random.rand(500) * 100.0
u_pod, v_pod = reconstruct(ux_c, uy_c, t_pod)

# Full Field POD
X_ff = np.hstack([u_pod - np.mean(u_pod, axis=0), v_pod - np.mean(v_pod, axis=0)])
U, S, Vh = np.linalg.svd(X_ff, full_matrices=False)
n_ff = np.argmax(np.cumsum(S**2)/np.sum(S**2) > 0.999) + 1
phi_ff = Vh[:n_ff, :]
mean_ff = np.mean(np.hstack([u_pod, v_pod]), axis=0)

# Strip POD
u_s_pod, v_s_pod = u_pod[:, strip_indices], v_pod[:, strip_indices]
X_ss = np.hstack([u_s_pod - np.mean(u_s_pod, axis=0), v_s_pod - np.mean(v_s_pod, axis=0)])
U_s, S_s, Vh_s = np.linalg.svd(X_ss, full_matrices=False)
n_ss = max(n_ff, np.argmax(np.cumsum(S_s**2)/np.sum(S_s**2) > 0.999) + 1)
phi_ss = Vh_s[:n_ss, :]
mean_ss = np.mean(np.hstack([u_s_pod, v_s_pod]), axis=0)

# 6. Generate Training Series
N_train = 100000
t_train = np.arange(N_train) * dt
a_ff_train = np.zeros((N_train, n_ff))
a_ss_train = np.zeros((N_train, n_ss))

print(f"Projecting {N_train} steps into POD space...")
chunk_size = 5000
for i in tqdm(range(0, N_train, chunk_size)):
    end = min(i + chunk_size, N_train)
    u, v = reconstruct(ux_c, uy_c, t_train[i:end])
    a_ff_train[i:end] = (np.hstack([u, v]) - mean_ff) @ phi_ff.T
    a_ss_train[i:end] = (np.hstack([u[:, strip_indices], v[:, strip_indices]]) - mean_ss) @ phi_ss.T

# 7. Project Original Test Data
print("Projecting original test data...")
with h5py.File(regular_h5, 'r') as f_orig:
    u_orig = f_orig['ux'][:].reshape(len(t_orig), -1)
    v_orig = f_orig['uy'][:].reshape(len(t_orig), -1)

X_test = np.hstack([u_orig, v_orig]) - mean_ff
a_ff_test = X_test @ phi_ff.T
X_s_test = np.hstack([u_orig[:, strip_indices], v_orig[:, strip_indices]]) - mean_ss
a_ss_test = X_s_test @ phi_ss.T

# 8. Final Save
with h5py.File(output_h5, 'w') as f:
    f.create_dataset('phi_ff', data=phi_ff)
    f.create_dataset('phi_ss', data=phi_ss)
    f.create_dataset('mean_ff', data=mean_ff)
    f.create_dataset('mean_ss', data=mean_ss)
    f.create_dataset('strip_indices', data=strip_indices)
    f.create_dataset('dt', data=dt)
    
    tr = f.create_group('train')
    tr.create_dataset('a_ff', data=a_ff_train)
    tr.create_dataset('a_ss', data=a_ss_train)
    
    ts = f.create_group('test')
    ts.create_dataset('a_ff', data=a_ff_test)
    ts.create_dataset('a_ss', data=a_ss_test)
    ts.create_dataset('time', data=t_orig)

print(f"Extraction complete. Training shape: {a_ff_train.shape}, Test shape: {a_ff_test.shape}")