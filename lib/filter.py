import numpy as np
from scipy.signal import convolve2d
import warnings

#  filtering tools

def conv2d(signal,kernel):
    # handle the boundary conditions for discrete convolution between flow field and kernel
    # we assume that time is the third dimension
    filtered_signal = np.zeros(signal.shape)
    
    # handle the edge case of a single frame:
    if signal.ndim==2:
        filtered_signal = convolve2d(signal,kernel,'same','symm')
    else:
        for t_i in range(filtered_signal.shape[2]):
            filtered_signal[:,:,t_i] = convolve2d(signal[:,:,t_i],kernel,'same','symm')

    return filtered_signal




# gaussian kernels as per MST 2020

def gaussian_kernel_w(w,h_star):
    # w must be an integer
    if not np.isclose(w,np.int64(w)):
        raise ValueError('filter width w should be an integer')

    # compute standard deviation of the filter
    # w = 4c/h*+1; thus c = h*(w-1)/4
    c_star = h_star*(w-1)/4.0

    # compute the kernel elements
    kernel_x = h_star*np.arange(-(w-1)/2,(w-1)/2+1)
    kernel_y = h_star*np.arange(-(w-1)/2,(w-1)/2+1)
    kernel_X,kernel_Y = np.meshgrid(kernel_x,kernel_y)

    kernel = np.exp(-np.power(kernel_X,2.0)/(2*np.power(c_star,2.0))-np.power(kernel_Y,2.0)/(2*np.power(c_star,2.0)))
    # normalize so energy is preserved
    kernel = kernel/(np.sum(kernel))
    return kernel

def gaussian_kernel_c_star(c_star,h_star):
    # compute the size of the filter
    # w = 4c/h*+1;
    w = 4*c_star/h_star+1
    if not np.isclose(w,np.int64(w)):
        warnings.warn("c* and h* should be chosen so the filter width w is an integer. w: "+str(w)+" Continuing with next largest integer...")
    w = np.int64(np.ceil(w)) # force w to be an int
    # compute the kernel elements
    kernel_x = h_star*np.arange(-(w-1)/2,(w-1)/2+1)
    kernel_y = h_star*np.arange(-(w-1)/2,(w-1)/2+1)
    kernel_X,kernel_Y = np.meshgrid(kernel_x,kernel_y)

    kernel = np.exp(-np.power(kernel_X,2.0)/(2*np.power(c_star,2.0))-np.power(kernel_Y,2.0)/(2*np.power(c_star,2.0)))
    # normalize so energy is preserved
    kernel = kernel/(np.sum(kernel))
    return kernel

# 1d gaussian kernel

def gaussian_kernel_1d_w(w,h_star):
    # w must be an integer
    if not np.isclose(w,np.int64(w)):
        raise ValueError('filter width w should be an integer')

    # compute standard deviation of the filter
    # w = 4c/h*+1; thus c = h*(w-1)/4
    c_star = h_star*(w-1)/4.0

    # compute the kernel elements
    kernel_x = h_star*np.arange(-(w-1)/2,(w-1)/2+1)

    kernel = np.exp(-np.power(kernel_x,2.0)/(2*np.power(c_star,2.0)))
    # normalize so energy is preserved
    kernel = kernel/(np.sum(kernel))
    return kernel

## univeral outlier detection from westerweel2005universal

def UniversalOutlierDetection(U,V,threshold):
    """
    PIV outlier detection based on normalized median test.
    
    This function detects outliers in PIV (Particle Image Velocimetry) data
    by comparing each velocity vector to the median of its neighborhood.
    
    Parameters:
    -----------
    U : numpy.ndarray
        2D array of U-component (horizontal) velocity field
    V : numpy.ndarray  
        2D array of V-component (vertical) velocity field
    threshold : float
        Detection threshold (commonly 2)
    include_borders : bool, optional
        If True, extends detection to border pixels (default: True)
        
    Returns:
    --------
    info : numpy.ndarray
        2D boolean array where True indicates outlier detection
        Same size as input U and V arrays
    """
    
    # Get dimensions
    rows, cols = U.shape
    
    # Initialize arrays
    median_residual = np.zeros((rows, cols))
    normalised_fluctuation = np.zeros((rows, cols, 2))
    
    # Set neighborhood radius (commonly 1 or 2)
    R = 1
    # Set noise threshold level (e.g. 0.1 or eps*1)  
    eps = 0.1
    
    # Loop over velocity components (U and V)
    vel = [U, V]
    for i_c in range(len(vel)):
        # Loop over all data points
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                # Get neighborhood excluding center point
                neighborhood = []
                for di in range(-R, R+1):
                    for dj in range(-R, R+1):
                        if di == 0 and dj == 0:
                            continue  # Skip center point
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            neighborhood.append(vel[i_c][ni, nj])
                
                if len(neighborhood) == 0:
                    raise ValueError("neighbourhood should not have zero length")

                neighborhood = np.array(neighborhood)
                
                # Calculate median of neighborhood
                neigh_median = np.median(neighborhood)
                
                # Calculate fluctuation w.r.t. median
                fluctuation = vel[i_c][i, j] - neigh_median
                
                # Calculate residual (neighborhood fluctuation w.r.t. median)  
                residual = np.abs(neighborhood - neigh_median)
                
                # Calculate median of residual
                median_res = np.median(np.abs(residual))
                
                # Store values
                median_residual[i, j] = median_res
                
                # Calculate normalized fluctuation
                normalised_fluctuation[i, j, i_c] = np.abs(fluctuation / (median_res+eps))

    
    # Apply detection criterion
    info = np.power(np.power(normalised_fluctuation[:,:,0],2.0)+np.power(normalised_fluctuation[:,:,1],2.0),0.5) > threshold
    
    return info


import numpy as np

def UniversalOutlierReplacement(U, V, threshold):
    """
    PIV outlier detection based on normalized median test.
    
    This function detects outliers in PIV (Particle Image Velocimetry) data
    by comparing each velocity vector to the median of its neighborhood.
    
    Parameters:
    -----------
    U : numpy.ndarray
        2D or 3D array of U-component (horizontal) velocity field
        For 3D: shape (rows, cols, frames/time)
    V : numpy.ndarray  
        2D or 3D array of V-component (vertical) velocity field
        For 3D: shape (rows, cols, frames/time)
    threshold : float
        Detection threshold (commonly 2)
        
    Returns:
    --------
    U_temp : numpy.ndarray
        U-component with outliers replaced by neighborhood mean
        Same shape as input U
    V_temp : numpy.ndarray
        V-component with outliers replaced by neighborhood mean  
        Same shape as input V
    """
    
    # Check input dimensions
    if U.ndim not in [2, 3] or V.ndim not in [2, 3]:
        raise ValueError("U and V must be 2D or 3D arrays")
    
    if U.shape != V.shape:
        raise ValueError("U and V must have the same shape")
    
    # Handle 2D case by promoting to 3D temporarily
    was_2d = U.ndim == 2
    if was_2d:
        U = U[:, :, np.newaxis]
        V = V[:, :, np.newaxis]
    
    # Get dimensions
    rows, cols, frames = U.shape
    
    # Initialize output arrays
    U_temp = np.copy(U)
    V_temp = np.copy(V)
    
    # Set neighborhood radius and noise threshold
    R = 1
    eps = 0.1
    
    # Process each frame independently
    for frame in range(frames):
        # Get current frame
        U_frame = U[:, :, frame]
        V_frame = V[:, :, frame]
        
        # Initialize arrays for current frame
        median_residual = np.zeros((rows, cols))
        normalised_fluctuation = np.zeros((rows, cols, 2))
        neighborhood_mean = np.zeros((rows, cols, 2))
        
        # Loop over velocity components (U and V)
        vel = [U_frame, V_frame]
        for i_c in range(len(vel)):
            # Loop over all data points (excluding borders)
            for i in range(1, rows-1):
                for j in range(1, cols-1):
                    # Get neighborhood excluding center point
                    neighborhood = []
                    for di in range(-R, R+1):
                        for dj in range(-R, R+1):
                            if di == 0 and dj == 0:
                                continue  # Skip center point
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols:
                                neighborhood.append(vel[i_c][ni, nj])
                    
                    if len(neighborhood) == 0:
                        raise ValueError("neighbourhood should not have zero length")

                    neighborhood = np.array(neighborhood)
                    
                    # Calculate median of neighborhood
                    neigh_median = np.median(neighborhood)
                    neighborhood_mean[i, j, i_c] = np.mean(neighborhood)
                    
                    # Calculate fluctuation w.r.t. median
                    fluctuation = vel[i_c][i, j] - neigh_median
                    
                    # Calculate residual (neighborhood fluctuation w.r.t. median)  
                    residual = np.abs(neighborhood - neigh_median)
                    
                    # Calculate median of residual
                    median_res = np.median(np.abs(residual))
                    
                    # Store values
                    median_residual[i, j] = median_res
                    
                    # Calculate normalized fluctuation
                    normalised_fluctuation[i, j, i_c] = np.abs(fluctuation / (median_res + eps))

        # Apply detection criterion
        info = np.power(np.power(normalised_fluctuation[:, :, 0], 2.0) + 
                       np.power(normalised_fluctuation[:, :, 1], 2.0), 0.5) > threshold
        
        # Replace outliers with neighborhood mean
        U_temp[info, frame] = neighborhood_mean[info, 0]
        V_temp[info, frame] = neighborhood_mean[info, 1]
    
    # If input was 2D, reduce back to 2D
    if was_2d:
        U_temp = U_temp[:, :, 0]
        V_temp = V_temp[:, :, 0]
    
    return U_temp, V_temp

def UniversalOutlierReplacementVectorized(U, V, threshold):
    """
    Improved vectorized version that correctly implements the original algorithm.
    
    The original algorithm calculates a single median_residual per spatial location
    by processing both U and V components together. This version replicates that
    behavior more accurately.
    """
    
    # Check input dimensions
    if U.ndim not in [2, 3] or V.ndim not in [2, 3]:
        raise ValueError("U and V must be 2D or 3D arrays")
    
    if U.shape != V.shape:
        raise ValueError("U and V must have the same shape")
    
    # Handle 2D case by promoting to 3D temporarily
    was_2d = U.ndim == 2
    if was_2d:
        U = U[:, :, np.newaxis]
        V = V[:, :, np.newaxis]
    
    # Get dimensions
    rows, cols, frames = U.shape
    
    # Initialize output arrays
    U_temp = np.copy(U)
    V_temp = np.copy(V)
    
    eps = 0.1
    
    # Define 3x3 neighborhood offsets (excluding center)
    offsets = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            offsets.append((di, dj))
    
    # Process each frame independently
    for frame in range(frames):
        # Get current frame
        U_frame = U[:, :, frame]
        V_frame = V[:, :, frame]
        
        # Work on interior points only
        inner_rows, inner_cols = rows - 2, cols - 2
        
        # Initialize arrays to store results for interior points
        median_residual = np.zeros((inner_rows, inner_cols))
        U_norm_fluct = np.zeros((inner_rows, inner_cols))
        V_norm_fluct = np.zeros((inner_rows, inner_cols))
        U_mean_vals = np.zeros((inner_rows, inner_cols))
        V_mean_vals = np.zeros((inner_rows, inner_cols))
        
        # Process U and V components
        for i_c, vel_frame in enumerate([U_frame, V_frame]):
            # Create neighborhood array
            neighborhoods = np.zeros((inner_rows, inner_cols, 8))
            
            # Extract all neighborhoods
            for k, (di, dj) in enumerate(offsets):
                neighborhoods[:, :, k] = vel_frame[1+di:rows-1+di, 1+dj:cols-1+dj]
            
            # Get center values
            center_vals = vel_frame[1:rows-1, 1:cols-1]
            
            # Calculate neighborhood medians and means
            neigh_median = np.median(neighborhoods, axis=2)
            neigh_mean = np.mean(neighborhoods, axis=2)
            
            # Store means for replacement
            if i_c == 0:
                U_mean_vals = neigh_mean
            else:
                V_mean_vals = neigh_mean
            
            # Calculate fluctuation w.r.t. median
            fluctuation = center_vals - neigh_median
            
            # Calculate residuals
            residuals = np.abs(neighborhoods - neigh_median[:, :, np.newaxis])
            median_res = np.median(residuals, axis=2)
            
            # Store median residual (will be overwritten for each component, 
            # matching original behavior)
            median_residual = median_res
            
            # Calculate normalized fluctuation
            norm_fluct = np.abs(fluctuation) / (median_res + eps)
            
            if i_c == 0:
                U_norm_fluct = norm_fluct
            else:
                V_norm_fluct = norm_fluct
        
        # Apply detection criterion
        magnitude = np.sqrt(U_norm_fluct**2 + V_norm_fluct**2)
        outlier_mask = magnitude > threshold
        
        # Replace outliers
        U_temp[1:rows-1, 1:cols-1, frame][outlier_mask] = U_mean_vals[outlier_mask]
        V_temp[1:rows-1, 1:cols-1, frame][outlier_mask] = V_mean_vals[outlier_mask]
    
    # If input was 2D, reduce back to 2D
    if was_2d:
        U_temp = U_temp[:, :, 0]
        V_temp = V_temp[:, :, 0]
    
    return U_temp, V_temp
