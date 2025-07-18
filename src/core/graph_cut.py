# graph_cut.py
"""
Implementation of graph cut segmentation algorithms based on the GrabCut paper (Rother et al., 2004).
This module provides functions for segmenting images using graph cuts with GMM-based data terms.
"""

import numpy as np
import maxflow
from sklearn.mixture import GaussianMixture


def learn_gmm_with_component_assignment(conf_map, mask, n_components=5):
    """
    Learn GMM model and assign components to pixels according to GrabCut paper.

    Args:
        conf_map: Confidence map
        mask: Binary mask (1=foreground, 0=background)
        n_components: Number of GMM components

    Returns:
        gmm: Learned GMM model
        components: Component assignments for each pixel
    """
    # Extract pixels from the mask
    pixels = conf_map[mask > 0].reshape(-1, 1)

    # Ensure we have enough samples
    min_samples = max(n_components * 2, 10)
    if len(pixels) < min_samples:
        raise ValueError(f"Not enough samples: {len(pixels)} < {min_samples}")

    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(pixels)

    # Initialize component assignments
    h, w = conf_map.shape
    components = np.zeros((h, w), dtype=np.int32)

    # Assign components to pixels in the mask
    if np.any(mask > 0):
        # Get coordinates of pixels in the mask
        y_indices, x_indices = np.where(mask > 0)

        # Extract pixel values
        mask_pixels = conf_map[mask > 0].reshape(-1, 1)

        # Predict component for each pixel
        pixel_components = gmm.predict(mask_pixels)

        # Assign components
        for i, (y, x) in enumerate(zip(y_indices, x_indices)):
            components[y, x] = pixel_components[i]

    return gmm, components


def calculate_data_term_with_components(conf_map, fg_gmm, bg_gmm, fg_components, bg_components, fg_mask, bg_mask):
    """
    Calculate data term using assigned GMM components according to GrabCut paper.

    Args:
        conf_map: Confidence map
        fg_gmm: Foreground GMM model
        bg_gmm: Background GMM model
        fg_components: Component assignments for foreground pixels
        bg_components: Component assignments for background pixels
        fg_mask: Foreground mask
        bg_mask: Background mask

    Returns:
        fg_cost: Foreground cost (negative log probability)
        bg_cost: Background cost (negative log probability)
    """
    h, w = conf_map.shape

    # Initialize costs
    fg_cost = np.zeros((h, w), dtype=np.float32)
    bg_cost = np.zeros((h, w), dtype=np.float32)

    # Reshape for GMM calculation
    pixels = conf_map.reshape(-1, 1)

    # Calculate log probabilities for all components
    fg_log_probs = np.zeros((h*w, fg_gmm.n_components))
    bg_log_probs = np.zeros((h*w, bg_gmm.n_components))

    for k in range(fg_gmm.n_components):
        # Get component parameters
        weight_fg = fg_gmm.weights_[k]
        mean_fg = fg_gmm.means_[k]
        cov_fg = fg_gmm.covariances_[k]

        # Calculate log probability: log(π_k) + log(N(z; μ_k, Σ_k))
        log_det_fg = np.log(np.linalg.det(cov_fg))
        log_prob_fg = np.log(weight_fg) - 0.5 * (log_det_fg +
                      ((pixels - mean_fg) ** 2 / cov_fg).sum(axis=1) +
                      np.log(2 * np.pi))

        fg_log_probs[:, k] = log_prob_fg

    for k in range(bg_gmm.n_components):
        weight_bg = bg_gmm.weights_[k]
        mean_bg = bg_gmm.means_[k]
        cov_bg = bg_gmm.covariances_[k]

        log_det_bg = np.log(np.linalg.det(cov_bg))
        log_prob_bg = np.log(weight_bg) - 0.5 * (log_det_bg +
                      ((pixels - mean_bg) ** 2 / cov_bg).sum(axis=1) +
                      np.log(2 * np.pi))

        bg_log_probs[:, k] = log_prob_bg

    # Assign costs based on component assignments
    for y in range(h):
        for x in range(w):
            idx = y * w + x

            # Foreground pixels
            if fg_mask[y, x] > 0:
                k = fg_components[y, x]
                fg_cost[y, x] = -fg_log_probs[idx, k]
                # Use maximum probability for background
                bg_cost[y, x] = -np.max(bg_log_probs[idx])

            # Background pixels
            elif bg_mask[y, x] > 0:
                k = bg_components[y, x]
                bg_cost[y, x] = -bg_log_probs[idx, k]
                # Use maximum probability for foreground
                fg_cost[y, x] = -np.max(fg_log_probs[idx])

            # Uncertain pixels
            else:
                # Use maximum probability for both
                fg_cost[y, x] = -np.max(fg_log_probs[idx])
                bg_cost[y, x] = -np.max(bg_log_probs[idx])

    return fg_cost, bg_cost


# Smoothness term calculation functions
def calculate_smoothness_term(conf_map, lambda_param=100, connectivity=4, beta=30):
    """
    Calculate smoothness term for graph cut using the GrabCut paper's formulation.

    Args:
        conf_map: Confidence map
        lambda_param: Weight for smoothness term
        connectivity: Connectivity type (4 or 8)
        beta: Parameter controlling the sensitivity to intensity differences (default: 30)

    Returns:
        weights: Edge weights for graph cut
        structure: Connectivity structure for graph cut
    """
    h, w = conf_map.shape

    # Create arrays to store the weights for each direction
    if connectivity == 4:
        # 4-connectivity: right, left, down, up
        weights = np.zeros((h, w, 4), dtype=np.float32)

        # Calculate horizontal differences (right - left)
        diff_x = np.zeros((h, w-1), dtype=np.float32)
        for y in range(h):
            for x in range(w-1):
                diff = conf_map[y, x+1] - conf_map[y, x]
                diff_x[y, x] = diff * diff

        # Calculate vertical differences (down - up)
        diff_y = np.zeros((h-1, w), dtype=np.float32)
        for y in range(h-1):
            for x in range(w):
                diff = conf_map[y+1, x] - conf_map[y, x]
                diff_y[y, x] = diff * diff

        # Calculate weights using the formula: λ⋅exp(−β⋅(Ip-Iq)²)
        # Right direction
        weights[:, :-1, 0] = lambda_param * np.exp(-beta * diff_x)
        # Left direction
        weights[:, 1:, 1] = lambda_param * np.exp(-beta * diff_x)
        # Down direction
        weights[:-1, :, 2] = lambda_param * np.exp(-beta * diff_y)
        # Up direction
        weights[1:, :, 3] = lambda_param * np.exp(-beta * diff_y)

        # 4-connectivity structure
        structure = np.array([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]])

    elif connectivity == 8:
        # 8-connectivity: right, left, down, up, down-right, up-left, down-left, up-right
        weights = np.zeros((h, w, 8), dtype=np.float32)

        # Calculate horizontal differences (right - left)
        diff_x = np.zeros((h, w-1), dtype=np.float32)
        for y in range(h):
            for x in range(w-1):
                diff = conf_map[y, x+1] - conf_map[y, x]
                diff_x[y, x] = diff * diff

        # Calculate vertical differences (down - up)
        diff_y = np.zeros((h-1, w), dtype=np.float32)
        for y in range(h-1):
            for x in range(w):
                diff = conf_map[y+1, x] - conf_map[y, x]
                diff_y[y, x] = diff * diff

        # Calculate diagonal differences (down-right - up-left)
        diff_diag1 = np.zeros((h-1, w-1), dtype=np.float32)
        for y in range(h-1):
            for x in range(w-1):
                diff = conf_map[y+1, x+1] - conf_map[y, x]
                diff_diag1[y, x] = diff * diff

        # Calculate diagonal differences (down-left - up-right)
        diff_diag2 = np.zeros((h-1, w-1), dtype=np.float32)
        for y in range(h-1):
            for x in range(1, w):
                diff = conf_map[y+1, x-1] - conf_map[y, x]
                diff_diag2[y, x-1] = diff * diff

        # Calculate weights using the formula: λ⋅exp(−β⋅(Ip-Iq)²)
        # Right direction
        weights[:, :-1, 0] = lambda_param * np.exp(-beta * diff_x)
        # Left direction
        weights[:, 1:, 1] = lambda_param * np.exp(-beta * diff_x)
        # Down direction
        weights[:-1, :, 2] = lambda_param * np.exp(-beta * diff_y)
        # Up direction
        weights[1:, :, 3] = lambda_param * np.exp(-beta * diff_y)
        # Down-right direction
        weights[:-1, :-1, 4] = lambda_param * np.exp(-beta * diff_diag1)
        # Up-left direction
        weights[1:, 1:, 5] = lambda_param * np.exp(-beta * diff_diag1)
        # Down-left direction
        weights[:-1, 1:, 6] = lambda_param * np.exp(-beta * diff_diag2)
        # Up-right direction
        weights[1:, :-1, 7] = lambda_param * np.exp(-beta * diff_diag2)

        # 8-connectivity structure
        structure = np.array([[1, 1, 1],
                              [1, 0, 1],
                              [1, 1, 1]])

    else:
        raise ValueError(f"Invalid connectivity: {connectivity}. Must be 4 or 8.")

    return weights, structure


# Graph cut implementation functions
def run_graph_cut(conf_map, fg_cost, bg_cost, weights, fg_seed, bg_seed, structure=None):
    """
    Run graph cut optimization with fixed handling of hard and soft constraints.

    Args:
        conf_map: Confidence map
        fg_cost: Foreground cost
        bg_cost: Background cost
        weights: Edge weights
        fg_seed: Foreground seed mask
        bg_seed: Background seed mask
        structure: Connectivity structure (if None, 4-connectivity is used)

    Returns:
        binary_mask: Binary segmentation mask
    """
    h, w = conf_map.shape
    num_pixels = h * w

    # Create graph
    # For 8-connectivity, we need more edges
    max_edges = num_pixels * 8 if structure is not None and np.sum(structure) > 4 else num_pixels * 4
    g = maxflow.Graph[float](num_pixels, max_edges)
    node_ids = g.add_grid_nodes((h, w))

    # Use provided structure or default to 4-connectivity
    if structure is None:
        structure = np.array([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]])

    # Add edges between neighboring pixels
    # Get the number of directions (4 or 8)
    n_directions = weights.shape[2]

    # Create a 2D weights array for each direction
    for d in range(n_directions):
        # Extract weights for this direction
        direction_weights = weights[:, :, d]

        # Skip zeros (no connections in this direction for these pixels)
        mask = direction_weights > 0
        if not np.any(mask):
            continue

        # Create a structure with only this direction
        if d == 0:  # Right
            dir_structure = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
        elif d == 1:  # Left
            dir_structure = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        elif d == 2:  # Down
            dir_structure = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
        elif d == 3:  # Up
            dir_structure = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        elif d == 4:  # Down-right
            dir_structure = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        elif d == 5:  # Up-left
            dir_structure = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        elif d == 6:  # Down-left
            dir_structure = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
        elif d == 7:  # Up-right
            dir_structure = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])

        # Add edges for this direction
        g.add_grid_edges(node_ids, weights=direction_weights, structure=dir_structure, symmetric=False)

    # Combine hard constraints (seeds) and soft constraints (GMM probabilities)
    # Create modified cost maps that incorporate both
    fg_cost_modified = fg_cost.copy()
    bg_cost_modified = bg_cost.copy()

    # Apply hard constraints
    # Foreground seeds: set foreground cost to 0 and background cost to infinity
    fg_cost_modified[fg_seed > 0] = 0
    bg_cost_modified[fg_seed > 0] = 1e9

    # Background seeds: set background cost to 0 and foreground cost to infinity
    fg_cost_modified[bg_seed > 0] = 1e9
    bg_cost_modified[bg_seed > 0] = 0

    # Add terminal edges with combined costs (only once)
    g.add_grid_tedges(node_ids, fg_cost_modified, bg_cost_modified)

    # Run max-flow
    g.maxflow()

    # Get segmentation result
    segmentation = g.get_grid_segments(node_ids)

    # Convert to binary mask (foreground = 1)
    binary_mask = np.logical_not(segmentation).astype(np.uint8)

    return binary_mask


def iterative_graph_cut_grabcut(conf_map, fg_seed, bg_seed, lambda_param=100, n_iterations=3, n_components=5, connectivity=4, beta=30):
    """
    Run iterative graph cut segmentation following the GrabCut algorithm (Rother et al., 2004).

    This implementation follows the paper's energy minimization approach:
    1. Assign GMM components to pixels
    2. Learn GMM parameters
    3. Calculate data term based on assigned components
    4. Run graph cut to update segmentation
    5. Repeat until convergence

    Args:
        conf_map: Confidence map
        fg_seed: Foreground seed mask
        bg_seed: Background seed mask
        lambda_param: Weight for smoothness term
        n_iterations: Number of iterations
        n_components: Number of GMM components
        connectivity: Connectivity type (4 or 8)
        beta: Parameter controlling the sensitivity to intensity differences (default: 30)

    Returns:
        binary_mask: Binary segmentation mask
    """
    h, w = conf_map.shape

    # Initialize trimap
    # T_B: definite background (bg_seed)
    # T_F: definite foreground (fg_seed)
    # T_U: uncertain region (everything else)
    T_B = bg_seed.copy()
    T_F = fg_seed.copy()
    T_U = np.ones((h, w), dtype=np.uint8) - T_F - T_B

    # Initialize alpha (segmentation)
    alpha = np.zeros((h, w), dtype=np.uint8)
    alpha[T_F > 0] = 1  # definite foreground
    alpha[T_U > 0] = 1  # initially assume uncertain region is foreground

    # Initialize component assignments
    fg_components = np.zeros((h, w), dtype=np.int32)
    bg_components = np.zeros((h, w), dtype=np.int32)

    # Iterative optimization
    for i in range(n_iterations):
        # 1. Update foreground/background masks based on current alpha
        fg_mask = alpha > 0
        bg_mask = alpha == 0

        # 2. Learn GMM models and assign components
        fg_gmm, fg_components_new = learn_gmm_with_component_assignment(conf_map, fg_mask, n_components)
        bg_gmm, bg_components_new = learn_gmm_with_component_assignment(conf_map, bg_mask, n_components)

        # Update component assignments
        fg_components = fg_components_new
        bg_components = bg_components_new

        # 3. Calculate data term using component assignments
        fg_cost, bg_cost = calculate_data_term_with_components(
            conf_map, fg_gmm, bg_gmm, fg_components, bg_components, fg_mask, bg_mask)

        # 4. Calculate smoothness term
        weights, structure = calculate_smoothness_term(conf_map, lambda_param, connectivity, beta)

        # 5. Run graph cut
        binary_mask = run_graph_cut(conf_map, fg_cost, bg_cost, weights, T_F, T_B, structure)

        # 6. Update alpha (preserving definite foreground/background)
        alpha_new = binary_mask.copy()
        alpha_new[T_F > 0] = 1  # Keep definite foreground
        alpha_new[T_B > 0] = 0  # Keep definite background

        # Check for convergence
        if np.array_equal(alpha, alpha_new):
            print(f"Converged after {i+1} iterations")
            break

        alpha = alpha_new

    return alpha


# Threshold setting functions
def set_thresholds_manual(th1, th2):
    """
    Use user-specified threshold values.

    Args:
        th1: Lower threshold (values below this are considered background)
        th2: Upper threshold (values above this are considered foreground)

    Returns:
        th1: Lower threshold
        th2: Upper threshold
    """
    return th1, th2


def set_thresholds_without_gt_gmm(conf_map):
    """
    Set thresholds using GMM without ground truth.

    Args:
        conf_map: Confidence map

    Returns:
        th1: Lower threshold
        th2: Upper threshold
    """
    # Flatten and filter valid values
    conf_map_flat = conf_map.flatten()
    valid_mask = ~np.isnan(conf_map_flat)
    conf_map_valid = conf_map_flat[valid_mask]

    # Reshape for sklearn
    X = conf_map_valid.reshape(-1, 1)

    # Fit GMM with 3 components (background, uncertain, building)
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X)

    # Get the means of the 3 components
    means = gmm.means_.flatten()
    means.sort()

    # Use the means to determine thresholds
    th1 = (means[0] + means[1]) / 2  # between first and second component
    th2 = (means[1] + means[2]) / 2  # between second and third component

    return th1, th2


# Main segmentation function
def graph_cut_segmentation(conf_map, mode='gmm', n_iterations=3, n_components=5, lambda_param=100, connectivity=4, beta=30, **kwargs):
    """
    Run the complete graph cut segmentation pipeline.

    Args:
        conf_map: Confidence map
        mode: Threshold setting mode ('gmm' or 'manual')
        n_iterations: Number of iterations
        n_components: Number of GMM components
        lambda_param: Weight for smoothness term
        connectivity: Connectivity type (4 or 8)
        beta: Parameter controlling the sensitivity to intensity differences (default: 30)
        **kwargs: Additional parameters (th1 and th2 for manual mode)

    Returns:
        Binary segmentation mask
    """
    # 1. Set thresholds based on mode
    if mode == 'gmm':
        th1, th2 = set_thresholds_without_gt_gmm(conf_map)
    elif mode == 'manual':
        # Check for required parameters
        if 'th1' not in kwargs or 'th2' not in kwargs:
            raise ValueError("Manual mode requires th1 and th2 parameters")
        th1, th2 = set_thresholds_manual(kwargs['th1'], kwargs['th2'])
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # 2. Generate initial seeds
    fg_seed = (conf_map >= th2).astype(np.int32)
    bg_seed = (conf_map <= th1).astype(np.int32)

    # 3. Run iterative graph cut using GrabCut algorithm
    binary_mask = iterative_graph_cut_grabcut(
        conf_map,
        fg_seed,
        bg_seed,
        lambda_param=lambda_param,
        n_iterations=n_iterations,
        n_components=n_components,
        connectivity=connectivity,
        beta=beta
    )

    return binary_mask
