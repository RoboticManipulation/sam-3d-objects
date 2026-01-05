# Copyright (c) Meta Platforms, Inc. and affiliates.
import numpy as np
import torch
from PIL import Image
import utils3d
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import Transform3d


def load_depth_image(depth_path, scale_factor=1000.0):
    """
    Load depth image and convert to meters.

    Args:
        depth_path: path to depth image file
        scale_factor: factor to divide depth by (default 1000 for mm to meters)

    Returns:
        depth: numpy array of shape (H, W) with depth values in meters
    """
    depth = Image.open(depth_path)
    depth = np.array(depth).astype(np.float32)
    # YCB-Video depth is typically in millimeters, so divide by 1000 to get meters
    depth = depth / scale_factor
    return depth


def load_camera_intrinsics(K_path):
    """
    Load camera intrinsics from K.txt file.

    Args:
        K_path: path to camera intrinsics file (3x3 matrix)

    Returns:
        K: camera intrinsics matrix (3x3 numpy array)
    """
    K = np.loadtxt(K_path)
    return K


def depth_to_pointmap(depth, K, apply_pytorch3d_convention=True, min_depth=0.01, max_depth=10.0):
    """
    Convert depth image to pointmap using camera intrinsics.

    Args:
        depth: numpy array of shape (H, W) with depth values in meters
        K: camera intrinsics matrix (3x3)
        apply_pytorch3d_convention: if True, applies PyTorch3D camera convention transform
        min_depth: minimum valid depth in meters (default 0.01m = 1cm)
        max_depth: maximum valid depth in meters (default 10.0m)

    Returns:
        pointmap: torch tensor of shape (H, W, 3) with 3D points in camera space
    """
    # Convert depth to torch tensor
    depth_tensor = torch.from_numpy(depth).float()

    # Create mask for valid depth values
    valid_depth_mask = (depth_tensor > min_depth) & (depth_tensor < max_depth)

    # Convert K to torch tensor
    K_tensor = torch.from_numpy(K).float()

    # Extract focal lengths and principal point from intrinsics
    fx = K_tensor[0, 0]
    fy = K_tensor[1, 1]
    cx = K_tensor[0, 2]
    cy = K_tensor[1, 2]

    # Normalize to [0, 1] range for utils3d
    H, W = depth.shape
    fx_normalized = fx / W
    fy_normalized = fy / H
    cx_normalized = cx / W
    cy_normalized = cy / H

    # Create normalized intrinsics for utils3d
    intrinsics = utils3d.torch.intrinsics_from_focal_center(
        fx_normalized, fy_normalized, cx_normalized, cy_normalized
    )

    # Convert depth to 3D points
    points = utils3d.torch.depth_to_points(depth_tensor, intrinsics=intrinsics)

    # Set invalid depth points to inf so they're filtered out during processing
    points = torch.where(valid_depth_mask.unsqueeze(-1), points, torch.tensor(float('inf')))

    # Apply PyTorch3D camera convention transformation if requested
    if apply_pytorch3d_convention:
        # This transforms from standard camera space to PyTorch3D camera space
        # Standard: X-right, Y-down, Z-forward -> PyTorch3D: X-left, Y-up, Z-forward
        r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
            eye=np.array([[0, 0, -1]]),
            at=np.array([[0, 0, 0]]),
            up=np.array([[0, -1, 0]]),
            device=points.device,
        )
        camera_convention_transform = Transform3d().rotate(r3_to_p3d_R)
        # Flatten to (H*W, 3) for transform, then reshape back
        original_shape = points.shape
        points_flat = points.reshape(-1, 3)
        points_transformed = camera_convention_transform.transform_points(points_flat)
        points = points_transformed.reshape(original_shape)

    return points


def create_pointmap_from_data(depth_path, K_path, depth_scale_factor=1000.0,
                             min_depth=0.01, max_depth=10.0):
    """
    Convenience function to create pointmap from file paths.

    Args:
        depth_path: path to depth image
        K_path: path to camera intrinsics file
        depth_scale_factor: factor to divide depth by to get meters
                           (1000.0 for mm, 1.0 if already in meters)
        min_depth: minimum valid depth in meters (default 0.01m = 1cm)
        max_depth: maximum valid depth in meters (default 10.0m)

    Returns:
        pointmap: torch tensor of shape (H, W, 3)
    """
    depth = load_depth_image(depth_path, scale_factor=depth_scale_factor)
    K = load_camera_intrinsics(K_path)
    pointmap = depth_to_pointmap(depth, K, apply_pytorch3d_convention=True,
                                min_depth=min_depth, max_depth=max_depth)
    return pointmap


if __name__ == "__main__":
    # Example usage
    depth_path = "notebook/images/ref_views_4/ob_0000005/depth/0000000.png"
    K_path = "notebook/images/ref_views_4/ob_0000005/K.txt"

    # For YCB-Video dataset, depth is in millimeters, so use scale_factor=1000.0
    pointmap = create_pointmap_from_data(depth_path, K_path, depth_scale_factor=1000.0)

    print(f"Created pointmap with shape: {pointmap.shape}")
    print(f"Pointmap dtype: {pointmap.dtype}")
    print(f"Pointmap range: [{pointmap.min():.3f}, {pointmap.max():.3f}]")

    # Check for valid values
    valid_mask = torch.isfinite(pointmap).all(dim=-1)
    print(f"Valid points: {valid_mask.sum()} / {valid_mask.numel()}")
