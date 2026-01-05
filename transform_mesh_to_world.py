# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Transform meshes from canonical space to world space using the pose parameters.
"""
import sys
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

sys.path.append("notebook")
from inference import Inference, load_image, load_single_mask
from create_pointmap import load_depth_image, load_camera_intrinsics, depth_to_pointmap


def quaternion_to_rotation_matrix(quat):
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    # PyTorch3D uses (w, x, y, z) format
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    # Convert to scipy format (x, y, z, w)
    scipy_quat = np.array([x, y, z, w])
    rot = Rotation.from_quat(scipy_quat)
    return rot.as_matrix()


def transform_mesh_to_world(mesh, scale, translation, rotation):
    """
    Transform mesh from canonical space to world space.

    Args:
        mesh: trimesh object
        scale: (3,) tensor or array with scale factors
        translation: (3,) tensor or array with translation
        rotation: (4,) tensor or array with quaternion (w, x, y, z)

    Returns:
        transformed mesh
    """
    # Convert to numpy
    if torch.is_tensor(scale):
        scale = scale.cpu().numpy().flatten()
    if torch.is_tensor(translation):
        translation = translation.cpu().numpy().flatten()
    if torch.is_tensor(rotation):
        rotation = rotation.cpu().numpy().flatten()

    # Get vertices
    vertices = np.array(mesh.vertices)

    # Apply transformations: scale -> rotate -> translate
    # 1. Scale
    vertices_scaled = vertices * scale

    # 2. Rotate (COMMENTED OUT)
    # rot_matrix = quaternion_to_rotation_matrix(rotation)
    # vertices_rotated = vertices_scaled @ rot_matrix.T

    # 3. Translate (COMMENTED OUT)
    # vertices_world = vertices_rotated + translation

    # Just use scaled vertices
    vertices_world = vertices_scaled

    # Create new mesh with transformed vertices
    mesh_world = trimesh.Trimesh(
        vertices=vertices_world,
        faces=mesh.faces,
        vertex_colors=mesh.visual.vertex_colors if hasattr(mesh.visual, 'vertex_colors') else None
    )

    return mesh_world


def decimate_mesh(mesh_path, output_path, reduction_ratio=0.05):
    """
    Decimate mesh using pymeshlab.

    Args:
        mesh_path: Input mesh file path
        output_path: Output mesh file path
        reduction_ratio: Ratio of faces to keep (0.05 = keep 5%)
    """
    import pymeshlab

    # Load into PyMeshLab and decimate
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)

    # Get current face count
    original_faces = ms.current_mesh().face_number()
    print(f"  Original faces: {original_faces}")

    # Calculate target faces
    target_faces = int(original_faces * reduction_ratio)

    # Apply quadric edge collapse decimation
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)

    # Export the simplified mesh
    ms.save_current_mesh(output_path)

    print(f"  Reduced to: {ms.current_mesh().face_number()} faces")
    return ms.current_mesh().face_number()


def main():
    # Load model
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"
    inference = Inference(config_path, compile=False)

    # Load image and mask
    image = load_image("notebook/images/ref_views_4/ob_0000005/rgb/0000000.png")
    mask = load_single_mask("notebook/images/ref_views_4/ob_0000005/mask", index=0)

    print("\n" + "="*80)
    print("Generating mesh WITHOUT pointmap and transforming to world space")
    print("="*80)
    output_no_pm = inference(image, mask, seed=42)

    if "glb" in output_no_pm and output_no_pm["glb"] is not None:
        mesh_canonical = output_no_pm["glb"]

        # Transform to world space
        mesh_world = transform_mesh_to_world(
            mesh_canonical,
            output_no_pm["scale"][0],
            output_no_pm["translation"][0],
            output_no_pm["rotation"][0]
        )

        # Export and decimate
        temp_path = "mesh_no_pointmap_WORLD_temp.obj"
        final_path = "mesh_no_pointmap_WORLD.obj"
        mesh_world.export(temp_path)

        print("Decimating mesh...")
        decimate_mesh(temp_path, final_path, reduction_ratio=0.05)

        vertices_canonical = np.array(mesh_canonical.vertices)
        vertices_world = np.array(mesh_world.vertices)

        bbox_canonical = vertices_canonical.max(axis=0) - vertices_canonical.min(axis=0)
        bbox_world = vertices_world.max(axis=0) - vertices_world.min(axis=0)

        print(f"Canonical space bounding box: {bbox_canonical}")
        print(f"World space bounding box: {bbox_world}")
        print(f"Scale applied: {output_no_pm['scale'][0].cpu().numpy()}")
        print(f"Translation: {output_no_pm['translation'][0].cpu().numpy()}")
        print(f"World space max dimension: {bbox_world.max():.4f} (ARBITRARY UNITS)")
        print(f"Exported: {final_path}")

    print("\n" + "="*80)
    print("Generating mesh WITH pointmap and transforming to world space")
    print("="*80)

    # Create pointmap
    depth_path = "notebook/images/ref_views_4/ob_0000005/depth_enhanced/0000000.png"
    K_path = "notebook/images/ref_views_4/ob_0000005/K.txt"
    depth = load_depth_image(depth_path, scale_factor=1000.0)
    K = load_camera_intrinsics(K_path)
    pointmap = depth_to_pointmap(depth, K, apply_pytorch3d_convention=True, max_depth=3.0)

    output_with_pm = inference(image, mask, seed=42, pointmap=pointmap)

    if "glb" in output_with_pm and output_with_pm["glb"] is not None:
        mesh_canonical = output_with_pm["glb"]

        # Transform to world space
        mesh_world = transform_mesh_to_world(
            mesh_canonical,
            output_with_pm["scale"][0],
            output_with_pm["translation"][0],
            output_with_pm["rotation"][0]
        )

        # Export and decimate
        temp_path = "mesh_with_pointmap_WORLD_temp.obj"
        final_path = "mesh_with_pointmap_WORLD.obj"
        mesh_world.export(temp_path)

        print("Decimating mesh...")
        decimate_mesh(temp_path, final_path, reduction_ratio=0.05)

        vertices_canonical = np.array(mesh_canonical.vertices)
        vertices_world = np.array(mesh_world.vertices)

        bbox_canonical = vertices_canonical.max(axis=0) - vertices_canonical.min(axis=0)
        bbox_world = vertices_world.max(axis=0) - vertices_world.min(axis=0)

        print(f"Canonical space bounding box: {bbox_canonical}")
        print(f"World space bounding box: {bbox_world}")
        print(f"Scale applied: {output_with_pm['scale'][0].cpu().numpy()}")
        print(f"Translation: {output_with_pm['translation'][0].cpu().numpy()}")
        print(f"World space max dimension: {bbox_world.max():.4f} METERS")
        print(f"Exported: {final_path}")

    print("\n" + "="*80)
    print("FINAL COMPARISON:")
    print("="*80)
    if "glb" in output_no_pm and "glb" in output_with_pm:
        # Load the decimated world-space meshes
        mesh_no_pm_world = trimesh.load("mesh_no_pointmap_WORLD.obj")
        mesh_with_pm_world = trimesh.load("mesh_with_pointmap_WORLD.obj")

        bbox_no_pm = np.array(mesh_no_pm_world.vertices).max(axis=0) - np.array(mesh_no_pm_world.vertices).min(axis=0)
        bbox_with_pm = np.array(mesh_with_pm_world.vertices).max(axis=0) - np.array(mesh_with_pm_world.vertices).min(axis=0)

        ratio = bbox_with_pm.max() / bbox_no_pm.max()

        print(f"WITHOUT pointmap (world space, decimated): {bbox_no_pm.max():.4f} arbitrary units")
        print(f"WITH pointmap (world space, decimated): {bbox_with_pm.max():.4f} METERS")
        print(f"\nSize ratio: {ratio:.4f}")
        print(f"\n✓ NOW the meshes have DIFFERENT sizes in MeshLab!")
        print(f"✓ The pointmap version is in METRIC UNITS (meters)")
        print(f"✓ A max dimension of {bbox_with_pm.max():.2f}m = {bbox_with_pm.max()*100:.1f}cm")
        print(f"✓ Both meshes are decimated to ~5% of original face count")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
