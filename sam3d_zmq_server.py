#!/usr/bin/env python3
"""
SAM 3D Objects ZeroMQ Inference Server
Run this with Python 3.11 in the sam3d-objects conda environment

Usage:
    conda activate sam3d-objects
    python3 sam3d_zmq_server.py --config checkpoints/hf/pipeline.yaml --port 5555
"""

import sys
import time
import zmq
import pickle
import numpy as np
import torch
import trimesh
from pathlib import Path
import traceback

# Import inference code
sys.path.append("notebook")
from inference import Inference

# Import pointmap creation utilities
from create_pointmap import depth_to_pointmap

class SAM3DServer:
    def __init__(self, config_path="checkpoints/hf/pipeline.yaml", port=5555):
        """Initialize the SAM3D inference server"""
        print("="*60)
        print("Initializing SAM 3D Objects Inference Server...")
        print("="*60)

        # Load model
        print(f"Loading model from config: {config_path}")
        self.inference = Inference(config_path, compile=False)
        print("Model loaded successfully!")

        # Setup ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        print(f"ZeroMQ server listening on port {port}")
        print("="*60 + "\n")

        # Output directory for meshes
        self.output_dir = Path("output_meshes")
        self.output_dir.mkdir(exist_ok=True)

        self.request_count = 0

    def process_single_inference(self, image, mask, seed=42, pointmap=None):
        """Run inference on a single image-mask pair (optionally with pointmap for metric scale)"""
        if pointmap is not None:
            print(f"  Running inference on image shape: {image.shape}, mask shape: {mask.shape}, pointmap shape: {pointmap.shape}")
            print(f"  Pointmap provided - output will be in METRIC SCALE (meters)")
        else:
            print(f"  Running inference on image shape: {image.shape}, mask shape: {mask.shape}")
            print(f"  No pointmap - output will be in ARBITRARY UNITS")

        start = time.time()
        output = self.inference(image, mask, seed=seed, pointmap=pointmap)
        elapsed = time.time() - start
        print(f"  Inference completed in: {elapsed:.3f} seconds")
        return output, elapsed

    def transform_mesh_to_world(self, mesh, scale):
        """
        Transform mesh from canonical space to world space using scale.

        Args:
            mesh: trimesh object
            scale: (3,) tensor or array with scale factors

        Returns:
            transformed mesh
        """
        # Convert to numpy
        if torch.is_tensor(scale):
            scale = scale.cpu().numpy().flatten()

        # Get vertices
        vertices = np.array(mesh.vertices)

        # Apply scale transformation (rotation and translation commented out, as in transform_mesh_to_world.py)
        vertices_scaled = vertices * scale

        # Create new mesh with transformed vertices
        mesh_world = trimesh.Trimesh(
            vertices=vertices_scaled,
            faces=mesh.faces,
            vertex_colors=mesh.visual.vertex_colors if hasattr(mesh.visual, 'vertex_colors') else None
        )

        return mesh_world

    def export_mesh(self, mesh, output_path, decimation_ratio=0.05):
        """Export and optionally decimate a mesh"""
        temp_path = output_path.parent / f"temp_{output_path.name}"

        # Export temporary mesh
        mesh.export(str(temp_path))

        # Decimate mesh using PyMeshLab if available
        try:
            import pymeshlab
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(temp_path))

            original_faces = ms.current_mesh().face_number()
            target_faces = max(1, int(original_faces * decimation_ratio))

            print(f"  Original faces: {original_faces}")
            print(f"  Target faces: {target_faces}")

            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
            ms.save_current_mesh(str(output_path))

            reduced_faces = ms.current_mesh().face_number()
            print(f"  Reduced to: {reduced_faces} faces")

            # Clean up temp file
            temp_path.unlink()
            return original_faces, reduced_faces

        except ImportError:
            print("  PyMeshLab not available, saving without decimation")
            temp_path.rename(output_path)
            return None, None

    def process_request(self, request):
        """Process an inference request"""
        try:
            # Extract data from request
            # Support both single and multiple image-mask pairs
            images = request.get('images')  # list of numpy arrays
            masks = request.get('masks')    # list of numpy arrays

            # Depth and K matrices for metric scale (ROS2 sends these, we create pointmaps)
            depths = request.get('depths', None)  # list of numpy arrays (optional)
            K_matrices = request.get('K_matrices', None)  # list of numpy arrays (optional)
            min_depth = request.get('min_depth', 0.01)
            max_depth = request.get('max_depth', 10.0)

            # Legacy support: pre-computed pointmaps (for backward compatibility)
            pointmaps = request.get('pointmaps', None)  # list of torch tensors (optional)

            # Backwards compatibility: support single image/mask
            if images is None and 'image' in request:
                images = [request['image']]
            if masks is None and 'mask' in request:
                masks = [request['mask']]

            seed = request.get('seed', 42)
            output_name = request.get('output_name', f'mesh_{self.request_count:06d}')
            decimation_ratio = request.get('decimation_ratio', 0.05)
            merge_scene = request.get('merge_scene', False)  # whether to merge multiple objects

            print(f"\n{'='*60}")
            print(f"Processing request #{self.request_count}")
            print(f"  Number of image-mask pairs: {len(images)}")

            # Create pointmaps from depth + K if provided
            if depths is not None and K_matrices is not None:
                print(f"  Creating pointmaps from {len(depths)} depth images + K matrices")
                print(f"  Depth range: [{min_depth}, {max_depth}] meters")
                print(f"  METRIC SCALE ENABLED")

                pointmaps = []
                for idx, (depth, K) in enumerate(zip(depths, K_matrices)):
                    print(f"  Creating pointmap {idx+1}/{len(depths)}...")
                    print(f"    Depth shape: {depth.shape}, dtype: {depth.dtype}")
                    print(f"    Depth range: [{depth.min():.6f}, {depth.max():.6f}] meters")
                    print(f"    K matrix:\n{K}")
                    print(f"    min_depth: {min_depth}, max_depth: {max_depth}")

                    pointmap = depth_to_pointmap(
                        depth, K,
                        apply_pytorch3d_convention=True,
                        min_depth=min_depth,
                        max_depth=max_depth
                    )
                    pointmaps.append(pointmap)

                    # Count valid points
                    valid_mask = torch.isfinite(pointmap).all(dim=-1)
                    valid_points_tensor = pointmap[valid_mask]
                    valid_points = valid_mask.sum().item()
                    total_points = valid_mask.numel()
                    print(f"    Valid points: {valid_points}/{total_points} ({100.0*valid_points/total_points:.1f}%)")
                    if valid_points > 0:
                        print(f"    Pointmap range: [{valid_points_tensor.min():.6f}, {valid_points_tensor.max():.6f}]")
                        print(f"    Mean Z-depth: {valid_points_tensor[:, 2].mean():.6f} meters")

            elif pointmaps is not None:
                print(f"  Number of pointmaps: {len(pointmaps)} (METRIC SCALE ENABLED)")
            else:
                print(f"  No depth/K or pointmaps provided (arbitrary scale)")

            print(f"  Seed: {seed}")
            print(f"  Output name: {output_name}")
            print(f"  Decimation ratio: {decimation_ratio}")
            print(f"  Merge scene: {merge_scene}")
            print("="*60)

            # Run inference on each image-mask pair (with optional pointmap)
            outputs = []
            total_inference_time = 0

            for idx, (image, mask) in enumerate(zip(images, masks)):
                print(f"\nProcessing pair {idx+1}/{len(images)}:")
                # Get pointmap for this pair if available
                pointmap = pointmaps[idx] if pointmaps is not None else None
                output, elapsed = self.process_single_inference(image, mask, seed, pointmap=pointmap)
                outputs.append(output)
                total_inference_time += elapsed

            print(f"\nTotal inference time: {total_inference_time:.3f} seconds")

            # Export mesh(es)
            export_start = time.time()
            mesh_paths = []

            if merge_scene and len(outputs) > 1:
                # Merge multiple objects into a single scene
                print("\nMerging multiple objects into scene...")
                from inference import make_scene

                scene_gs = make_scene(*outputs)

                # Export merged scene (note: this exports gaussian splat, not mesh)
                # For mesh, we'd need to convert or handle differently
                # For now, we'll export individual meshes
                print("Note: Scene merging creates Gaussian splats. Exporting individual meshes.")

            # Export individual meshes
            print("\nExporting meshes...")
            mesh_data_list = []  # Store mesh data to send to client

            for idx, output in enumerate(outputs):
                if "glb" in output and output["glb"] is not None:
                    if len(outputs) > 1:
                        mesh_name = f"{output_name}_{idx:03d}.obj"
                    else:
                        mesh_name = f"{output_name}.obj"

                    mesh_path = self.output_dir / mesh_name

                    print(f"\nExporting mesh {idx+1}/{len(outputs)}: {mesh_name}")

                    # Get the mesh from output
                    mesh_canonical = output["glb"]

                    # DEBUG: Check what we have
                    print(f"  DEBUG: pointmaps is not None: {pointmaps is not None}")
                    print(f"  DEBUG: 'scale' in output: {'scale' in output}")
                    print(f"  DEBUG: output keys: {output.keys()}")

                    # Apply scale transformation if pointmaps were used (metric scale)
                    if pointmaps is not None and "scale" in output:
                        print(f"  Applying scale transformation for metric units...")
                        scale = output["scale"][0]  # Get scale for this output
                        print(f"  Scale: {scale.cpu().numpy() if torch.is_tensor(scale) else scale}")

                        # Transform mesh to world space (metric scale)
                        mesh_world = self.transform_mesh_to_world(mesh_canonical, scale)

                        # Get bounding box info
                        bbox_canonical = np.array(mesh_canonical.vertices).max(axis=0) - np.array(mesh_canonical.vertices).min(axis=0)
                        bbox_world = np.array(mesh_world.vertices).max(axis=0) - np.array(mesh_world.vertices).min(axis=0)
                        print(f"  Canonical bbox: {bbox_canonical}")
                        print(f"  World bbox (meters): {bbox_world}")
                        print(f"  Max dimension: {bbox_world.max():.4f} meters = {bbox_world.max()*100:.2f} cm")

                        mesh_to_export = mesh_world
                    else:
                        print(f"  No scale transformation (arbitrary units)")
                        mesh_to_export = mesh_canonical

                    # Export and decimate
                    original_faces, reduced_faces = self.export_mesh(
                        mesh_to_export,
                        mesh_path,
                        decimation_ratio
                    )

                    mesh_paths.append(str(mesh_path.absolute()))

                    # Read mesh file and store data to send to client
                    with open(mesh_path, 'rb') as f:
                        mesh_bytes = f.read()
                    mesh_data_list.append({
                        'filename': mesh_name,
                        'data': mesh_bytes,
                        'size_bytes': len(mesh_bytes)
                    })
                    print(f"  Mesh size: {len(mesh_bytes) / 1024:.2f} KB")
                else:
                    print(f"Warning: No mesh found in output {idx}")

            export_elapsed = time.time() - export_start
            total_elapsed = total_inference_time + export_elapsed

            print(f"\n{'='*60}")
            print(f"Mesh export completed in: {export_elapsed:.3f} seconds")
            print(f"Total processing time: {total_elapsed:.3f} seconds")
            print(f"Saved {len(mesh_paths)} mesh(es)")
            print("="*60 + "\n")

            return {
                'success': True,
                'mesh_paths': mesh_paths,
                'mesh_data': mesh_data_list,  # Actual mesh file data
                'num_meshes': len(mesh_paths),
                'inference_time': total_inference_time,
                'export_time': export_elapsed,
                'total_time': total_elapsed
            }

        except Exception as e:
            print(f"Error processing request: {str(e)}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def run(self):
        """Main server loop"""
        print("Server ready to accept requests. Press Ctrl+C to stop.\n")

        try:
            while True:
                # Wait for request
                print(f"Waiting for request #{self.request_count}...")
                message = self.socket.recv()

                # Deserialize request
                request = pickle.loads(message)

                # Process request
                response = self.process_request(request)

                # Send response
                self.socket.send(pickle.dumps(response))

                self.request_count += 1

        except KeyboardInterrupt:
            print("\n\nShutting down server...")
        finally:
            self.socket.close()
            self.context.term()
            print("Server stopped.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='SAM 3D Objects ZeroMQ Inference Server')
    parser.add_argument('--config', type=str, default='checkpoints/hf/pipeline.yaml',
                        help='Path to model config file')
    parser.add_argument('--port', type=int, default=5555,
                        help='ZeroMQ port to listen on')

    args = parser.parse_args()

    server = SAM3DServer(config_path=args.config, port=args.port)
    server.run()


if __name__ == '__main__':
    main()
