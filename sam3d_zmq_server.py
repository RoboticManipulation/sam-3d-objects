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
from pathlib import Path
import traceback

# Import inference code
sys.path.append("notebook")
from inference import Inference

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

    def process_single_inference(self, image, mask, seed=42):
        """Run inference on a single image-mask pair"""
        print(f"  Running inference on image shape: {image.shape}, mask shape: {mask.shape}")
        start = time.time()
        output = self.inference(image, mask, seed=seed)
        elapsed = time.time() - start
        print(f"  Inference completed in: {elapsed:.3f} seconds")
        return output, elapsed

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
            print(f"  Seed: {seed}")
            print(f"  Output name: {output_name}")
            print(f"  Decimation ratio: {decimation_ratio}")
            print(f"  Merge scene: {merge_scene}")
            print("="*60)

            # Run inference on each image-mask pair
            outputs = []
            total_inference_time = 0

            for idx, (image, mask) in enumerate(zip(images, masks)):
                print(f"\nProcessing pair {idx+1}/{len(images)}:")
                output, elapsed = self.process_single_inference(image, mask, seed)
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
                    original_faces, reduced_faces = self.export_mesh(
                        output["glb"],
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
