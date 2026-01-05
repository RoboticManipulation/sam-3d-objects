#!/usr/bin/env python3
"""
Test client for SAM 3D Objects ROS2 Action Server

Usage:
    # Basic usage (arbitrary scale):
    python3 test_client.py --images img1.png --masks mask1.png

    # With metric scale (depth + camera intrinsics):
    python3 test_client.py --images img1.png --masks mask1.png --depths depth1.png --k-files K.txt

    # Quick test with default data and metric scale:
    python3 test_client.py --use-metric

    # Full example:
    python3 test_client.py \
        --images notebook/images/ref_views_4/ob_0000005/rgb/0000000.png \
        --masks notebook/images/ref_views_4/ob_0000005/mask/0.png \
        --depths notebook/images/ref_views_4/ob_0000005/depth_enhanced/0000000.png \
        --k-files notebook/images/ref_views_4/ob_0000005/K.txt \
        --output-name my_mesh \
        --seed 42 \
        --decimation 0.05 \
        --depth-scale 1000.0 \
        --min-depth 0.01 \
        --max-depth 10.0
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import argparse
import sys
import os
import numpy as np

from sam3d_ros2_interface.action import Sam3DInference


class Sam3DTestClient(Node):
    def __init__(self):
        super().__init__('sam3d_test_client')
        self._action_client = ActionClient(self, Sam3DInference, 'sam3d_inference')
        self.bridge = CvBridge()

    def send_goal(self, image_paths, mask_paths, output_name='test_mesh', seed=42, decimation_ratio=0.05,
                  depth_paths=None, k_paths=None, depth_scale=1000.0, min_depth=0.01, max_depth=10.0):
        """Send inference goal to the action server"""
        goal_msg = Sam3DInference.Goal()

        use_depth = (depth_paths is not None and k_paths is not None)
        if use_depth:
            self.get_logger().info(f'Preparing goal with {len(image_paths)} image-mask-depth-camera_info sets (metric scale enabled)')
        else:
            self.get_logger().info(f'Preparing goal with {len(image_paths)} image-mask pairs')

        # Load and convert images
        for idx, img_path in enumerate(image_paths):
            self.get_logger().info(f'Loading image {idx+1}/{len(image_paths)}: {img_path}')

            # Handle relative paths from SAM3D directory
            if not os.path.isabs(img_path):
                # Try to find the SAM3D root directory
                sam3d_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                test_path = os.path.join(sam3d_root, img_path)
                if os.path.exists(test_path):
                    img_path = test_path

            img = cv2.imread(img_path)
            if img is None:
                self.get_logger().error(f'Failed to load image: {img_path}')
                return False

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ros_img = self.bridge.cv2_to_imgmsg(img_rgb, encoding='rgb8')
            goal_msg.images.append(ros_img)

        # Load and convert masks
        for idx, mask_path in enumerate(mask_paths):
            self.get_logger().info(f'Loading mask {idx+1}/{len(mask_paths)}: {mask_path}')

            # Handle relative paths from SAM3D directory
            if not os.path.isabs(mask_path):
                sam3d_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                test_path = os.path.join(sam3d_root, mask_path)
                if os.path.exists(test_path):
                    mask_path = test_path

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                self.get_logger().error(f'Failed to load mask: {mask_path}')
                return False

            ros_mask = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
            goal_msg.masks.append(ros_mask)

        # Load and convert depth images and camera info if provided
        if use_depth:
            for idx, depth_path in enumerate(depth_paths):
                self.get_logger().info(f'Loading depth {idx+1}/{len(depth_paths)}: {depth_path}')

                # Handle relative paths from SAM3D directory
                if not os.path.isabs(depth_path):
                    sam3d_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                    test_path = os.path.join(sam3d_root, depth_path)
                    if os.path.exists(test_path):
                        depth_path = test_path

                # Load depth image (typically 16-bit)
                depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                if depth is None:
                    self.get_logger().error(f'Failed to load depth image: {depth_path}')
                    return False

                # Convert to ROS message (keep as mono16 if 16-bit, otherwise mono8)
                if depth.dtype == np.uint16:
                    ros_depth = self.bridge.cv2_to_imgmsg(depth, encoding='mono16')
                else:
                    ros_depth = self.bridge.cv2_to_imgmsg(depth, encoding='mono8')
                goal_msg.depth_images.append(ros_depth)

            # Load camera intrinsics and create CameraInfo messages
            for idx, k_path in enumerate(k_paths):
                self.get_logger().info(f'Loading camera intrinsics {idx+1}/{len(k_paths)}: {k_path}')

                # Handle relative paths from SAM3D directory
                if not os.path.isabs(k_path):
                    sam3d_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                    test_path = os.path.join(sam3d_root, k_path)
                    if os.path.exists(test_path):
                        k_path = test_path

                # Load K matrix from file (3x3 matrix)
                K = np.loadtxt(k_path)
                if K.shape != (3, 3):
                    self.get_logger().error(f'Invalid K matrix shape: {K.shape}, expected (3, 3)')
                    return False

                # Create CameraInfo message
                cam_info = CameraInfo()
                # K is a 3x3 matrix, flatten to 9-element array (row-major)
                cam_info.k = K.flatten().tolist()

                # Set dummy values for other fields
                cam_info.height = 480  # Will be overridden by actual image height
                cam_info.width = 640   # Will be overridden by actual image width
                cam_info.distortion_model = "plumb_bob"
                cam_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion
                cam_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # Identity rotation
                cam_info.p = [K[0,0], 0.0, K[0,2], 0.0,
                             0.0, K[1,1], K[1,2], 0.0,
                             0.0, 0.0, 1.0, 0.0]  # Projection matrix from K

                goal_msg.camera_infos.append(cam_info)
                self.get_logger().info(f'  K matrix: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}')

        # Set parameters
        goal_msg.seed = seed
        goal_msg.output_name = output_name
        goal_msg.decimation_ratio = decimation_ratio
        goal_msg.merge_scene = False
        goal_msg.depth_scale = depth_scale
        goal_msg.min_depth = min_depth
        goal_msg.max_depth = max_depth

        self.get_logger().info('Waiting for action server...')
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Action server not available!')
            return False

        self.get_logger().info('Sending goal...')
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        return True

    def feedback_callback(self, feedback_msg):
        """Handle feedback from the action server"""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'[{feedback.progress_percent:5.1f}%] {feedback.status} '
            f'(Step {feedback.current_step}/{feedback.total_steps})'
        )

    def goal_response_callback(self, future):
        """Handle goal acceptance/rejection"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by server!')
            return

        self.get_logger().info('Goal accepted by server, waiting for result...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle result from the action server"""
        result = future.result().result

        print("\n" + "="*60)
        if result.success:
            print("SUCCESS! Inference completed.")
            print("="*60)
            print(f"Generated {result.num_meshes} mesh file(s):")

            # Save mesh data if received
            if hasattr(result, 'mesh_data') and len(result.mesh_data) > 0:
                print("\nSaving received mesh files to local directory...")

                # Create output directory for received meshes
                output_dir = os.path.join(os.getcwd(), 'received_meshes')
                os.makedirs(output_dir, exist_ok=True)

                for idx, mesh_msg in enumerate(result.mesh_data):
                    filename = mesh_msg.filename
                    mesh_bytes = bytes(mesh_msg.data)

                    # Save mesh to local filesystem
                    local_path = os.path.join(output_dir, filename)
                    with open(local_path, 'wb') as f:
                        f.write(mesh_bytes)

                    size_kb = mesh_msg.size_bytes / 1024.0
                    print(f"  [{idx+1}] {filename} ({size_kb:.2f} KB) -> {local_path}")

                print(f"\nAll meshes saved to: {output_dir}")
            else:
                # Fallback: show remote paths if mesh data not available
                print("\nRemote mesh paths:")
                for idx, path in enumerate(result.mesh_paths):
                    print(f"  [{idx+1}] {path}")

            print(f"\nTiming:")
            print(f"  Inference time: {result.inference_time:.2f} seconds")
            print(f"  Export time:    {result.export_time:.2f} seconds")
            print(f"  Total time:     {result.total_time:.2f} seconds")
        else:
            print("FAILED!")
            print("="*60)
            print(f"Error: {result.error_message}")

        print("="*60 + "\n")

        # Shutdown after receiving result
        rclpy.shutdown()


def main(args=None):
    parser = argparse.ArgumentParser(description='Test client for SAM3D ROS2 action server')
    parser.add_argument('--images', nargs='+',
                        help='Paths to input images')
    parser.add_argument('--masks', nargs='+',
                        help='Paths to mask images')
    parser.add_argument('--depths', nargs='+',
                        help='Paths to depth images (optional, for metric scale)')
    parser.add_argument('--k-files', nargs='+',
                        help='Paths to K.txt camera intrinsics files (optional, for metric scale)')
    parser.add_argument('--output-name', type=str, default='test_mesh',
                        help='Base name for output mesh files (default: test_mesh)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for inference (default: 42)')
    parser.add_argument('--decimation', type=float, default=0.05,
                        help='Mesh decimation ratio 0.01-1.0 (default: 0.05)')
    parser.add_argument('--depth-scale', type=float, default=1000.0,
                        help='Depth scale factor (1000.0 for mm->m, 1.0 if already in meters)')
    parser.add_argument('--min-depth', type=float, default=0.01,
                        help='Minimum valid depth in meters (default: 0.01)')
    parser.add_argument('--max-depth', type=float, default=10.0,
                        help='Maximum valid depth in meters (default: 10.0)')
    parser.add_argument('--use-metric', action='store_true',
                        help='Use metric scale inference with default depth/K paths')

    # Parse known args to allow ROS2 args to pass through
    parsed_args, unknown = parser.parse_known_args(args)

    # Use default demo images if none provided
    if parsed_args.images is None:
        print("No images specified, using default demo image...")
        parsed_args.images = ['notebook/images/ref_views_4/ob_0000005/rgb/0000000.png']
        parsed_args.masks = ['notebook/images/ref_views_4/ob_0000005/mask/0.png']

        # If --use-metric flag is set, also load depth and K
        if parsed_args.use_metric:
            print("Using metric scale with default depth and camera intrinsics...")
            parsed_args.depths = ['notebook/images/ref_views_4/ob_0000005/depth_enhanced/0000000.png']
            parsed_args.k_files = ['notebook/images/ref_views_4/ob_0000005/K.txt']

    # Validate inputs
    if len(parsed_args.images) != len(parsed_args.masks):
        print(f"Error: Number of images ({len(parsed_args.images)}) must match number of masks ({len(parsed_args.masks)})")
        sys.exit(1)

    # Validate depth and K files if provided
    if parsed_args.depths is not None or parsed_args.k_files is not None:
        if parsed_args.depths is None or parsed_args.k_files is None:
            print("Error: Both --depths and --k-files must be provided together for metric scale inference")
            sys.exit(1)
        if len(parsed_args.depths) != len(parsed_args.images):
            print(f"Error: Number of depth images ({len(parsed_args.depths)}) must match number of images ({len(parsed_args.images)})")
            sys.exit(1)
        if len(parsed_args.k_files) != len(parsed_args.images):
            print(f"Error: Number of K files ({len(parsed_args.k_files)}) must match number of images ({len(parsed_args.images)})")
            sys.exit(1)

    # Initialize ROS2
    rclpy.init(args=unknown)
    client = Sam3DTestClient()

    # Send goal
    success = client.send_goal(
        parsed_args.images,
        parsed_args.masks,
        output_name=parsed_args.output_name,
        seed=parsed_args.seed,
        decimation_ratio=parsed_args.decimation,
        depth_paths=parsed_args.depths,
        k_paths=parsed_args.k_files,
        depth_scale=parsed_args.depth_scale,
        min_depth=parsed_args.min_depth,
        max_depth=parsed_args.max_depth
    )

    if success:
        # Spin until result is received
        rclpy.spin(client)
    else:
        print("Failed to send goal!")
        rclpy.shutdown()


if __name__ == '__main__':
    main()
