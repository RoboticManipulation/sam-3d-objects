#!/usr/bin/env python3
"""
SAM 3D Objects ROS2 Action Server
Run this with system Python 3.10 (ROS2 Humble environment)

This node:
1. Receives Sam3DInference action goals with images and masks
2. Communicates with the SAM3D ZeroMQ server (Python 3.11)
3. Returns mesh file paths as action results

Usage:
    ros2 run sam3d_ros2_bridge action_server
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import zmq
import pickle
import numpy as np
from cv_bridge import CvBridge

from sam3d_ros2_interface.action import Sam3DInference
from sam3d_ros2_interface.msg import MeshData


class Sam3DActionServer(Node):
    def __init__(self, zmq_host='localhost', zmq_port=5555):
        super().__init__('sam3d_action_server')

        # Declare parameters
        self.declare_parameter('zmq_host', zmq_host)
        self.declare_parameter('zmq_port', zmq_port)

        # Get parameters
        self.zmq_host = self.get_parameter('zmq_host').value
        self.zmq_port = self.get_parameter('zmq_port').value

        # ZeroMQ setup
        self.zmq_context = zmq.Context()
        self.socket = None
        self.connect_zmq()

        # CV Bridge for ROS Image <-> OpenCV/Numpy conversion
        self.bridge = CvBridge()

        # Action server with reentrant callback group for concurrent execution
        self._action_server = ActionServer(
            self,
            Sam3DInference,
            'sam3d_inference',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )

        self.get_logger().info('SAM 3D Objects Action Server started')
        self.get_logger().info(f'Connected to ZeroMQ server at {self.zmq_host}:{self.zmq_port}')
        self.get_logger().info('Waiting for action goals...')

    def connect_zmq(self):
        """Connect to the ZeroMQ inference server"""
        try:
            if self.socket:
                self.socket.close()

            self.socket = self.zmq_context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{self.zmq_host}:{self.zmq_port}")
            self.get_logger().info(f'Connected to ZeroMQ server at {self.zmq_host}:{self.zmq_port}')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to ZeroMQ server: {e}')
            raise

    def goal_callback(self, goal_request):
        """Accept or reject incoming action goals"""
        self.get_logger().info('Received goal request')

        # Validate input
        if len(goal_request.images) == 0:
            self.get_logger().warn('Rejecting goal: No images provided')
            return GoalResponse.REJECT

        if len(goal_request.masks) == 0:
            self.get_logger().warn('Rejecting goal: No masks provided')
            return GoalResponse.REJECT

        if len(goal_request.images) != len(goal_request.masks):
            self.get_logger().warn(
                f'Rejecting goal: Number of images ({len(goal_request.images)}) '
                f'does not match number of masks ({len(goal_request.masks)})'
            )
            return GoalResponse.REJECT

        self.get_logger().info(f'Accepting goal with {len(goal_request.images)} image-mask pairs')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle action cancellation requests"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def ros_image_to_numpy(self, ros_image):
        """Convert ROS Image message to numpy array"""
        try:
            # Convert ROS Image to OpenCV format (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='passthrough')

            # Handle different encodings
            if ros_image.encoding == 'bgr8' or ros_image.encoding == 'rgb8':
                # Color image
                if ros_image.encoding == 'bgr8':
                    # Convert BGR to RGB
                    cv_image = cv_image[:, :, ::-1]
                return cv_image.astype(np.uint8)

            elif ros_image.encoding == 'mono8' or ros_image.encoding == '8UC1':
                # Grayscale/mask image
                return cv_image.astype(np.uint8)

            else:
                # Try to handle other encodings
                self.get_logger().warn(f'Unexpected encoding: {ros_image.encoding}, attempting conversion')
                return cv_image.astype(np.uint8)

        except Exception as e:
            self.get_logger().error(f'Failed to convert ROS image to numpy: {e}')
            raise

    def execute_callback(self, goal_handle):
        """Execute the SAM3D inference action"""
        self.get_logger().info('Executing goal...')

        goal = goal_handle.request
        feedback_msg = Sam3DInference.Feedback()

        try:
            # Step 1: Convert ROS images to numpy arrays
            feedback_msg.status = 'Converting images and masks to numpy arrays'
            feedback_msg.current_step = 1
            feedback_msg.total_steps = 4
            feedback_msg.progress_percent = 25.0
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(feedback_msg.status)

            images = []
            masks = []

            for idx, (img_msg, mask_msg) in enumerate(zip(goal.images, goal.masks)):
                self.get_logger().info(f'Converting image-mask pair {idx+1}/{len(goal.images)}')

                # Convert images
                img_np = self.ros_image_to_numpy(img_msg)
                mask_np = self.ros_image_to_numpy(mask_msg)

                # Ensure mask is binary
                if mask_np.ndim == 3:
                    mask_np = mask_np[:, :, 0]  # Take first channel if multi-channel

                # Convert mask to boolean
                mask_np = (mask_np > 127).astype(np.uint8)

                images.append(img_np)
                masks.append(mask_np)

                self.get_logger().info(
                    f'  Image shape: {img_np.shape}, dtype: {img_np.dtype}, '
                    f'Mask shape: {mask_np.shape}, dtype: {mask_np.dtype}'
                )

            # Step 2: Prepare request for ZeroMQ server
            feedback_msg.status = 'Preparing inference request'
            feedback_msg.current_step = 2
            feedback_msg.progress_percent = 50.0
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(feedback_msg.status)

            request = {
                'images': images,
                'masks': masks,
                'seed': goal.seed,
                'output_name': goal.output_name if goal.output_name else 'mesh',
                'decimation_ratio': goal.decimation_ratio,
                'merge_scene': goal.merge_scene
            }

            # Step 3: Send request to ZeroMQ server and wait for response
            feedback_msg.status = 'Running SAM3D inference (this may take a while...)'
            feedback_msg.current_step = 3
            feedback_msg.progress_percent = 60.0
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(feedback_msg.status)

            # Send request
            self.socket.send(pickle.dumps(request))

            # Wait for response (this may take a while)
            self.get_logger().info('Waiting for inference server response...')
            response_bytes = self.socket.recv()
            response = pickle.loads(response_bytes)

            # Step 4: Process response
            feedback_msg.status = 'Processing response'
            feedback_msg.current_step = 4
            feedback_msg.progress_percent = 90.0
            goal_handle.publish_feedback(feedback_msg)

            if not response['success']:
                error_msg = response.get('error', 'Unknown error')
                self.get_logger().error(f'Inference failed: {error_msg}')

                goal_handle.abort()
                result = Sam3DInference.Result()
                result.success = False
                result.error_message = error_msg
                return result

            # Success!
            feedback_msg.status = 'Inference completed successfully'
            feedback_msg.progress_percent = 100.0
            goal_handle.publish_feedback(feedback_msg)

            self.get_logger().info('Inference completed successfully!')
            self.get_logger().info(f'Generated {response["num_meshes"]} mesh(es)')
            self.get_logger().info(f'Inference time: {response["inference_time"]:.2f}s')
            self.get_logger().info(f'Export time: {response["export_time"]:.2f}s')
            self.get_logger().info(f'Total time: {response["total_time"]:.2f}s')

            for mesh_path in response['mesh_paths']:
                self.get_logger().info(f'  Mesh: {mesh_path}')

            # Mark goal as succeeded
            goal_handle.succeed()

            # Prepare result
            result = Sam3DInference.Result()
            result.success = True
            result.mesh_paths = response['mesh_paths']
            result.num_meshes = response['num_meshes']
            result.inference_time = response['inference_time']
            result.export_time = response['export_time']
            result.total_time = response['total_time']
            result.error_message = ''

            # Convert mesh_data from ZMQ response to ROS message format
            result.mesh_data = []
            if 'mesh_data' in response:
                for mesh_info in response['mesh_data']:
                    mesh_msg = MeshData()
                    mesh_msg.filename = mesh_info['filename']
                    mesh_msg.data = list(mesh_info['data'])  # Convert bytes to list of uint8
                    mesh_msg.size_bytes = mesh_info['size_bytes']
                    result.mesh_data.append(mesh_msg)
                    self.get_logger().info(f'  Prepared mesh data: {mesh_msg.filename} ({mesh_msg.size_bytes} bytes)')

            return result

        except Exception as e:
            self.get_logger().error(f'Error during execution: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())

            goal_handle.abort()
            result = Sam3DInference.Result()
            result.success = False
            result.error_message = str(e)
            return result

    def destroy_node(self):
        """Clean up resources"""
        if self.socket:
            self.socket.close()
        self.zmq_context.term()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    # Create node
    try:
        action_server = Sam3DActionServer()

        # Use MultiThreadedExecutor to handle concurrent callbacks
        executor = MultiThreadedExecutor()
        executor.add_node(action_server)

        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            action_server.destroy_node()
            executor.shutdown()

    except Exception as e:
        print(f'Failed to start action server: {e}')
        import traceback
        traceback.print_exc()

    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
