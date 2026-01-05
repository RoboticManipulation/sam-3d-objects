# SAM 3D Objects - ROS2 Bridge Workspace


## Quick Start

### 1. Install Dependencies

```bash
# For system Python 3.10 (ROS2)
sudo apt install python3-zmq python3-cv-bridge
# or
pip3 install pyzmq opencv-python

# For SAM3D conda environment (Python 3.11)
conda activate sam3d-objects
pip install pyzmq

```

### 2. Build the Workspace

```bash
cd ~/ros2/object_placement/sam-3d-objects/ros2_bridge_ws
conda deactivate
# Build all packages
colcon build

# Source the workspace
source install/setup.bash
```

### 3. Run the System

**Terminal 1 - SAM3D Inference Server (Python 3.11):**
```bash
cd ~/ros2/object_placement/sam-3d-objects
conda activate sam3d-objects
python3 sam3d_zmq_server.py
```

**Terminal 2 - ROS2 Bridge (Python 3.10):**
```bash
cd ~/ros2/object_placement/sam-3d-objects/ros2_bridge_ws
source install/setup.bash

# Option A: Run directly
ros2 run sam3d_ros2_bridge action_server

# Option B: Use launch file
ros2 launch sam3d_ros2_bridge sam3d_bridge.launch.py
```

**Terminal 3 - Test Client:**
```bash
source ~/ros2/object_placement/sam-3d-objects/ros2_bridge_ws/install/setup.bash



# Or with custom images
ros2 run sam3d_ros2_bridge test_client --images path/to/img.png --masks path/to/mask.png
```

## Usage Examples

### Using the Action from Python

```python
import rclpy
from rclpy.action import ActionClient
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sam3d_ros2_interface.action import Sam3DInference

# Initialize
rclpy.init()
node = rclpy.create_node('my_sam3d_client')
client = ActionClient(node, Sam3DInference, 'sam3d_inference')
bridge = CvBridge()

# Create goal
goal = Sam3DInference.Goal()

# Load and convert images
import cv2
img = cv2.imread('image.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
goal.images.append(bridge.cv2_to_imgmsg(img_rgb, encoding='rgb8'))

mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
goal.masks.append(bridge.cv2_to_imgmsg(mask, encoding='mono8'))

# Set parameters
goal.seed = 42
goal.output_name = 'my_object'
goal.decimation_ratio = 0.05

# Send goal
client.wait_for_server()
future = client.send_goal_async(goal)
rclpy.spin_until_future_complete(node, future)

# Get result
result = future.result().get_result_async()
rclpy.spin_until_future_complete(node, result)
print(f"Mesh saved to: {result.result().mesh_paths}")
```

### Using the Launch File with Parameters

```bash
# Custom ZeroMQ server location
ros2 launch sam3d_ros2_bridge sam3d_bridge.launch.py \
    zmq_host:=192.168.1.100 \
    zmq_port:=5556
```

### Using with Config File

```bash
ros2 run sam3d_ros2_bridge action_server \
    --ros-args --params-file src/sam3d_ros2_bridge/config/sam3d_bridge.yaml
```

## Action Interface

### Goal
- `sensor_msgs/Image[] images` - RGB images
- `sensor_msgs/Image[] masks` - Binary masks (mono8)
- `int32 seed` - Random seed (default: 42)
- `string output_name` - Output filename base
- `float32 decimation_ratio` - Mesh decimation 0.01-1.0 (default: 0.05)
- `bool merge_scene` - Merge multiple objects (default: false)

### Result
- `bool success` - Success flag
- `string[] mesh_paths` - Absolute paths to .obj files
- `int32 num_meshes` - Number of meshes generated
- `float32 inference_time` - Inference time (seconds)
- `float32 export_time` - Export time (seconds)
- `float32 total_time` - Total time (seconds)
- `string error_message` - Error if failed

### Feedback
- `string status` - Current status message
- `int32 current_step` - Current step (1-4)
- `int32 total_steps` - Total steps (4)
- `float32 progress_percent` - Progress 0-100

##  Commands

```bash
# Build workspace
colcon build

# Build specific package
colcon build --packages-select sam3d_ros2_interface
colcon build --packages-select sam3d_ros2_bridge

# Clean build
rm -rf build install log
colcon build

# Run action server
ros2 run sam3d_ros2_bridge action_server

# Run test client
ros2 run sam3d_ros2_bridge test_client

# Launch with launch file
ros2 launch sam3d_ros2_bridge sam3d_bridge.launch.py

# List actions
ros2 action list

# Show action interface
ros2 interface show sam3d_ros2_interface/action/Sam3DInference

# Monitor action
ros2 action info /sam3d_inference
```

## Troubleshooting




