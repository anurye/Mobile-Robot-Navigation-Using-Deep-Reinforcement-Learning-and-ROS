# DRL-for-Mobile-Robot-Navigation-Using-ROS2

<video controls src="docs/simulation.mp4" title="Title"></video>

## Table of Contents
1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Build](#build)
4. [Training](#training)
5. [Testing](#testing)
6. [Additional Demos](#additional-demos)

## Project Structure
```txt
.
├── 📂 docs/: contains demo videos
│   ├── 📄 dynamic_environment.mp4
│   ├── 📄 slam.mp4
│   └── 📄 simulation.mp4
├── 📂 drl_agent/: main deep reinforcement learning agent directory
│   ├── 📂 config/: contains configuration files
│   ├── 📂 launch/: contains launch files
│   ├── 📂 scripts/: contains code for environment, policies, and utilities
│   └── 📂 temp/: stores models, logs, and results
├── 📂 drl_agent_description/: contains robot description files, models, and URDFs
│   ├── 📂 launch/: launch files for agent description
│   ├── 📂 meshes/: 3D models of the robot
│   ├── 📂 models/: contains specific model files for kinect sensors
│   └── 📂 urdf/: URDF files for camera, laser, and robot description
├── 📂 drl_agent_gazebo/: contains Gazebo simulation configuration and world files
│   ├── 📂 config/: simulation and SLAM configuration files
│   ├── 📂 launch/: Gazebo launch files for various setups
│   ├── 📂 models/: Gazebo models used in the simulation
│   └── 📂 worlds/: simulation worlds for training and testing environments
├── 📂 drl_agent_interfaces/: custom action, message, and service definitions
│   ├── 📂 action/: defines DRL session actions
│   ├── 📂 msg/: empty for now
│   └── 📂 srv/: service definitions for environment and robot interactions
├── 📂 velodyne_simulator/: Velodyne LiDAR simulation setup

```

## Requirements
- Install [Ubuntu 22.04](https://www.releases.ubuntu.com/jammy/)
- Install [ROS2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)
- Install [Gazebo](https://classic.gazebosim.org/tutorials?tut=install_ubuntu&cat=install)
- Install `gazebo_ros_pkgs` by running:
    ```bash
    sudo apt install ros-humble-gazebo-*
    ```
- Install [PyTorch 2.3.1](https://pytorch.org/get-started/locally/)

### Other requirements
```bash
pip install -r requirements.txt
```

## Build
- Clone this repository:
    ```bash
    mkdir -p ~/drl_agent_ws/src
    cd ~/drl_agent_ws/src
    git clone --recurse-submodules git@github.com:anurye/DRL-for-Mobile-Robot-Navigation-Using-ROS2.git .
    ```
- Install dependencies:
    ```bash
    cd ~/drl_agent_ws
    rosdep install --from-path src -yi --rosdistro humble
    ```
- Build the workspace:
    ```bash
    cd ~/drl_agent_ws
    colcon build
    ```

## Training
- Export the environment variable `DRL_AGENT_PACKAGE_PATH`:
    ```bash
    echo 'export DRL_AGENT_PACKAGE_PATH=~/drl_agent_ws/src/drl_agent' >> ~/.bashrc
    source ~/.bashrc
    ```
- Launch the simulation:

    Terminal 1:
    ```bash
    cd ~/drl_agent_ws
    source install/setup.bash
    ros2 launch drl_agent_gazebo simulation.launch.py
    ```

    Terminal 2:
    ```bash
    cd ~/drl_agent_ws
    source install/setup.bash
    ros2 run drl_agent_gazebo environment.py 
    ```

    Terminal 3:
    ```bash
    cd ~/drl_agent_ws
    source install/setup.bash
    ros2 run drl_agent train_td7_agent.py
    ```

## Testing
If you have closed the terminals, restart the simulation in Terminal 1 and Terminal 2 as described above.

Terminal 3:
```bash
cd ~/drl_agent_ws
source install/setup.bash
ros2 run drl_agent test_td7_agent.py
```

## Additional Demos

<div style="display: flex; justify-content: space-between;">
  <video controls src="docs/dynamic_environment.mp4" title="Dynamic Environment" style="width: 48%;"></video>
  <video controls src="docs/slam.mp4" title="SLAM" style="width: 48%;"></video>
</div>

<!-- ```txt
@mastersthesis{Nurye-2024,
author = {Ahmed Yesuf Nurye},
title = {Mobile Robot Navigation in Dynamic Environments},
year = {2024},
month = {October},
school = {Warsaw University of Technology},
address = {Warsaw, Poland},
number = {WUT4f18e5c2cd214a9cb555f730fa440901},
keywords = {Mobile Robot Navigation, Deep Reinforcement Learning, ROS2, Gazebo},
}
``` -->
