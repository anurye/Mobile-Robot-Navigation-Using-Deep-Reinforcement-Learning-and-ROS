# DRL-for-Mobile-Robot-Navigation-Using-ROS2

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
    git clone git@github.com:anurye/DRL-for-Mobile-Robot-Navigation-Using-ROS2.git
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
    ros2 launch drl_agent_bringup simulation.launch.py
    ```

    Terminal 2:
    ```bash
    cd ~/drl_agent_ws
    source install/setup.bash
    ros2 run drl_agent_bringup environment.py 
    ```

    Terminal 3:
    ```bash
    cd ~/drl_agent_ws
    source install/setup.bash
    ros2 run drl_agent train_td7_agent.py
    ```

> [!NOTE]
> The training process can take a a bit of time. You can stop it using `CTRL+C` once you notice an improvement.

## Testing
If you have closed the terminals, restart the simulation in Terminal 1 and Terminal 2 as described above.

Terminal 3:
```bash
cd ~/drl_agent_ws
source install/setup.bash
ros2 run drl_agent test_td7_agent.py
```

## Contribution
Feel free to contribute or ask questions by opening an issue
