# 3D Rigid Body Simulation and ROS Bag Data Processing

## Overview
This project consists of a 3D rigid body simulation model and a ROS2 bag data processing module. It integrates real-world data from ROS2 bag files with a simulated model to compare and validate system dynamics.

## Features
- **Bag Data Processing**: Reads and extracts messages from ROS2 MCAP bag files.
- **3D Rigid Body Simulation**: Implements a physics-based motion model for state estimation.
- **Data Visualization**: Compares real and simulated data using Matplotlib plots.

## Dependencies
This project requires the following Python packages:
- `numpy`
- `scipy`
- `matplotlib`
- `rosbag2_py`
- `rclpy`
- `rosidl_runtime_py`

To install dependencies, run:
```sh
pip install -r requirements.txt
```

## Usage
### Running the Comparator
The `comparator.py` script extracts data from a ROS2 bag, runs the simulator, and visualizes the results.
```sh
python comparator.py
```

### Running the Simulator Independently
You can test the `simulator.py` separately:
```sh
python simulator.py
```
This will simulate a simple scenario with predefined inputs.

## License
This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Author
André Rebelo Teixeira

