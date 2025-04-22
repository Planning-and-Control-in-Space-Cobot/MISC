# Optimal Trajectory Generation

## Overview

The project in this folder is capable of planning an optimal trajectory for a robot in 2D environment considering the dynamics of the system.

## Features
 - Generating a random map in a 2D environment
 - Computing the Signed distance Field for such map
 - Creating and solving multiple optimization problem for the trajectory of the robot showcasing multiple cost function and their advantages

## Dependencies
This script requires:
 - numpy
 - matplotlib
 - shapely
 - skfmm
 - casadi

## Usage
To run this script simply install all dependencies of the project and run the `main.py`, two different modes exist, the first one that generated only one map and that is invoked like this

```bash
python3 main.py
```

And the second one, that generate 100 different random maps showcasing the abilities of the program

```bash
python3 main.py test
```

## File explanation
### MAP
This file contains a class that stores the start and goal position of the robot, as well as the position of each obstacle in the map.

### RRT
This file contains a class that is responsible for generating a path between the start point and the goal, this path is not yet optimal, multiple algorithm exist to execute this, but here we use algorithm of the Random Recursive Tree family.

### SDF
This file contains a class that is responsible for computing the signed distance field for the map, this is a field that is capable of representing the distance of each point to the closest obstacle, here we are still working in a matrix based form, but latter we use an interpolation to have a continual field

### Path Optimizer
This class receives the map and the sdf and generates and solves the optimization problem for the optimal trajectory with dynamic constraints

## License 
This project is licenses under the MIT License - see [LICENSE](../LICENSE) file for more details

## Author
André Rebelo Teixeira