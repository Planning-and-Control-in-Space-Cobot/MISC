import numpy as np
import scipy as sp
import pyvista as pv
import argparse
import open3d as o3d

import time
import os 
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
import json 
import pickle


from Environment import EnvironmentHandler
from RRT import RRTPlanner3D, RRTState
import rrtcxx


class Run:  
    def __init__(self, goalBias, stepSize, minPos, maxPos, startPos, startQ, goalPos, goalQ, pcdPath):
        self.goalBias = goalBias
        self.stepSize = stepSize
        self.minPos = minPos
        self.maxPos = maxPos
        self.startPos = startPos
        self.startQ = startQ
        self.goalPos = goalPos
        self.goalQ = goalQ
        self.pcdPath = pcdPath

def execute_run(run):
    env = EnvironmentHandler(o3d.io.read_point_cloud(run.pcdPath))
    planner = rrtcxx.RRTPlanner3D(
            env.triangleVertex, 
            env.triangleIndex,
            np.array([0.0, 0.0, 0.0]), 
            np.array([0.0, 0.0, 0.0]), 
            False, 
            100000, 
            run.stepSize, 
            run.goalBias, 
            run.minPos[0], run.maxPos[0], 
            run.minPos[1], run.maxPos[1], 
            run.minPos[2], run.maxPos[2]
    )
    
    startTime = time.time()
    path = planner.plan(
        rrtcxx.State(run.startPos, run.startQ), 
        rrtcxx.State(run.goalPos, run.goalQ)
    )
    timeTaken = time.time() - startTime



    return {"success": path != [], "timeTaken": timeTaken, "goalBias": run.goalBias, "stepSize": run.stepSize, "pcd" : run.pcdPath}

def main():
    def strToBool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="RRT Path Planning")
    parser.add_argument("--env", type=str, default="default_env", help="Environment name")
    parser.add_argument("--iterations", type=int, default=100000, help="Number of RRT iterations")
    parser.add_argument("--study-case", type=strToBool, default=False)

    args = parser.parse_args()
    env_name = args.env
    iterations = args.iterations

    if not os.path.exists(os.path.join(os.path.dirname(__file__), env_name)):
        raise FileNotFoundError(f"Environment '{env_name}' does not exist.")

    pcd = o3d.io.read_point_cloud(os.path.join(os.path.dirname(__file__), env_name))
    env = EnvironmentHandler(pcd)

    robot = env.buildBox()
    robotMesh = pv.Box(bounds=(-0.225, 0.225, -0.225, 0.225, -0.06, 0.06))
    start = RRTState(np.array([2.75, 2, 1]), np.array([0, 0.707, 0, 0.707]))
    goal = RRTState(np.array([2.75, 2, 7]), np.array([0, 0.707, 0, 0.707]))
    minPos = np.array([1.5, 1, 0])
    maxPos = np.array([4.0, 6, 10])
    ## Now do same path with rrtcxx and compare time taken

    start = rrtcxx.State(np.array([0.5, 3.0, 1.0]), np.array([0, 0, 0, 1]))
    goal = rrtcxx.State(np.array([0.5, 5.0, 6.0]), np.array([0, 0, 0, 1]))
    boundsX = np.array([0.0, 3.0])
    boundsY = np.array([3.0, 5.0])
    boundsZ = np.array([0.0, 7.0])
    numRunsMap = 3
    if args.study_case:
        allRuns = []

        # Load simple map
        simplePcdPath = os.path.join(os.path.dirname(__file__), "simpleMap.pcd")
        simpleMinPos = np.array([0, -2, 0])
        simpleMaxPos = np.array([10, 6, 5])
        simpleStartPos = np.array([3.0, -1.0, 3.0])
        simpleGoalPos = np.array([3, 6, 3])
        simpleStartQ = np.array([0, 0, 0, 1])
        simpleGoalQ = np.array([0, 0, 0, 1])
        simpleStepSizes = np.random.uniform(0.05, 0.45, numRunsMap)
        simpleGoalBiases = np.random.uniform(0.05, 0.45, numRunsMap)
        maps = [
            Run(simpleGoalBiases[i], simpleStepSizes[i], simpleMinPos, simpleMaxPos, simpleStartPos, simpleStartQ, simpleGoalPos, simpleGoalQ, simplePcdPath)
            for i in range(numRunsMap)
        ]
        allRuns.extend(maps)


        # Load Middle Map
        middlePcdPath = os.path.join(os.path.dirname(__file__), "middleMap.pcd")
        middleMinPos = np.array([1.5, 1, 0])
        middleMaxPos = np.array([4.0, 6, 10])
        middleStartPos = np.array([2.75, 2, 1])
        middleGoalPos = np.array([2.75, 2, 7])
        middleStartQ = np.array([0, 0.707, 0, 0.707])
        middleGoalQ = np.array([0, 0.707, 0, 0.707])
        # Randomize step sizes and goal biases
        middleStepSizes = np.random.uniform(0.05, 0.45, numRunsMap)
        middleGoalBiases = np.random.uniform(0.05, 0.45, numRunsMap)
        maps = [
            Run(middleGoalBiases[i], middleStepSizes[i], middleMinPos, middleMaxPos, middleStartPos, middleStartQ, middleGoalPos, middleGoalQ, middlePcdPath)
            for i in range(numRunsMap)
        ]
        allRuns.extend(maps)

        complexPcdPath = os.path.join(os.path.dirname(__file__), "complexMap.pcd")
        complexMinPos = np.array([1.5, 1, 0])
        complexMaxPos = np.array([4.0, 6, 10])
        complexStartPos = np.array([0.5, 3.0, 1.0]) 
        complexGoalPos = np.array([0.5, 5.0, 6.0])
        complexStartQ = np.array([0, 0, 0, 1])
        complexGoalQ = np.array([0, 0, 0, 1])

        complexStepSizes = np.random.uniform(0.05, 0.45, numRunsMap)
        complexGoalBiases = np.random.uniform(0.05, 0.45, numRunsMap)
        maps = [
            Run(complexGoalBiases[i], complexStepSizes[i], complexMinPos, complexMaxPos, complexStartPos, complexStartQ, complexGoalPos, complexGoalQ, complexPcdPath)
            for i in range(numRunsMap)
        ]
        allRuns.extend(maps)

        # Run in parallel
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(execute_run, allRuns)

        output_json = "rrt_results.json"
        output_pkl = "rrt_results.pkl"

        # Save as JSON
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=4)

        # Save as PKL
        with open(output_pkl, 'wb') as f:
            pickle.dump(results, f)
 

        print(f"Results saved to {output_json} and {output_pkl}")
        return
        


        



    startTime = time.time()
    payloadTranslation = np.array([-0.45, 0.0, 0.0])
    payloadSize = np.array([0.45, 0.45, 0.12])
    usePayload = False
    planner = rrtcxx.RRTPlanner3D(env.triangleVertex, env.triangleIndex, payloadTranslation, payloadSize, usePayload, 100000, 0.2, 0.1, 0.0, 3.0,
                                   3.0, 6.0, 0.0, 7.0)
    path = planner.plan(start, goal)
    endTime = time.time()
    print(f"RRTCXX Planning time: {endTime - startTime:.2f} seconds")
    pv_ = pv.Plotter()
    pv_.add_mesh(env.voxel_mesh, color='white', show_edges=True, opacity=0.5)
    
    
    
    if path == []:
        return
    for p in path:
        x = p.position
        q = p.q

        T = np.eye(4)
        T[:3, :3] = sp.spatial.transform.Rotation.from_quat(q, scalar_first=True).as_matrix()
        T[:3, 3] = x
        cube = robotMesh.copy()
        cube.transform(T)
        pv_.add_mesh(cube, color='blue', show_edges=True)
        

    pv_.show()


    positions = np.array([p.position for p in path])
    orientations = np.array([p.q for p in path])

    np.savez(os.path.join(os.path.dirname(__file__), "path.npz"), positions=positions, orientations=orientations)






if __name__ == "__main__":
    main()