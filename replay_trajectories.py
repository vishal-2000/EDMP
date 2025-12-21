
import pickle
import numpy as np
import argparse
import time
import os
from lib.environment import RobotEnvironment

def replay_trajectory(file_path, speed=0.05, cam_dist=1.5, cam_yaw=90, cam_pitch=-25, cam_target=[0, 0, 0]):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded trajectory data from {file_path}")
    print(f"Scene Type: {data.get('scene_type', 'Unknown')}")
    print(f"Success: {data.get('success', 'Unknown')}")

    # Initialize Environment
    # gui=True is crucial for visualization
    env = RobotEnvironment(gui=True, manipulator=True, benchmarking=False)
    
    # Set Camera
    print(f"Setting camera: dist={cam_dist}, yaw={cam_yaw}, pitch={cam_pitch}, target={cam_target}")
    env.client_id.resetDebugVisualizerCamera(
        cameraDistance=cam_dist,
        cameraYaw=cam_yaw,
        cameraPitch=cam_pitch,
        cameraTargetPosition=cam_target
    )
    
    # Reset Environment
    env.clear_obstacles()
    env.go_home()

    # Load Obstacles
    cuboid_config = data.get('cuboid_config')
    cylinder_config = data.get('cylinder_config')

    if cuboid_config is not None and len(cuboid_config) > 0:
        # Use collision spawner which includes visual shape as well
        env.spawn_collision_cuboids(cuboid_config)
    
    if cylinder_config is not None and len(cylinder_config) > 0:
        env.spawn_collision_cylinders(cylinder_config)

    # Get trajectory and joints
    trajectory = data['trajectory']
    start_joints = data['start_joints']
    # goal_joints = data['goal_joints']

    # Set robot to start position
    # env.move_joints(start_joints) # Use move_joints or resetJointState loop
    for i, joint_ind in enumerate(env.joints):
        env.client_id.resetJointState(env.manipulator, joint_ind, start_joints[i])

    # --- Visualization Start ---
    # import pybullet as p # No longer needed, use env.client_id
    
    col_red = [1, 0, 0]
    col_green = [0, 1, 0]
    col_blue = [0, 0, 1]

    # 1. Visualize Start (Red Sphere)
    start_pos = env.forward_kinematics(start_joints)[:3, 3] # Get XYZ
    env.client_id.addUserDebugText("Start", start_pos, col_red, textSize=1.5)
    
    visual_shape_id_start = env.client_id.createVisualShape(shapeType=env.client_id.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 0.8])
    env.client_id.createMultiBody(baseVisualShapeIndex=visual_shape_id_start, basePosition=start_pos)

    # 2. Visualize Goal (Green Sphere)
    if 'goal_joints' in data:
        goal_joints = data['goal_joints']
        goal_pos = env.forward_kinematics(goal_joints)[:3, 3]
        env.client_id.addUserDebugText("Goal", goal_pos, col_green, textSize=1.5)
        
        visual_shape_id_goal = env.client_id.createVisualShape(shapeType=env.client_id.GEOM_SPHERE, radius=0.03, rgbaColor=[0, 1, 0, 0.8])
        env.client_id.createMultiBody(baseVisualShapeIndex=visual_shape_id_goal, basePosition=goal_pos)
    else:
        print("Warning: goal_joints not found in data, skipping goal visualization.")

    # 3. Visualize Trajectory Path (Blue Lines)
    print("Drawing trajectory path...")
    traj_points = []
    T = trajectory.shape[1]
    
    # Pre-calculate positions
    for t in range(T):
        joints = trajectory[:, t]
        pos = env.forward_kinematics(joints)[:3, 3]
        traj_points.append(pos)
    
    # Draw lines
    for i in range(len(traj_points) - 1):
        env.client_id.addUserDebugLine(traj_points[i], traj_points[i+1], lineColorRGB=col_blue, lineWidth=3.0)
    
    # --- Visualization End ---

    input("Press Enter to start replay...")

    # Execute Trajectory
    # trajectory shape is expected to be (7, T)
    T = trajectory.shape[1]
    
    for t in range(T):
        target_joints = trajectory[:, t]
        env.move_joints(target_joints)
        env.client_id.stepSimulation()
        time.sleep(speed)
    
    print("Replay finished.")
    input("Press Enter to exit...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replay saved trajectories from EDMP inference.")
    parser.add_argument('-f', '--file', type=str, required=True, help="Path to the .pkl file containing trajectory data.")
    parser.add_argument('-s', '--speed', type=float, default=0.05, help="Time delay between steps (seconds). Default 0.05")

    args = parser.parse_args()

    replay_trajectory(args.file, args.speed)
