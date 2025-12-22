import pickle
import numpy as np
import argparse
import time
import os
from lib.environment import RobotEnvironment

def inspect_step(file_path, target_t):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded trajectory data from {file_path}")

    # Initialize Environment
    env = RobotEnvironment(gui=True, manipulator=True, benchmarking=False)
    
    # Set Camera
    env.client_id.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=90,
        cameraPitch=-25,
        cameraTargetPosition=[0, 0, 0]
    )
    
    # Reset Environment
    env.clear_obstacles()
    env.go_home()

    # Load Obstacles
    cuboid_config = data.get('cuboid_config')
    cylinder_config = data.get('cylinder_config')

    if cuboid_config is not None and len(cuboid_config) > 0:
        env.spawn_collision_cuboids(cuboid_config)

    if cylinder_config is not None and len(cylinder_config) > 0:
        env.spawn_collision_cylinders(cylinder_config)

    # --- Start/Goal Visualization ---
    if 'start_joints' in data:
        start_pos = env.forward_kinematics(data['start_joints'])[:3, 3]
        env.client_id.addUserDebugText("Start", start_pos, [1, 0, 0], textSize=1.5)
        vis_start = env.client_id.createVisualShape(shapeType=env.client_id.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 0.8])
        env.client_id.createMultiBody(baseVisualShapeIndex=vis_start, basePosition=start_pos)

    if 'goal_joints' in data:
        goal_pos = env.forward_kinematics(data['goal_joints'])[:3, 3]
        env.client_id.addUserDebugText("Goal", goal_pos, [0, 1, 0], textSize=1.5)
        vis_goal = env.client_id.createVisualShape(shapeType=env.client_id.GEOM_SPHERE, radius=0.03, rgbaColor=[0, 1, 0, 0.8])
        env.client_id.createMultiBody(baseVisualShapeIndex=vis_goal, basePosition=goal_pos)

    # --- Final Path Visualization ---
    if 'trajectory' in data:
        print("Drawing final trajectory path...")
        trajectory = data['trajectory']
        # Handle batch dimension if present
        if trajectory.ndim == 3: trajectory = trajectory[0]

        traj_points = []
        for i in range(trajectory.shape[1]):
            joints = trajectory[:, i]
            pos = env.forward_kinematics(joints)[:3, 3]
            traj_points.append(pos)
        
        for i in range(len(traj_points) - 1):
            env.client_id.addUserDebugLine(traj_points[i], traj_points[i+1], lineColorRGB=[0, 0, 1], lineWidth=2.0)

    # --- Visualization Logic ---
    if 'intermediate_trajectories' not in data:
        print("Error: No intermediate trajectories found in pickle.")
        return

    intermediates = data['intermediate_trajectories']
    
    # Check if target step exists
    if target_t not in intermediates:
        print(f"Error: Step {target_t} not found in data. Available steps: {sorted([k for k in intermediates.keys() if isinstance(k, int)])}")
        return

    step_data = intermediates[target_t]
    
    # Components
    initial_noise = intermediates.get('initial_noise')
    raw_traj = step_data.get('raw')
    guided_traj = step_data.get('guided')
    per_step_noise = step_data.get('noise')
    grad_norm = step_data.get('gradient_norm', 0)

    # Handle Batch Dimension (take first element)
    if initial_noise is not None and initial_noise.ndim == 3: initial_noise = initial_noise[0]
    if raw_traj is not None and raw_traj.ndim == 3: raw_traj = raw_traj[0]
    if guided_traj is not None and guided_traj.ndim == 3: guided_traj = guided_traj[0]
    if per_step_noise is not None and per_step_noise.ndim == 3: per_step_noise = per_step_noise[0]

    print(f"Inspecting Step {target_t} | Gradient: {grad_norm:.4f}")
    
    # Debug Text
    env.client_id.addUserDebugText(f"Step {target_t} | Grad: {grad_norm:.4f}", [0, 0, 1.0], [0, 0, 0], textSize=1.0)
    legend = "Black: Input Noise | Orange: Raw | Blue: Guided"
    env.client_id.addUserDebugText(legend, [0, 0, 0.9], [0, 0, 0], textSize=0.8)

    # Draw !
    waypoint_stride = 5

    # 1. Input Noise (Pre-denoise state at step t) -> Black
    # If per_step_noise is missing (old pickle), fallback to initial_noise but warn
    noise_to_draw = per_step_noise if per_step_noise is not None else initial_noise
    
    if noise_to_draw is not None:
        for i in range(0, noise_to_draw.shape[1], waypoint_stride):
            joints = noise_to_draw[:, i]
            pos = env.forward_kinematics(joints)[:3, 3]
            vis_id = env.client_id.createVisualShape(shapeType=env.client_id.GEOM_SPHERE, radius=0.012, rgbaColor=[0, 0, 0, 0.3])
            env.client_id.createMultiBody(baseVisualShapeIndex=vis_id, basePosition=pos)
    else:
        print("Warning: No noise data found.")

    # 2. Raw (Orange)
    if raw_traj is not None:
        for i in range(0, raw_traj.shape[1], waypoint_stride):
            joints = raw_traj[:, i]
            pos = env.forward_kinematics(joints)[:3, 3]
            vis_id = env.client_id.createVisualShape(shapeType=env.client_id.GEOM_SPHERE, radius=0.015, rgbaColor=[1, 0.5, 0, 0.8])
            env.client_id.createMultiBody(baseVisualShapeIndex=vis_id, basePosition=pos)
    
    # 3. Guided (Blue)
    if guided_traj is not None:
        for i in range(0, guided_traj.shape[1], waypoint_stride):
            joints = guided_traj[:, i]
            pos = env.forward_kinematics(joints)[:3, 3]
            vis_id = env.client_id.createVisualShape(shapeType=env.client_id.GEOM_SPHERE, radius=0.015, rgbaColor=[0, 0.5, 1, 0.9])
            env.client_id.createMultiBody(baseVisualShapeIndex=vis_id, basePosition=pos)

    print("Visualization ready. Press Enter to exit.")
    input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect a specific diffusion step (Noise/Raw/Guided).")
    parser.add_argument('-f', '--file', type=str, required=True, help="Path to the .pkl file.")
    parser.add_argument('-t', '--target', type=int, required=True, help="Target timestep to inspect (e.g. 150).")
    
    args = parser.parse_args()

    inspect_step(args.file, args.target)
