import pickle
import numpy as np
import argparse
import time
import os
from lib.environment import RobotEnvironment

def replay_evolution(file_path, speed=0.05, cam_dist=1.5, cam_yaw=90, cam_pitch=-25, cam_target=[0, 0, 0], 
                      stage_duration=1.0):
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
    
    current_bodies = []
    current_debug_items = []

    # Define steps to visualize: 250 down to 25, step 25
    target_steps = list(range(250, 0, -25)) 
    # Ensure they exist in data
    available_steps = [k for k in intermediates.keys() if isinstance(k, int)]
    valid_steps = [t for t in target_steps if t in available_steps]
    
    if not valid_steps:
        print("Warning: None of the target steps (250, 225... 25) found in data. Using all available.")
        valid_steps = sorted(available_steps, reverse=True)

    print(f"Visualizing steps: {valid_steps}")
    print("Starting Raw vs Guided evolution replay...")

    waypoint_stride = 5 # Hardcoded default for visual clarity

    for t in valid_steps:
        step_data = intermediates[t]
        
        raw_traj = step_data.get('raw')
        guided_traj = step_data.get('guided')
        per_step_noise = step_data.get('noise')
        grad_norm = step_data.get('gradient_norm', 0)

        if raw_traj is None or guided_traj is None:
            continue

        if raw_traj.ndim == 3: raw_traj = raw_traj[0]
        if guided_traj.ndim == 3: guided_traj = guided_traj[0]
        if per_step_noise is not None and per_step_noise.ndim == 3: per_step_noise = per_step_noise[0]

        # Cleanup previous bodies and debug items
        for b_id in current_bodies:
            env.client_id.removeBody(b_id)
        current_bodies = []
        for d_id in current_debug_items:
            env.client_id.removeUserDebugItem(d_id)
        current_debug_items = []

        print(f"Showing Step {t} | Gradient: {grad_norm:.4f}")

        # Text
        info_text = f"Step {t} | Grad: {grad_norm:.4f}"
        current_debug_items.append(env.client_id.addUserDebugText(info_text, [0, 0, 1.0], [0, 0, 0], textSize=1.0))
        current_debug_items.append(env.client_id.addUserDebugText("Black: In-Noise | Orange: Raw | Blue: Guided", [0, 0, 0.9], [0, 0, 0], textSize=0.8))

        # Plot Input Noise (Black)
        if per_step_noise is not None:
            for wp_idx in range(0, per_step_noise.shape[1], waypoint_stride):
                joints = per_step_noise[:, wp_idx]
                pos = env.forward_kinematics(joints)[:3, 3]
                vis_id = env.client_id.createVisualShape(shapeType=env.client_id.GEOM_SPHERE, radius=0.012, rgbaColor=[0, 0, 0, 0.3])
                body_id = env.client_id.createMultiBody(baseVisualShapeIndex=vis_id, basePosition=pos)
                current_bodies.append(body_id)

        # Plot Raw (Orange)
        waypoints_indices = range(0, raw_traj.shape[1], waypoint_stride)
        for wp_idx in waypoints_indices:
            if wp_idx >= raw_traj.shape[1]: continue
            joints = raw_traj[:, wp_idx]
            pos = env.forward_kinematics(joints)[:3, 3]
            vis_id = env.client_id.createVisualShape(shapeType=env.client_id.GEOM_SPHERE, radius=0.015, rgbaColor=[1, 0.5, 0, 0.8])
            body_id = env.client_id.createMultiBody(baseVisualShapeIndex=vis_id, basePosition=pos)
            current_bodies.append(body_id)

        # Plot Guided (Blue)
        for wp_idx in waypoints_indices:
            if wp_idx >= guided_traj.shape[1]: continue
            joints = guided_traj[:, wp_idx]
            pos = env.forward_kinematics(joints)[:3, 3]
            vis_id = env.client_id.createVisualShape(shapeType=env.client_id.GEOM_SPHERE, radius=0.015, rgbaColor=[0, 0.5, 1, 0.9])
            body_id = env.client_id.createMultiBody(baseVisualShapeIndex=vis_id, basePosition=pos)
            current_bodies.append(body_id)

        time.sleep(stage_duration)

    print("Replay finished.")
    input("Press Enter to exit...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Raw vs Guided evolution (Steps 250->25).")
    parser.add_argument('-f', '--file', type=str, required=True, help="Path to the .pkl file.")
    parser.add_argument('--time', type=float, default=1.0, help="Duration for each stage. Default 1.0s.")
    
    args = parser.parse_args()

    replay_evolution(args.file, stage_duration=args.time)
