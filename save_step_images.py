import pickle
import numpy as np
import argparse
import time
import os
import cv2
from lib.environment import RobotEnvironment

def save_step_images(file_path, output_dir="screenshots", cam_dist=1.7, cam_yaw=50, cam_pitch=-55, cam_target=[0, 0, 0]):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded trajectory data from {file_path}")

    # Initialize Environment (GUI needed for rendering)
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
        # print("Drawing final trajectory path...")
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

    # Specifically requested steps
    target_steps = [250, 100, 50, 5]
    
    print(f"Target steps: {target_steps}")

    waypoint_stride = 5 
    
    # Image Capture settings
    width, height = 1920, 1080 

    for t in target_steps:
        if t not in intermediates:
            print(f"Warning: Step {t} not found in data. Skipping.")
            continue
            
        step_data = intermediates[t]
        
        raw_traj = step_data.get('raw')
        guided_traj = step_data.get('guided')
        per_step_noise = step_data.get('noise')
        grad_norm = step_data.get('gradient_norm', 0)

        # Handle batch dimensions
        if raw_traj is not None and raw_traj.ndim == 3: raw_traj = raw_traj[0]
        if guided_traj is not None and guided_traj.ndim == 3: guided_traj = guided_traj[0]
        if per_step_noise is not None and per_step_noise.ndim == 3: per_step_noise = per_step_noise[0]

        # Cleanup previous bodies and debug items
        for b_id in current_bodies:
            env.client_id.removeBody(b_id)
        current_bodies = []
        for d_id in current_debug_items:
            env.client_id.removeUserDebugItem(d_id)
        current_debug_items = []

        print(f"Processing Step {t} | Gradient: {grad_norm:.4f}")

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
        if raw_traj is not None:
            for wp_idx in range(0, raw_traj.shape[1], waypoint_stride):
                joints = raw_traj[:, wp_idx]
                pos = env.forward_kinematics(joints)[:3, 3]
                vis_id = env.client_id.createVisualShape(shapeType=env.client_id.GEOM_SPHERE, radius=0.015, rgbaColor=[1, 0.5, 0, 0.8])
                body_id = env.client_id.createMultiBody(baseVisualShapeIndex=vis_id, basePosition=pos)
                current_bodies.append(body_id)

        # Plot Guided (Blue)
        if guided_traj is not None:
            for wp_idx in range(0, guided_traj.shape[1], waypoint_stride):
                joints = guided_traj[:, wp_idx]
                pos = env.forward_kinematics(joints)[:3, 3]
                vis_id = env.client_id.createVisualShape(shapeType=env.client_id.GEOM_SPHERE, radius=0.015, rgbaColor=[0, 0.5, 1, 0.9])
                body_id = env.client_id.createMultiBody(baseVisualShapeIndex=vis_id, basePosition=pos)
                current_bodies.append(body_id)

        # Give simulation a moment to render
        # time.sleep(0.5) 
        
        # Force Camera Reset to ensure consistency
        env.client_id.resetDebugVisualizerCamera(
            cameraDistance=cam_dist,
            cameraYaw=cam_yaw,
            cameraPitch=cam_pitch,
            cameraTargetPosition=cam_target
        )
        
        # Capture Image
        img_arr = env.client_id.getCameraImage(width=width, height=height, renderer=env.client_id.ER_BULLET_HARDWARE_OPENGL)[2]
        img_data = np.reshape(img_arr, (height, width, 4))
        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
        
        # Save
        filename = os.path.join(output_dir, f"step_{t:03d}.png")
        cv2.imwrite(filename, img_bgr)
        print(f"Saved {filename}")

    print("All images saved.")
    # input("Press Enter to exit...") # Optional, maybe we want to close immediately if automating

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save images for steps 250, 100, 50, 5.")
    parser.add_argument('-f', '--file', type=str, required=True, help="Path to the .pkl file.")
    parser.add_argument('-o', '--out', type=str, default="screenshots", help="Output directory.")
    
    args = parser.parse_args()

    save_step_images(args.file, output_dir=args.out)
