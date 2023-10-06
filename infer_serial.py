from lib import *
from diffusion import *
from datasets.load_test_dataset import TestDataset

import argparse

from autolab_core import YamlConfig

import os
import wandb

import time

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog='Benchmarking Diffusion',
        description='Benchmarking with IK on Test sets',
    )
    parser.add_argument('-c', '--cfg_path', type=str, default='./benchmark/cfgs/cfg1.yaml') # access as args.cfg_path


    args = parser.parse_args() 

    # Load config file
    benchmark_cfg = YamlConfig(args.cfg_path)

    # Load guide params
    guides = benchmark_cfg['guide']['guides'] # list
    guide_dpath = benchmark_cfg['guide']['guide_path'] 

    # Load model params
    device = benchmark_cfg['model']['device'] if torch.cuda.is_available() else 'cpu'
    traj_len = benchmark_cfg['model']['traj_len']
    T = benchmark_cfg['model']['T']
    num_channels = benchmark_cfg['model']['num_channels']

    # Load dataset
    dataset = TestDataset(benchmark_cfg['dataset']['dataset_type'], d_path=benchmark_cfg['dataset']['path']) # Dataset: global, hybrid, or both; d_path: path to dataset
    print(f"Benchmarking dataset: {benchmark_cfg['dataset']['dataset_type']} \tScene types: {benchmark_cfg['dataset']['scene_types']}\tguides: {benchmark_cfg['guide']['guides']}")

    # Load models
    # Load Models:
    env = RobotEnvironment(gui = benchmark_cfg['general']['gui'])
    diffuser = Diffusion(T=benchmark_cfg['model']['T'], device = device)
    model_name = benchmark_cfg['model']['model_dir'] + "TemporalUNetModel" + str(T) + "_N" + str(traj_len)
    if not os.path.exists(model_name):
        print("Model does not exist for these parameters. Train a model first.")
        _ = input("Press anything to exit")
        exit()
    denoiser = TemporalUNet(model_name = model_name, input_dim = num_channels, time_dim = 32, dims=(32, 64, 128, 256, 512, 512),
                            device = device)
    

    enable_wandb = benchmark_cfg['general']['wandb']['enable_wandb']
    

    guide_cfgs = []

    for g_index in guides:
        print(f"Loading Guide {g_index}" + "+-+-"*10)
        g_cfg = YamlConfig(benchmark_cfg['guide']['guide_path'] + f'cfgs/guide{g_index}.yaml')
        obstacle_clearance = np.linspace(g_cfg['hyperparameters']['obstacle_clearance']['range'][0], g_cfg['hyperparameters']['obstacle_clearance']['range'][1], 255)
        expansion_schedule = np.ones(shape = (255,))
        o_e_cfg = g_cfg['hyperparameters']['obstacle_expansion']
        expansion_schedule[o_e_cfg['isr1'][0]: o_e_cfg['isr1'][1]] = np.linspace(o_e_cfg['val1'][0], o_e_cfg['val1'][1], num=abs(o_e_cfg['isr1'][1] - o_e_cfg['isr1'][0]))
        expansion_schedule[o_e_cfg['isr2'][0]: o_e_cfg['isr2'][1]] = np.linspace(o_e_cfg['val2'][0], o_e_cfg['val2'][1], num=abs(o_e_cfg['isr2'][1] - o_e_cfg['isr2'][0]))
        expansion_schedule[o_e_cfg['isr3'][0]: o_e_cfg['isr3'][1]] = np.linspace(o_e_cfg['val3'][0], o_e_cfg['val3'][1], num=abs(o_e_cfg['isr3'][1] - o_e_cfg['isr3'][0]))
        volume_trust_region = g_cfg['hyperparameters']['volume_trust_region']

        guide_cfg = {
            'index': g_index,
            'clearance': obstacle_clearance, 
            'expansion': expansion_schedule, 
            'guidance_method': g_cfg['hyperparameters']['guidance_method'], 
            'grad_norm': g_cfg['hyperparameters']['grad_norm'],
            'volume_trust_region': g_cfg['hyperparameters']['volume_trust_region']
        }
        guide_cfgs.append(guide_cfg)
   

    t_success = 0
    for scene_type in benchmark_cfg['dataset']['scene_types']:
        print(f"Loading Scene Type: {scene_type}" + "+-+-"*10)
        i = 0
        for scene_num in range(dataset.data_nums[scene_type]):
            print(f"Scene num: {i+1}\tSuccess_rate: {t_success}/{i}")
            env.clear_obstacles()
            env.go_home()

            obstacle_config, cuboid_config, cylinder_config, num_cuboids, num_cylinders, start_joints, all_ik_goals = dataset.fetch_data(scene_num=scene_num, scene_type=scene_type)

            guide_set = []
            final_trajectories = []

            start_time = time.time()
            st_time2 = time.time()
            for guide_cfg in guide_cfgs:
                print(f"Guide: {guide_cfg['index']}, Previous planning time: {time.time() - st_time2}") #, end = "")
                st_time2 = time.time()
                guide = IntersectionVolumeGuide(obstacle_config = obstacle_config, device = device, clearance = guide_cfg['clearance'], expansion = guide_cfg['expansion'], 
                                                guidance_method=guide_cfg['guidance_method'], grad_norm=guide_cfg['grad_norm'])
                metrics_calculator = MetricsCalculator(guide)
                guide_set.append([guide, metrics_calculator])

                # Filter final IKs
                st3 = time.time()
                volumes = guide.cost(torch.tensor(all_ik_goals.reshape((-1, 7, 1)), device=device), 0).sum(axis=(1, 2)).cpu().numpy()
                # Sort joints and volumes in the increasing order of volumes
                min_volume = np.min(volumes)
                indices = np.argsort(volumes)#[:10]
                rearranged_volumes = volumes[indices]
                goal_joints = all_ik_goals[indices]
                goal_joints = goal_joints[rearranged_volumes < min_volume + volume_trust_region]
                ideal_ind = np.argmin(np.linalg.norm(start_joints - goal_joints, axis=1))

                goal_joints = goal_joints[ideal_ind]
                print(f"IK time: {time.time() - st3}")

                st4 = time.time()

                trajectories = diffuser.denoise_guided(model = denoiser,
                                                        guide = guide,
                                                        batch_size = g_cfg['hyperparameters']['batch_size'],
                                                        traj_len = traj_len,
                                                        num_channels = num_channels,
                                                        condition = True,
                                                        benchmarking = True,
                                                        start = start_joints,
                                                        goal = goal_joints,
                                                        guidance_schedule=g_cfg['hyperparameters']['guidance_schedule']['type'],
                                                        guidance_scale=g_cfg['hyperparameters']['guidance_schedule']['scale_val'])
                                
                print(f"Denoiser time: {time.time() - st4}")

                trajectory = guide.choose_best_trajectory(start_joints, goal_joints, trajectories)
                final_trajectories.append(trajectory.copy())
                # trajectory is (7, 50) numpy array
                end_time = time.time()

            main_guide = IntersectionVolumeGuide(obstacle_config = obstacle_config, device = device, clearance = guide_cfgs[0]['clearance'], expansion = guide_cfgs[0]['expansion'], 
                                                guidance_method=guide_cfgs[0]['guidance_method'], grad_norm=guide_cfgs[0]['grad_norm'])
            
            trajectory = main_guide.choose_best_trajectory_final(np.array(final_trajectories))

            print(f"\nPlanning Time: {time.time() - start_time} seconds\n")

            if num_cuboids > 0:
                env.spawn_collision_cuboids(cuboid_config)
            if num_cylinders > 0:
                env.spawn_collision_cylinders(cylinder_config)

            
            success = env.benchmark_trajectory(trajectory)
            
            t_success += success
            # print(f"Success: {success}")

            i+=1


    