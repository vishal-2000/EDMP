import pickle
import torch
import sys
import numpy as np
import os

current_dir = os.getcwd()

sys.path.append(f'{current_dir}/motion-policy-networks/') # path to mpinets

from mpinets.types import PlanningProblem, ProblemSet
from geometrout.primitive import Cuboid, Cylinder
from robofin.robots import FrankaRobot

class TestDataset:
    def __init__(self, type='global', d_path='./datasets/') -> None:
        '''

        Parameters:
        type: 'global' or 'hybrid' or 'both'
        d_path: path to the datasets directory
        '''
        self.type = type

        if d_path[-1]!='/':
            d_path += '/'

        if type=='global':
            # self.data = np.load(d_path + 'global_solvable_problems.pkl', allow_pickle=True)
            with open(d_path + f'global_solvable_problems.pkl', 'rb') as f:
                self.data = pickle.load(f)
        elif type=='hybrid':
            with open(d_path + f'hybrid_solvable_problems.pkl', 'rb') as f:
                self.data = pickle.load(f)
        elif type=='both':
            with open(d_path + f'both_solvable_problems.pkl', 'rb') as f:
                self.data = pickle.load(f)
        else:
            raise ModuleNotFoundError("No such dataset exists")
    
        print('-'*50)
        print(f"Loaded the dataset with {self.data.keys()} scene types, each consisting of {self.data['tabletop'].keys()} problem types")
        print('-'*50)
        print("Dataset statistics:"+'*'*20)
        total_data = 0
        for key1 in self.data.keys():
            for key2 in self.data[key1]:
                print(f"{key1}: {key2}: {len(self.data[key1][key2])} problems")
                total_data += len(self.data[key1][key2])
        print('-'*50)
        print(f"Total data: {total_data}")
        print('-'*50)

        self.tabletop_data = np.hstack((self.data['tabletop']['task_oriented'], self.data['tabletop']['neutral_start'], self.data['tabletop']['neutral_goal']))
        self.cubby_data = np.hstack((self.data['cubby']['task_oriented'], self.data['cubby']['neutral_start'], self.data['cubby']['neutral_goal']))
        self.merged_cubby_data = np.hstack((self.data['merged_cubby']['task_oriented'], self.data['merged_cubby']['neutral_start'], self.data['merged_cubby']['neutral_goal']))
        self.dresser_data = np.hstack((self.data['dresser']['task_oriented'], self.data['dresser']['neutral_start'], self.data['dresser']['neutral_goal']))

        self.data_nums = {}
        self.data_nums['tabletop'] = len(self.tabletop_data)
        self.data_nums['cubby'] = len(self.cubby_data)
        self.data_nums['merged_cubby'] = len(self.cubby_data)
        self.data_nums['dresser'] = len(self.dresser_data)


    def fetch_batch(self, problem_type='random', task_type='random'):
        '''
        Parameters:
        scene_type: 'all' or 'tabletop' or 'cubby' or 'merged_cubby' or 'dresser'
        problem_type: 'random' or 'task_oriented' or 'neutral_start' or 'neutral_goal'
        '''
        raise NotImplementedError("Not implemented yet!")
    
    

    def fetch_data(self, scene_num, scene_type='tabletop'):
        '''
        Parameters:
        scene_type: tabletop or cubby or merged_cubby or dresser
        '''
        if scene_type=='tabletop':
            data = self.tabletop_data[scene_num]
        elif scene_type=='cubby':
            data = self.cubby_data[scene_num]
        elif scene_type=='merged_cubby':
            data = self.merged_cubby_data[scene_num]
        elif scene_type=='dresser':
            data = self.dresser_data[scene_num]
        else:
            raise ModuleNotFoundError("What are you looking for? This dataset only has 4 options. Try to choose one of them and retry!")

        # Initialize lists to store centers and quaternions
        cuboid_centers = []
        cuboid_dims = []
        cuboid_quaternions = []

        cylinder_centers = []
        cylinder_heights = []
        cylinder_radii = []
        cylinder_quaternions = []

        num_cuboids = 0
        num_cylinders = 0

        # Extract centers and quaternions based on object type
        for obstacle in data.obstacles:
            if isinstance(obstacle, Cuboid):
                cuboid_centers.append(np.array(obstacle.center))
                cuboid_quaternions.append(np.array(list(obstacle._pose._so3._quat)))
                cuboid_dims.append(np.array(obstacle.dims))
                num_cuboids += 1
            elif isinstance(obstacle, Cylinder):
                cylinder_centers.append(np.array(obstacle.center))
                cylinder_heights.append(np.array(obstacle.height))
                cylinder_radii.append(np.array(obstacle.radius))
                cylinder_quaternions.append(np.array(list(obstacle._pose._so3._quat)))
                num_cylinders += 1

        # Convert the lists to NumPy arrays
        cuboid_config = []
        cylinder_config = []
        if(num_cuboids >= 1):
            # Convert the lists to NumPy arrays
            cuboid_centers = np.array(cuboid_centers)
            cuboid_dims = np.array(cuboid_dims)
            cuboid_quaternions = np.roll(np.array(cuboid_quaternions), -1, axis = 1)
            cuboid_config = np.concatenate([cuboid_centers, cuboid_quaternions, cuboid_dims], axis = 1)

        if(num_cylinders >= 1):
            cylinder_centers = np.array(cylinder_centers)
            cylinder_heights = np.array(cylinder_heights).reshape(-1, 1)
            cylinder_radii = np.array(cylinder_radii).reshape(-1, 1)
            cylinder_quaternions = np.roll(np.array(cylinder_quaternions), -1, axis = 1)
            cylinder_config = np.concatenate([cylinder_centers, cylinder_quaternions, cylinder_radii, cylinder_heights], axis = 1)

            cylinder_cuboid_dims = np.zeros((num_cylinders, 3))
            cylinder_cuboid_dims[:, 0] = cylinder_radii[:, 0]
            cylinder_cuboid_dims[:, 1] = cylinder_radii[:, 0]
            cylinder_cuboid_dims[:, 2] = cylinder_heights[:, 0]

        if(num_cylinders >= 1):
            obstacle_centers = np.concatenate([cuboid_centers, cylinder_centers], axis = 0)
            obstacle_dims = np.concatenate([cuboid_dims, cylinder_cuboid_dims], axis = 0)
            obstacle_quaternions = np.concatenate([cuboid_quaternions, cylinder_quaternions], axis = 0)
        else: 
            obstacle_centers = cuboid_centers
            obstacle_dims = cuboid_dims
            obstacle_quaternions = cuboid_quaternions

        obstacle_config = np.concatenate([obstacle_centers, obstacle_quaternions, obstacle_dims], axis = 1)
        # cuboid_centers = np.array(cuboid_centers)
        # cuboid_dims = np.array(cuboid_dims)
        # cuboid_quaternions = np.array(cuboid_quaternions)
        # cuboid_config = np.concatenate([cuboid_centers, cuboid_quaternions, cuboid_dims], axis = 1)

        # cylinder_centers = np.array(cylinder_centers)
        # cylinder_heights = np.array(cylinder_heights).reshape(-1, 1)
        # cylinder_radii = np.array(cylinder_radii).reshape(-1, 1)
        # cylinder_quaternions = np.array(cylinder_quaternions)
        # cylinder_config = np.concatenate([cylinder_centers, cylinder_quaternions, cylinder_radii, cylinder_heights], axis = 1)

        # cylinder_cuboid_dims = np.zeros((num_cylinders, 3))
        # cylinder_cuboid_dims[:, 0] = cylinder_radii[:, 0]
        # cylinder_cuboid_dims[:, 1] = cylinder_radii[:, 0]
        # cylinder_cuboid_dims[:, 2] = cylinder_heights[:, 0]

        # obstacle_centers = np.concatenate([cuboid_centers, cylinder_centers], axis = 0)
        # obstacle_dims = np.concatenate([cuboid_dims, cylinder_cuboid_dims], axis = 0)
        # obstacle_quaternions = np.concatenate([cuboid_quaternions, cylinder_quaternions], axis = 0)

        # obstacle_config = np.concatenate([obstacle_centers, obstacle_quaternions, obstacle_dims], axis = 1)

        start = data.q0
        goal_ee = data.target

        joint_7_samples = np.concatenate((np.random.uniform(-2.8973, 2.8973, 50), np.linspace(-2.8973, 2.8973, 50)))
        all_ik_goals = []
        for ik_sol_num, joint_ang in enumerate(joint_7_samples):
            ik_solutions = np.array(FrankaRobot.ik(goal_ee, joint_ang, 'right_gripper')).reshape((-1, 7))
            if len(ik_solutions)==0:
                continue
        
            if len(all_ik_goals)==0:
                all_ik_goals = ik_solutions.reshape((-1, 7))
            else:
                all_ik_goals = np.vstack((all_ik_goals, ik_solutions[0].reshape((-1, 7)))) # n*7
            
        # print(f"Total IK Solutions: {len(all_ik_goals)}")
        return obstacle_config, cuboid_config, cylinder_config, num_cuboids, num_cylinders, start, all_ik_goals
    
    def random(self):
        # Initialize lists to store centers and quaternions
        cuboid_centers = []
        cuboid_dims = []
        cuboid_quaternions = []

        cylinder_centers = []
        cylinder_heights = []
        cylinder_radii = []
        cylinder_quaternions = []

        num_cuboids = 0
        num_cylinders = 0

        cuboid_config, cylinder_config = None, None

        # Extract centers and quaternions based on object type
        for obstacle in obstacles:
            if isinstance(obstacle, Cuboid):
                cuboid_centers.append(np.array(obstacle.center))
                cuboid_quaternions.append(np.array(list(obstacle._pose._so3._quat)))
                cuboid_dims.append(np.array(obstacle.dims))
                num_cuboids += 1
            elif isinstance(obstacle, Cylinder):
                cylinder_centers.append(np.array(obstacle.center))
                cylinder_heights.append(np.array(obstacle.height))
                cylinder_radii.append(np.array(obstacle.radius))
                cylinder_quaternions.append(np.array(list(obstacle._pose._so3._quat)))
                num_cylinders += 1

        if(num_cuboids >= 1):
            # Convert the lists to NumPy arrays
            cuboid_centers = np.array(cuboid_centers)
            cuboid_dims = np.array(cuboid_dims)
            cuboid_quaternions = np.roll(np.array(cuboid_quaternions), -1, axis = 1)
            cuboid_config = np.concatenate([cuboid_centers, cuboid_quaternions, cuboid_dims], axis = 1)

        if(num_cylinders >= 1):
            cylinder_centers = np.array(cylinder_centers)
            cylinder_heights = np.array(cylinder_heights).reshape(-1, 1)
            cylinder_radii = np.array(cylinder_radii).reshape(-1, 1)
            cylinder_quaternions = np.roll(np.array(cylinder_quaternions),  -1, axis = 1)
            # print('cyl centres shape: {}'.format(cylinder_centers.shape))
            cylinder_config = np.concatenate([cylinder_centers, cylinder_quaternions, cylinder_radii, cylinder_heights], axis = 1)

            cylinder_cuboid_dims = np.zeros((num_cylinders, 3))
            cylinder_cuboid_dims[:, 0] = cylinder_radii[:, 0]
            cylinder_cuboid_dims[:, 1] = cylinder_radii[:, 0]
            cylinder_cuboid_dims[:, 2] = cylinder_heights[:, 0]

        if(num_cylinders >= 1):
            obstacle_centers = np.concatenate([cuboid_centers, cylinder_centers], axis = 0)
            obstacle_dims = np.concatenate([cuboid_dims, cylinder_cuboid_dims], axis = 0)
            obstacle_quaternions = np.concatenate([cuboid_quaternions, cylinder_quaternions], axis = 0)
        else: 
            obstacle_centers = cuboid_centers
            obstacle_dims = cuboid_dims
            obstacle_quaternions = cuboid_quaternions

        obstacle_config = np.concatenate([obstacle_centers, obstacle_quaternions, obstacle_dims], axis = 1)

        return obstacle_config, cuboid_config, cylinder_config, num_cuboids, num_cylinders
        
        
if __name__=='__main__':
    tData = TestDataset()

    tData.fetch_data(scene_num=5)
