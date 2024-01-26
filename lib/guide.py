import numpy as np
import torch
import os
import re
from einops.layers.torch import Rearrange
from scipy.spatial.transform import Rotation as R
# import pybullet as p

import pybullet_data

class IntersectionVolumeGuide:

    def __init__(self, obstacle_config, device, guide_cfgs, batch_size):
        '''
        Parameters:

        cost_type = 'iv' or 'sv' or 'ipv' (intersection volume or swept volume or interpolated volume) 
        grad_norm = True or False (indicates whether to take the norm of the gradient or not)
        '''

        # self.env = env
        self.device = device
        self.guide_cfgs = guide_cfgs
        
        self.obstacle_config = np.array(obstacle_config)
        self.obs_ids = []
        self.batch_size = batch_size

        self.static_dh_params = torch.tensor([[0, 0.333, 0, 0],
                                  [0, 0, -torch.pi / 2, 0],
                                  [0, 0.316, torch.pi / 2, 0],
                                  [0.0825, 0, torch.pi / 2, 0],
                                  [-0.0825, 0.384, -torch.pi / 2, 0],
                                  [0, 0, torch.pi / 2, 0],
                                  [0.088, 0, torch.pi / 2, 0],
                                  [0, 0.107, 0, 0],
                                  [0, 0, 0, -torch.pi / 4],
                                  [0.0, 0.1034, 0, 0]], dtype = torch.float32, device = self.device)
        
        self.define_link_information()
        # self.define_obstacles(obstacle_config)

        self.rearrange_joints = Rearrange('batch channels traj_len -> batch traj_len channels')
    
    def get_tf_mat(self, dh_params):
        
        # dh_params is (batch, traj_len, 1, 4)
        a = dh_params[:, :, 0]
        d = dh_params[:, :, 1]
        alpha = dh_params[:, :, 2]
        q = dh_params[:, :, 3]

        transform = torch.zeros(dh_params.shape[0], dh_params.shape[1], 4, 4)

        transform[:, :, 0, 0] = torch.cos(q)
        transform[:, :, 0, 1] = -torch.sin(q)
        transform[:, :, 0, 3] = a

        transform[:, :, 1, 0] = torch.sin(q) * torch.cos(alpha)
        transform[:, :, 1, 1] = torch.cos(q) * torch.cos(alpha)
        transform[:, :, 1, 2] = -torch.sin(alpha)
        transform[:, :, 1, 3] = -torch.sin(alpha) * d

        transform[:, :, 2, 0] = torch.sin(q) * torch.sin(alpha)
        transform[:, :, 2, 1] = torch.cos(q) * torch.sin(alpha)
        transform[:, :, 2, 2] = torch.cos(alpha)
        transform[:, :, 2, 3] = torch.cos(alpha) * d

        transform[:, :, -1, -1] = 1

        # Return transform of (batch, traj_len, 4, 4)
        return transform
    
    def forward_kinematics(self, joints):
        """
        joint_angles => Array of shape (batch, 7, traj_len)
        """        

        dh_params = torch.clone(self.static_dh_params)
        dh_params = dh_params.unsqueeze(0).unsqueeze(0).repeat(joints.shape[0], joints.shape[1], 1, 1)
        dh_params[:, :, :7, 3] = joints[:, :, :]
        # dh_params is (batch, traj_len, 10, 4)

        fk = torch.zeros(size = (dh_params.shape[0], dh_params.shape[1], 9, 4, 4), dtype = torch.float32, device = self.device)
        # fk is (batch, traj_len, 9, 4, 4)
    
        T = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(joints.shape[0], joints.shape[1], 1, 1)
        for i in range(7):
            dh_matrix = self.get_tf_mat(dh_params[:, :, i, :]) 
            # T is (batch, traj_len, 4, 4)
            # dh_matrix is (batch, traj_len, 4, 4)
            T = torch.matmul(T, dh_matrix) 
            if i == 6:
                fk[:, :, i:, :, :] = T.unsqueeze(2)
            else:
                fk[:, :, i, :, :] = T

        return fk
    
    def get_end_effector_transform(self, joints):

        # joints is (1, 50, 7) tensor

        dh_params = torch.clone(self.static_dh_params)
        dh_params = dh_params.unsqueeze(0).unsqueeze(0).repeat(joints.shape[0], joints.shape[1], 1, 1)
        dh_params[:, :, :7, 3] = joints[:, :, :]
        # dh_params is (1, 50, 10, 4)

        T = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(joints.shape[0], joints.shape[1], 1, 1)
        for i in range(10):
            dh_matrix = self.get_tf_mat(dh_params[:, :, i, :]) 
            # T is (batch, traj_len, 4, 4)
            # dh_matrix is (batch, traj_len, 4, 4)
            T = torch.matmul(T, dh_matrix)

        return T        

    def define_obstacles(self, obstacle_config, t, batch_size = None):

        # Obstacle Config => (n, 10)
        
        # Hyperparameters used here are clearance and expansion, both defined in the init function.
        if batch_size != None:
            b = batch_size
        else:
            b = self.batch_size
        obstacle_sizes = np.array(obstacle_config[np.newaxis, :, 7:])
        obstacle_sizes = np.repeat(obstacle_sizes, b, axis = 0)
        # Now obstacle_sizes is (total_batch_size, n0, 3)

        if t != 0:
            # Expansion (total_batch_size, T):    
            obstacle_sizes = np.maximum(obstacle_sizes, self.guide_cfgs['expansion'][:, t-1, np.newaxis, np.newaxis])

            # Clearance (total_batch_size, T):
            obstacle_sizes = obstacle_sizes + self.guide_cfgs['clearance'][:, t-1, np.newaxis, np.newaxis]    # It's t-1 because the size of clearance is 255, so last index is 254
        
        obstacle_static_vertices = self.get_obstacle_vertices(obstacle_sizes)
        # obstacle_static_vertices => (total_batch_size, no, 4, 8)

        obstacle_transform = np.zeros((obstacle_config.shape[0], 4, 4))
        for i in range(obstacle_config.shape[0]):
            obstacle_transform[i, :3, :3] = R.from_quat(obstacle_config[i, 3:7]).as_matrix() #).reshape((3, 3))
            obstacle_transform[i, :3, -1] = obstacle_config[i, :3]
        obstacle_transform[:, -1, -1] = 1.
        # obstacle_transform => (no, 4, 4)
        obstacle_transform = obstacle_transform[np.newaxis, :]
        obstacle_transform = np.repeat(obstacle_transform, b, axis = 0)
        # Now obstacle_transform is (total_batch_size, n0, 4, 4)

        obstacle_transform = torch.tensor(obstacle_transform, dtype = torch.float32, device = self.device)
        obstacle_vertices = torch.matmul(obstacle_transform, obstacle_static_vertices)
        # obstacle_vertices => (total_batch_size, n0, 4, 8)

        self.obs_min = torch.min(obstacle_vertices, dim = -1)[0][:, :, :-1]
        self.obs_max = torch.max(obstacle_vertices, dim = -1)[0][:, :, :-1]
        # obs_min => (total_batch_size, n0, 3)
        # obs_max => (total_batch_size, n0, 3)
    
    def get_obstacle_vertices(self, dimensions):

        # dimensions -> (total_batch_size, no, 3)
        
        l, b, h = dimensions[:, :, 0], dimensions[:, :, 1], dimensions[:, :, 2]
        l = torch.tensor(l, dtype = torch.float32, device = self.device)
        b = torch.tensor(b, dtype = torch.float32, device = self.device)
        h = torch.tensor(h, dtype = torch.float32, device = self.device)
        obstacle_vertices = torch.zeros(size = (dimensions.shape[0], dimensions.shape[1], 4, 8), dtype = torch.float32, device = self.device)

        obstacle_vertices[:, :, 0, 0] = -l/2
        obstacle_vertices[:, :, 0, 1] = l/2
        obstacle_vertices[:, :, 0, 2] = l/2
        obstacle_vertices[:, :, 0, 3] = -l/2
        obstacle_vertices[:, :, 0, 4] = -l/2
        obstacle_vertices[:, :, 0, 5] = l/2
        obstacle_vertices[:, :, 0, 6] = l/2
        obstacle_vertices[:, :, 0, 7] = -l/2

        obstacle_vertices[:, :, 1, 0] = -b/2
        obstacle_vertices[:, :, 1, 1] = -b/2
        obstacle_vertices[:, :, 1, 2] = b/2
        obstacle_vertices[:, :, 1, 3] = b/2
        obstacle_vertices[:, :, 1, 4] = -b/2
        obstacle_vertices[:, :, 1, 5] = -b/2
        obstacle_vertices[:, :, 1, 6] = b/2
        obstacle_vertices[:, :, 1, 7] = b/2

        obstacle_vertices[:, :, 2, 0] = -h/2
        obstacle_vertices[:, :, 2, 1] = -h/2
        obstacle_vertices[:, :, 2, 2] = -h/2
        obstacle_vertices[:, :, 2, 3] = -h/2
        obstacle_vertices[:, :, 2, 4] = h/2
        obstacle_vertices[:, :, 2, 5] = h/2
        obstacle_vertices[:, :, 2, 6] = h/2
        obstacle_vertices[:, :, 2, 7] = h/2

        obstacle_vertices[:, :, 3, :] = 1.
        
        # obstacle_vertices -> (total_batch_size, no, 4, 8)

        return obstacle_vertices
    
    def get_link_vertices(self, dimensions):

        # dimensions -> (nl, 3)
        
        l, b, h = dimensions[:, 0], dimensions[:, 1], dimensions[:, 2]
        link_vertices = torch.zeros(size = (dimensions.shape[0], 4, 8), dtype = torch.float32, device = self.device)

        link_vertices[:, 0, 0] = -l/2
        link_vertices[:, 0, 1] = l/2
        link_vertices[:, 0, 2] = l/2
        link_vertices[:, 0, 3] = -l/2
        link_vertices[:, 0, 4] = -l/2
        link_vertices[:, 0, 5] = l/2
        link_vertices[:, 0, 6] = l/2
        link_vertices[:, 0, 7] = -l/2

        link_vertices[:, 1, 0] = -b/2
        link_vertices[:, 1, 1] = -b/2
        link_vertices[:, 1, 2] = b/2
        link_vertices[:, 1, 3] = b/2
        link_vertices[:, 1, 4] = -b/2
        link_vertices[:, 1, 5] = -b/2
        link_vertices[:, 1, 6] = b/2
        link_vertices[:, 1, 7] = b/2

        link_vertices[:, 2, 0] = -h/2
        link_vertices[:, 2, 1] = -h/2
        link_vertices[:, 2, 2] = -h/2
        link_vertices[:, 2, 3] = -h/2
        link_vertices[:, 2, 4] = h/2
        link_vertices[:, 2, 5] = h/2
        link_vertices[:, 2, 6] = h/2
        link_vertices[:, 2, 7] = h/2

        link_vertices[:, 3, :] = 1.
        
        # link_vertices -> (nl, 4, 8)

        return link_vertices
    
    def define_link_information(self):

        links_folder_path = f'{pybullet_data.getDataPath()}/franka_panda/meshes/collision/'
        try:
            link_file_names = os.listdir(links_folder_path)
        except OSError as e:
            print(f"Error reading files in folder: {e}")

        self.link_index_to_name = ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7',
                                   'hand', 'finger']
        self.link_dimensions_from_name = {}
        
        for file_name in link_file_names:

            if file_name[-4:] == ".obj":
                vertices = []    
                link_name = file_name[:-4]
                with open(links_folder_path + file_name, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('v '):
                            vertex = re.split(r'\s+', line)[1:4]
                            vertex = np.array([float(coord) for coord in vertex])
                            vertices.append(vertex)
                max_point = np.max(np.array(vertices), axis = 0)
                min_point = np.min(np.array(vertices), axis = 0)
                self.link_dimensions_from_name[link_name] = max_point - min_point

        self.link_dimensions = []
        
        for link_index in range(len(self.link_index_to_name)):

            link_name = self.link_index_to_name[link_index]
            link_dimensions = self.link_dimensions_from_name[link_name].copy()
            
            if link_index == len(self.link_index_to_name) - 1:
                link_dimensions[1] *= 4
            
            self.link_dimensions.append(link_dimensions)
        self.link_dimensions = torch.tensor(np.array(self.link_dimensions), dtype = torch.float32, device = self.device)

        self.link_vertices = self.get_link_vertices(self.link_dimensions)

        self.link_static_joint_frame = [1, 2, 3, 4, 5, 6, 7, 7, 7] 
        self.static_frames = []

        # Link 0:
        self.static_frames.append([[1., 0., 0., 8.71e-05],
                                   [0., 1., 0., -3.709035e-02],
                                   [0., 0., 1., -6.851545e-02],
                                   [0., 0., 0., 1.]])
        # Link 1:
        self.static_frames.append([[1., 0., 0., -8.425e-05],
                                   [0., 1., 0., -6.93950016e-02],
                                   [0., 0., 1., 3.71961970e-02],
                                   [0., 0., 0., 1.]])
        
        # Link 2:
        self.static_frames.append([[1., 0., 0., 0.0414576],
                                   [0., 1., 0., 0.0281429],
                                   [0., 0., 1., -0.03293086],
                                   [0., 0., 0., 1.]])

        # Link 3:
        self.static_frames.append([[1., 0., 0., -4.12337575e-02],
                                   [0., 1., 0., 3.44296512e-02],
                                   [0., 0., 1., 2.79226985e-02],
                                   [0., 0., 0., 1.]])

        # Link 4:
        self.static_frames.append([[1., 0., 0., 3.3450000e-05],
                                   [0., 1., 0., 3.7388050e-02],
                                   [0., 0., 1., -1.0619285e-01],
                                   [0., 0., 0., 1.]])

        # Link 5:
        self.static_frames.append([[1., 0., 0., 4.21935000e-02],
                                   [0., 1., 0., 1.52195003e-02],
                                   [0., 0., 1., 6.07699933e-03],
                                   [0., 0., 0., 1.]])

        # Link 6:
        self.static_frames.append([[1., 0., 0., 1.86357500e-02],
                                   [0., 1., 0., 1.85788569e-02],
                                   [0., 0., 1., 7.94137484e-02],
                                   [0., 0., 0., 1.]])

        # Link 7:
        self.static_frames.append([[7.07106767e-01, 7.07106795e-01, 0., -1.26717073e-03],
                                   [-7.07106795e-01, 7.07106767e-01, 0., -1.25294673e-03],
                                   [0., 0., 1., 1.27018693e-01],
                                   [0., 0., 0., 1.]])

        # Link 8:
        self.static_frames.append([[7.07106767e-01, 7.07106795e-01, 0., 9.29352476e-03],
                                   [-7.07106795e-01, 7.07106767e-01, 0., 9.28272434e-03],
                                   [0., 0., 1., 1.92390375e-01],
                                   [0., 0., 0., 1.]])
        
        self.static_frames = torch.tensor(self.static_frames, dtype = torch.float32, device = self.device)
        
    def get_link_transform(self, joints):

        joint_transform = self.forward_kinematics(joints)

        # joint_transform => (batch, traj_len, 9, 4, 4)
        # static_frames => (9, 4, 4)
        link_transform = joint_transform @ self.static_frames.unsqueeze(0).unsqueeze(0).repeat(joints.shape[0], joints.shape[1], 1, 1, 1)

        return link_transform
    
    def cost(self, joint_input, t, batch_size = None):

        self.define_obstacles(self.obstacle_config, t, batch_size = batch_size)
        
        joints = self.rearrange_joints(joint_input)
        # Now joints is (batch, traj_len, 7)
        
        link_transform = self.get_link_transform(joints)

        b = link_transform.shape[0]
        n = link_transform.shape[1]
        nl = link_transform.shape[2]
        no = self.obs_min.shape[1]

        # link_transform => (batch, traj_len, 9, 4, 4)
        # self.link_vertices => (9, 4, 8)
        link_vertices = link_transform @ self.link_vertices.unsqueeze(0).unsqueeze(0).repeat(joints.shape[0], joints.shape[1], 1, 1, 1)
        link_vertices = link_vertices[:, :, :, :3, :]

        # link_vertices => (batch, traj_len, 9, 3, 8)
        link_min = torch.min(link_vertices, dim = -1)[0]
        link_max = torch.max(link_vertices, dim = -1)[0]

        # link_min => (batch, traj_len, 9, 3)
        # link_max => (batch, traj_len, 9, 3)
        expanded_link_min = link_min.unsqueeze(-2).repeat(1, 1, 1, no, 1).view(b, n, no*nl, 3)
        expanded_link_max = link_max.unsqueeze(-2).repeat(1, 1, 1, no, 1).view(b, n, no*nl, 3)

        # obs_min => (total_batch_size, n0, 3)
        # obs_max => (total_batch_size, n0, 3)
        expanded_obs_min = self.obs_min.unsqueeze(1).unsqueeze(1).repeat(1, n, nl, 1, 1).view(b, n, no*nl, 3)
        expanded_obs_max = self.obs_max.unsqueeze(1).unsqueeze(1).repeat(1, n, nl, 1, 1).view(b, n, no*nl, 3)

        overlap_min = torch.max(expanded_link_min, expanded_obs_min)
        overlap_max = torch.min(expanded_link_max, expanded_obs_max)

        overlap_lengths = overlap_max - overlap_min

        volumes = torch.prod(torch.clamp(overlap_lengths, min = 0), dim = -1)
        # volumes => (b, n, no*nl)

        return volumes
    
    def interpolation_cost(self, joint_input, start, goal, t):

        # joint_angles => (b, 7, n)
        # obs_min => (n, 3)
        # obs_max => (n, 3) 
        self.define_obstacles(self.obstacle_config, t)

        joints = self.rearrange_joints(joint_input)
        # Now joints is (batch, traj_len, 7)
        
        joint_trajectory = torch.zeros(size = (joints.shape[0], joints.shape[1] + 2, joints.shape[2]))
        joint_trajectory[:, 0, :] = start.unsqueeze(0).repeat(joints.shape[0], 1)
        joint_trajectory[:, -1, :] = goal.unsqueeze(0).repeat(joints.shape[0], 1)
        joint_trajectory[:, 1:-1, :] = joints
        # Now joint_trajectory is (batch, traj_len + 2, 7)

        interpolated_joints1 = 0.5 * joint_trajectory[:, 1:, :] + 0.5 * joint_trajectory[:, :-1, :]
        interpolated_joints2 = 0.75 * joint_trajectory[:, 1:, :] + 0.25 * joint_trajectory[:, :-1, :]
        interpolated_joints3 = 0.25 * joint_trajectory[:, 1:, :] + 0.75 * joint_trajectory[:, :-1, :]
        
        link_transform1 = self.get_link_transform(interpolated_joints1)
        link_transform2 = self.get_link_transform(interpolated_joints2)
        link_transform3 = self.get_link_transform(interpolated_joints3)

        b = link_transform1.shape[0]
        n = link_transform1.shape[1]
        nl = link_transform1.shape[2]
        no = self.obs_min.shape[0]

        # link_transform => (batch, traj_len + 1, 9, 4, 4)
        # self.link_vertices => (9, 4, 8)
        link_vertices1 = link_transform1 @ self.link_vertices.unsqueeze(0).unsqueeze(0).repeat(interpolated_joints1.shape[0], interpolated_joints1.shape[1], 1, 1, 1)
        link_vertices2 = link_transform2 @ self.link_vertices.unsqueeze(0).unsqueeze(0).repeat(interpolated_joints1.shape[0], interpolated_joints1.shape[1], 1, 1, 1)
        link_vertices3 = link_transform3 @ self.link_vertices.unsqueeze(0).unsqueeze(0).repeat(interpolated_joints1.shape[0], interpolated_joints1.shape[1], 1, 1, 1)
        
        link_vertices1 = link_vertices1[:, :, :, :3, :]
        link_vertices2 = link_vertices2[:, :, :, :3, :]
        link_vertices3 = link_vertices3[:, :, :, :3, :]

        # link_vertices => (batch, traj_len, 9, 3, 8)
        link_min1 = torch.min(link_vertices1, dim = -1)[0]
        link_max1 = torch.max(link_vertices1, dim = -1)[0]
        link_min2 = torch.min(link_vertices2, dim = -1)[0]
        link_max2 = torch.max(link_vertices2, dim = -1)[0]
        link_min3 = torch.min(link_vertices3, dim = -1)[0]
        link_max3 = torch.max(link_vertices3, dim = -1)[0]

        # link_min => (batch, traj_len, 9, 3)
        # link_max => (batch, traj_len, 9, 3)
        expanded_link_min1 = link_min1.unsqueeze(-2).repeat(1, 1, 1, no, 1).view(b, n, no*nl, 3)
        expanded_link_max1 = link_max1.unsqueeze(-2).repeat(1, 1, 1, no, 1).view(b, n, no*nl, 3)
        expanded_link_min2 = link_min2.unsqueeze(-2).repeat(1, 1, 1, no, 1).view(b, n, no*nl, 3)
        expanded_link_max2 = link_max2.unsqueeze(-2).repeat(1, 1, 1, no, 1).view(b, n, no*nl, 3)
        expanded_link_min3 = link_min3.unsqueeze(-2).repeat(1, 1, 1, no, 1).view(b, n, no*nl, 3)
        expanded_link_max3 = link_max3.unsqueeze(-2).repeat(1, 1, 1, no, 1).view(b, n, no*nl, 3)

        expanded_obs_min = self.obs_min.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(b, n, nl, 1, 1).view(b, n, no*nl, 3)
        expanded_obs_max = self.obs_max.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(b, n, nl, 1, 1).view(b, n, no*nl, 3)

        overlap_min1 = torch.max(expanded_link_min1, expanded_obs_min)
        overlap_max1 = torch.min(expanded_link_max1, expanded_obs_max)
        overlap_min2 = torch.max(expanded_link_min2, expanded_obs_min)
        overlap_max2 = torch.min(expanded_link_max2, expanded_obs_max)
        overlap_min3 = torch.max(expanded_link_min3, expanded_obs_min)
        overlap_max3 = torch.min(expanded_link_max3, expanded_obs_max)

        overlap_lengths1 = overlap_max1 - overlap_min1
        overlap_lengths2 = overlap_max2 - overlap_min2
        overlap_lengths3 = overlap_max3 - overlap_min3

        volumes1 = torch.prod(torch.clamp(overlap_lengths1, min = 0), dim = -1)
        volumes2 = torch.prod(torch.clamp(overlap_lengths2, min = 0), dim = -1)
        volumes3 = torch.prod(torch.clamp(overlap_lengths3, min = 0), dim = -1)

        return (volumes1 + volumes2 + volumes3)
    
    def swept_volume_cost(self, joint_input, start, goal, t, batch_size = None):

        # joint_angles => (b, 7, n)
        # obs_min => (n, 3)
        # obs_max => (n, 3) 

        self.define_obstacles(self.obstacle_config, t, batch_size = batch_size)
        
        joints = self.rearrange_joints(joint_input)
        # Now joints is (batch, traj_len, 7)

        joint_trajectory = torch.zeros(size = (joints.shape[0], joints.shape[1] + 2, joints.shape[2]))

        joint_trajectory[:, 0, :] = start.unsqueeze(0).repeat(joints.shape[0], 1)
        if list(goal.size()) != [joint_trajectory.shape[0], joint_trajectory.shape[-1]]:
            joint_trajectory[:, -1, :] = goal.unsqueeze(0).repeat(joints.shape[0], 1)
        else:
            joint_trajectory[:, -1, :] = goal

        joint_trajectory[:, 1:-1, :] = joints
        # Now joint_trajectory is (batch, traj_len + 2, 7)
        
        link_transform = self.get_link_transform(joint_trajectory)

        b = link_transform.shape[0]
        n = link_transform.shape[1]
        nl = link_transform.shape[2]
        no = self.obs_min.shape[1]

        # link_transform => (batch, traj_len+2, 9, 4, 4)
        # self.link_vertices => (9, 4, 8)
        link_vertices = link_transform @ self.link_vertices.unsqueeze(0).unsqueeze(0).repeat(joint_trajectory.shape[0], joint_trajectory.shape[1], 1, 1, 1)
        link_vertices = link_vertices[:, :, :, :3, :]

        # link_vertices => (batch, traj_len+2, 9, 3, 8)
        link_min = torch.min(link_vertices, dim = -1)[0]
        link_max = torch.max(link_vertices, dim = -1)[0]

        # link_min => (batch, traj_len+2, 9, 3)
        # link_max => (batch, traj_len+2, 9, 3)
        link_min_A = link_min[:, :-1, :, :]
        link_max_A = link_max[:, :-1, :, :]
        link_min_B = link_min[:, 1:, :, :]
        link_max_B = link_max[:, 1:, :, :]
        swept_volumes_min = torch.min(link_min_A, link_min_B)
        swept_volumes_max = torch.max(link_max_A, link_max_B)

        # swept_volumes_min => (batch, traj_len+1, 9, 3)
        # swept_volumes_max => (batch, traj_len+1, 9, 3)
        expanded_link_min = swept_volumes_min.unsqueeze(-2).repeat(1, 1, 1, no, 1).view(b, n-1, no*nl, 3)
        expanded_link_max = swept_volumes_max.unsqueeze(-2).repeat(1, 1, 1, no, 1).view(b, n-1, no*nl, 3)

        # obs_min => (total_batch_size, n0, 3)
        # obs_max => (total_batch_size, n0, 3)
        expanded_obs_min = self.obs_min.unsqueeze(1).unsqueeze(1).repeat(1, n-1, nl, 1, 1).view(b, n-1, no*nl, 3)
        expanded_obs_max = self.obs_max.unsqueeze(1).unsqueeze(1).repeat(1, n-1, nl, 1, 1).view(b, n-1, no*nl, 3)

        overlap_min = torch.max(expanded_link_min, expanded_obs_min)
        overlap_max = torch.min(expanded_link_max, expanded_obs_max)

        overlap_lengths = overlap_max - overlap_min

        volumes = torch.prod(torch.clamp(overlap_lengths, min = 0), dim = -1)

        return volumes
    
    def swept_volume_cost_final(self, joint_input, t):

        # joint_angles => (b, 7, n)
        # obs_min => (n, 3)
        # obs_max => (n, 3) 
        self.define_obstacles(self.obstacle_config, t)

        joints = self.rearrange_joints(joint_input)
        # Now joints is (batch, traj_len, 7)

        joint_trajectory = torch.zeros(size = (joints.shape[0], joints.shape[1] + 2, joints.shape[2]))
        # joint_trajectory[:, 0, :] = start.unsqueeze(0).repeat(joints.shape[0], 1)
        # joint_trajectory[:, -1, :] = goal.unsqueeze(0).repeat(joints.shape[0], 1)
        joint_trajectory[:, 1:-1, :] = joints
        # Now joint_trajectory is (batch, traj_len + 2, 7)
        
        link_transform = self.get_link_transform(joint_trajectory)

        b = link_transform.shape[0]
        n = link_transform.shape[1]
        nl = link_transform.shape[2]
        no = self.obs_min.shape[0]

        # link_transform => (batch, traj_len+2, 9, 4, 4)
        # self.link_vertices => (9, 4, 8)
        link_vertices = link_transform @ self.link_vertices.unsqueeze(0).unsqueeze(0).repeat(joint_trajectory.shape[0], joint_trajectory.shape[1], 1, 1, 1)
        link_vertices = link_vertices[:, :, :, :3, :]

        # link_vertices => (batch, traj_len+2, 9, 3, 8)
        link_min = torch.min(link_vertices, dim = -1)[0]
        link_max = torch.max(link_vertices, dim = -1)[0]

        # link_min => (batch, traj_len+2, 9, 3)
        # link_max => (batch, traj_len+2, 9, 3)
        link_min_A = link_min[:, :-1, :, :]
        link_max_A = link_max[:, :-1, :, :]
        link_min_B = link_min[:, 1:, :, :]
        link_max_B = link_max[:, 1:, :, :]
        swept_volumes_min = torch.min(link_min_A, link_min_B)
        swept_volumes_max = torch.max(link_max_A, link_max_B)

        # swept_volumes_min => (batch, traj_len+1, 9, 3)
        # swept_volumes_max => (batch, traj_len+1, 9, 3)
        expanded_link_min = swept_volumes_min.unsqueeze(-2).repeat(1, 1, 1, no, 1).view(b, n-1, no*nl, 3)
        expanded_link_max = swept_volumes_max.unsqueeze(-2).repeat(1, 1, 1, no, 1).view(b, n-1, no*nl, 3)

        expanded_obs_min = self.obs_min.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(b, n-1, nl, 1, 1).view(b, n-1, no*nl, 3)
        expanded_obs_max = self.obs_max.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(b, n-1, nl, 1, 1).view(b, n-1, no*nl, 3)

        overlap_min = torch.max(expanded_link_min, expanded_obs_min)
        overlap_max = torch.min(expanded_link_max, expanded_obs_max)

        overlap_lengths = overlap_max - overlap_min

        volumes = torch.prod(torch.clamp(overlap_lengths, min = 0), dim = -1)

        return volumes

    def get_gradient(self, joint_input, start, goal, t):

        joint_tensor1 = torch.tensor(joint_input, dtype = torch.float32, device = self.device)
        joint_tensor1.requires_grad = True
        joint_tensor1.grad = None

        start = torch.tensor(start, dtype = torch.float32, device = self.device)
        goal = torch.tensor(goal, dtype = torch.float32, device = self.device)
        # start => (7,)
        # goal => (7,)

        # Uses guidance_method type hyperparameter (total_batch_size,)
        # volumes => (b, n, no*nl)
        b = self.batch_size
        guidance_method = torch.tensor(self.guide_cfgs['guidance_method'], dtype = torch.float32, device = self.device).view(b, 1, 1)
        cost = torch.sum((1 - guidance_method) * self.cost(joint_tensor1, t)) + torch.sum(guidance_method * self.swept_volume_cost(joint_tensor1, start, goal, t))
        
        # if self.guidance_method =='iv':
        #     cost = torch.sum(self.cost(joint_tensor1, t)) # Intersection Volume
        # elif self.guidance_method == 'sv':
        #     cost = torch.sum(self.swept_volume_cost(joint_tensor1, start, goal, t)) # Swept Volume
        # elif self.guidance_method == 'ipv':
        #     cost = torch.sum(self.cost(joint_tensor1, t)) + torch.sum(self.interpolation_cost(joint_tensor1, start, goal, t)) # Interpolation Volume
        
        cost.backward()

        gradient1 = joint_tensor1.grad.cpu().numpy()
        # gradient1 -> (b, 7, n)

        # Uses grad_norm hyperparameter:
        grad_norm = self.guide_cfgs['grad_norm'][:, np.newaxis, np.newaxis]
        # grad_norm -> (b,)
        gradient1 = (1 - grad_norm) * gradient1 + grad_norm * (gradient1 / np.linalg.norm(gradient1))

        # if self.grad_norm == True:
        #     if np.linalg.norm(gradient1) > 0:    
        #         gradient1 = gradient1 / np.linalg.norm(gradient1)

        return gradient1
    
    def choose_best_trajectory(self, start, goal, trajectories):

        joint_tensor = torch.tensor(trajectories[:, :, 1:-1], dtype = torch.float32, device = self.device)
        
        start = torch.tensor(start, dtype = torch.float32, device = self.device)
        goal = torch.tensor(goal, dtype = torch.float32, device = self.device)
        
        # joint_angles => (b, 7, n)
        self.define_obstacles(self.obstacle_config, 0)    # Uses clearance and expansion hyperparameters
        
        overlap_volumes = torch.sum(self.swept_volume_cost(joint_tensor, start, goal, 0), dim = (1, 2))
        min_index = torch.argmin(overlap_volumes)

        # print(overlap_volumes)
        # print("best = ", overlap_volumes[min_index])

        return trajectories[min_index] #, min_index
    
    def choose_best_trajectory_final(self, trajectories):

        joint_tensor = torch.tensor(trajectories[:, :, 1:-1], dtype = torch.float32, device = self.device)
        
        # start = torch.tensor(start, dtype = torch.float32, device = self.device)
        # goal = torch.tensor(goal, dtype = torch.float32, device = self.device)
        
        overlap_volumes = torch.sum(self.swept_volume_cost_final(joint_tensor, 0), dim = (1, 2))
        min_index = torch.argmin(overlap_volumes)

        # print(overlap_volumes)
        # print("best = ", overlap_volumes[min_index])

        return trajectories[min_index] #, min_index

    def smoothness_cost(self, joints, start, goal):

        start = torch.tensor(start, dtype = torch.float32, device = self.device).view(1, 7, 1)
        goal = torch.tensor(goal, dtype = torch.float32, device = self.device).view(1, 7, 1)

        cost = torch.sum((joints[:, :, 0] - start)**2) + torch.sum((joints[:, :, 2:-1] - joints[:, :, 1:-2])**2) + torch.sum((goal - joints[:, :, -1])**2)

        return cost




