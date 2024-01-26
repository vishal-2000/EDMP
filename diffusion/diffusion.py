import numpy as np
import torch
import copy
import time

from .gaussian import Gaussian

class Diffusion(Gaussian):
    
    def __init__(self, T, device, variance_thresh = 0.02):

        self.T = T
        self.beta = self.schedule_variance(variance_thresh)

        self.alpha = 1 - self.beta
        self.alpha_bar = np.array(list(map(lambda t:np.prod(self.alpha[:t]), np.arange(self.T+1)[1:])))

        self.device = device

        Gaussian.__init__(self)
    
    def cosine_func(self, t):
        """
        Implements the cosine function at a timestep t
        Inputs:
        t -> Integer ; current timestep
        T -> Integer ; total number of timesteps
        Outputs:
        out -> Float ; cosine function output
        """
        
        s = 1e-10
        out = np.cos((t/self.T + s) * (np.pi/2) / (1 + s)) ** 0.15
        
        return out

    def schedule_variance(self, thresh = 0.02):
        """
        Schedules the variance for the diffuser
        Inputs:
        T      -> Integer ; total number of timesteps
        thresh -> Float ; variance threshold at the last step
        Outputs:
        schedule -> Numpy array of shape (2,) ; the variance schedule
        """
            
        schedule = np.linspace(0, thresh, self.T + 1)[1:]

        return schedule

    def q_sample(self, x, t, eps = None):
        """
        Generates q(xt+1/xt)
        Inputs:
        x0 -> Numpy array of shape (num_samples, num_channels, trajectory_length)
        t -> Numpy array of length (num_samples, ); timesteps (these are the output timesteps)
        Outputs:
        xt -> Numpy array of shape (num_samples, num_channels, trajectory_length)
        mean -> Numpy array of shape (num_samples, num_channels, trajectory_length); mean from which sample is taken
        var-> Numpy array of length (num_samples); variance from which sample is taken
        """

        # Gather trajectory params:
        b = x.shape[0]
        c = x.shape[1]
        n = x.shape[2]
        
        if type(eps) == type(None):
            eps = np.random.multivariate_normal(mean = np.zeros((n*c)), 
                                                cov = np.eye(n*c),
                                                size = (b,)).reshape(b, c, n)
        
        xt = np.sqrt(self.alpha[t-1, np.newaxis, np.newaxis]) * x + np.sqrt(1 - self.alpha[t-1, np.newaxis, np.newaxis]) * eps
        mean = (np.sqrt(self.alpha[t-1, np.newaxis, np.newaxis]) * x)
        var = np.sqrt(1 - self.alpha[t-1])

        return xt, mean, var
    
    def q_sample_from_x0(self, x0, t, eps = None):
        """
        Generates q(xt/x0)
        Inputs:
        x0 -> Numpy array of shape (num_samples, num_channels, trajectory_length)
        t -> Numpy array of length (num_samples, ); timesteps (these are the output timesteps)
        Outputs:
        xt -> Numpy array of shape (num_samples, num_channels, trajectory_length)
        mean -> Numpy array of shape (num_samples, num_channels, trajectory_length); mean from which sample is taken
        var-> Numpy array of length (num_samples); variance from which sample is taken
        """

        # Gather trajectory params:
        b = x0.shape[0]
        c = x0.shape[1]
        n = x0.shape[2]
        
        if type(eps) == type(None):
            eps = np.random.multivariate_normal(mean = np.zeros((n*c)), 
                                                cov = np.eye(n*c),
                                                size = (b,)).reshape(b, c, n)
        
        xt = np.sqrt(self.alpha_bar[t-1, np.newaxis, np.newaxis]) * x0 + np.sqrt(1 - self.alpha_bar[t-1, np.newaxis, np.newaxis]) * eps
        mean = (np.sqrt(self.alpha_bar[t-1, np.newaxis, np.newaxis]) * x0)
        var = np.sqrt(1 - self.alpha_bar[t-1, np.newaxis, np.newaxis])

        return xt, mean, var
    
    def p_sample(self, xt, t, eps):
        """
        Generates reverse probability p(xt-1/xt, eps) with no dependence on x0.
        """

        xt_prev = (xt - np.sqrt(1 - self.alpha[t-1]) * eps) / np.sqrt(self.alpha[t-1])

        return xt_prev
    
    def p_sample_using_posterior(self, xt, t, eps):
        """
        Generates reverse probability p(xt-1/xt, x0) using posterior mean and posterior variance.
        """        
        
        # Gather trajectory params:
        b = xt.shape[0]
        c = xt.shape[1]
        n = xt.shape[2]

        z = np.random.multivariate_normal(mean = np.zeros(n), cov = np.eye(n), size = (b, c))
        z[np.where(t == 1), :, :] = 0

        alpha = self.alpha[t-1, np.newaxis, np.newaxis]
        alpha_bar = self.alpha_bar[t-1, np.newaxis, np.newaxis]
        beta = self.beta[t-1, np.newaxis, np.newaxis]

        xt_prev = (xt - ((1 - alpha) / np.sqrt(1 - alpha_bar)) * eps) / np.sqrt(alpha) + beta*z

        return xt_prev.copy()

    def forward_diffuse(self, x0, condition = True):
        """
        Forward diffuses the trajectory till the last timestep T
        Inputs:
        x -> Numpy array of shape (num_samples, num_channels, trajectory_length)
        Outputs:
        diffusion_trajs -> Numpy array of shape (Timesteps, num_samples, num_channels, trajectory_length); diffused trajectories
        eps -> Numpy array of shape (Timesteps, num_samples, num_channels, trajectory_length); Noise added at each timestep
        kl_divs -> Numpy array of shape (Timesteps, num_samples); KL Divergence at each timestep
        """
                
        # Gather trajectory params:
        b = x0.shape[0]
        c = x0.shape[1]
        n = x0.shape[2]
        
        # Initialize the diffusion trajectories across time steps:
        diffusion_trajs = np.zeros((self.T + 1, b, c, n))
        diffusion_trajs[0] = x0.copy()

        # Calculate epsilons to forward diffuse current trajectory:
        mean = np.zeros(n*c)
        cov = np.eye(n*c)
        eps = np.random.multivariate_normal(mean, cov, size = (self.T, b))
        kl_divs = np.zeros((self.T, b))

        # ??? This loop can probably be vectorized
        
        for t in range(1, self.T + 1):

            diffusion_trajs[t] = self.q_sample(diffusion_trajs[t-1], t, eps[t-1])
            if condition:
                diffusion_trajs[t, :, :, 0] = x0[:, :, 0].copy()
                diffusion_trajs[t, :, :, -1] = x0[:, :, -1].copy()
            
            # ???
            kl_divs[t-1, :] = self.KL_divergence_against_gaussian(diffusion_trajs[t].flatten())

        return diffusion_trajs.copy(), eps.copy(), kl_divs.copy()

    def reverse_diffuse(self, xT, eps):
        """
        Reverse diffuses the trajectory from last timestep T to first timestep 0
        Inputs:
        xT  -> Numpy array of shape (num_samples, num_channels, trajectory_length)
        eps -> Numpy array of shape (Timesteps, num_samples, num_channels, trajectory_length); Noise added at each timestep
        Outputs:
        diffusion_trajs -> Numpy array of shape (Timesteps, num_samples, num_channels, trajectory_length); reverse diffused trajectories
        """
        
        # Gather trajectory params:
        b = xT.shape[0]
        c = xT.shape[1]
        n = xT.shape[2]
        
        diffusion_trajs = np.zeros((self.T + 1, b, c, n))
        diffusion_trajs[self.T] = xT.copy()

        for t in range(self.T, 0, -1):

            diffusion_trajs[t-1] = self.p_sample(diffusion_trajs[t], t, eps[t-1])

        return diffusion_trajs.copy()

    def generate_q_sample(self, x0, time_steps = None, condition = True, return_type = "tensor"):
        """
        Generates q samples for a random set of timesteps, useful for training\n\n
        Inputs:\n
        x0          -> Numpy array of shape (num_samples, num_channels, trajectory_length)\n
        condition   -> Bool ; whether to apply conditioning or not\n
        return_type -> String (either "tensor" or "numpy") ; whether to return a tensor or numpy array\n\n
        Outputs:\n
        X -> Numpy array of shape (num_samples, num_channels, trajectory_length); xt from each x0 diffused to a random timestep t\n
        Y -> Numpy array of shape (num_samples, num_channels, trajectory_length); the noise added to each x0\n
        time_step -> Numpy array of shape (num_samples, ); the timestep to which each x0 is diffused\n
        means -> Numpy array of shape (num_samples, num_channels, trajectory_length); mean of each x0 diffused to a random timestep t\n
        vars -> Numpy array of shape (num_samples, ); the variance at the timestep to which x0 is diffused\n
        """

        # Refer to the Training Algorithm in Ho et al 2020 for the psuedo code of this function
            
        # Gather trajectory params:
        b = x0.shape[0]
        c = x0.shape[1]
        n = x0.shape[2]
        
        if type(time_steps) == type(None):
            time_steps = np.random.randint(1, self.T + 1, size = (b, ))

        # Remember, for each channel, we get a multivariate normal distribution.
        eps = np.random.multivariate_normal(mean = np.zeros((n,)), cov = np.eye(n), size = (b, c))
        
        # Size chart:
        # x0         => (num_samples, 2, traj_len)
        # xt         => (num_samples, 2, traj_len)
        # alpha_bar  => (num_samples, 1, 1)
        # eps        => (num_samples, 2, traj_len)
        
        xt, means, vars = self.q_sample_from_x0(x0, time_steps, eps)  

        # CONDITIONING:
        if condition:
            xt[:, :, 0] = x0[:, :, 0].copy()
            xt[:, :, -1] = x0[:, :, -1].copy()

        # Type Conversion:
        if return_type == "tensor":
            X = torch.tensor(xt, dtype = torch.float32)
            Y = torch.tensor(eps, dtype = torch.float32)
            time_steps = torch.tensor(time_steps, dtype = torch.float32)
        elif return_type == "numpy":
            X = xt.copy()
            Y = eps.copy()

        return X, Y, time_steps, means, vars
    
    def denoise(self, model, traj_len, num_channels, start = None, goal = None, condition = True):
        
        X_t = np.random.multivariate_normal(mean = np.zeros(traj_len), cov = np.eye(traj_len), size = (1, num_channels))

        if condition:
            X_t[:, :, 0] = start[:]
            X_t[:, :, -1] = goal[:]

        model.train(False)

        for t in range(self.T, 0, -1):

            X_input = torch.tensor(X_t, dtype = torch.float32).to(self.device)
            time_in = torch.tensor([t], dtype = torch.float32).to(self.device)

            epsilon = model(X_input, time_in).numpy(force=True)

            X_t = self.p_sample_using_posterior(X_t, t, epsilon)

            if condition:
                X_t[:, :, 0] = start[:]
                X_t[:, :, -1] = goal[:]

            print("\rDenoised " + str(t-1) + " timesteps", end="", flush=True)

        return X_t[0]
    
    def clip_joints(self, joints):

        joint_lower_limits = np.array([-166*(np.pi/180), 
                                           -101*(np.pi/180), 
                                           -166*(np.pi/180), 
                                           -176*(np.pi/180),
                                           -166*(np.pi/180),
                                           -1*(np.pi/180),
                                           -166*(np.pi/180)])
        
        joint_upper_limits = np.array([166*(np.pi/180), 
                                           101*(np.pi/180), 
                                           166*(np.pi/180), 
                                           -4*(np.pi/180),
                                           166*(np.pi/180),
                                           215*(np.pi/180),
                                           166*(np.pi/180)])

        return np.clip(joints, joint_lower_limits[np.newaxis, :, np.newaxis], joint_upper_limits[np.newaxis, :, np.newaxis])
    
    def denoise_guided(self, model, guide, traj_len, num_channels, guidance_schedule, batch_size = 1, start = None, goal = None, 
                       condition = True, benchmarking = False):
        
        X_t = np.random.multivariate_normal(mean = np.zeros(traj_len), cov = np.eye(traj_len), size = (batch_size, num_channels))

        if condition:
            X_t[:, :, 0] = start[:]
            X_t[:, :, -1] = goal[:]

        model.train(False) 

        period = 2

        # full_info = np.zeros((self.T+1, batch_size, 7, 50))
        for t in range(self.T, 0, -1):
            print(f"\rDenoising: " + str(t) + " ", end="")

            # full_info[t, :, : ,:] = X_t.copy()  # copy.deepcopy(X_t)

            X_input = torch.tensor(X_t, dtype = torch.float32).to(self.device)
            time_in = torch.tensor([t], dtype = torch.float32).to(self.device)

            epsilon = model(X_input, time_in).numpy(force=True)

            X_t = self.p_sample_using_posterior(X_t, t, epsilon)

            if (t%period) < (period/2):
                if t >= 5:
                    clipped_joints = self.clip_joints(X_t[:, :, 1:-1])
                    # st = time.time()
                    gradient = guide.get_gradient(clipped_joints, start[:], goal[:], t)
                    # print(f"\nGradient time: {time.time() - st}")
                    
                    # if guidance_schedule == 'constant':
                    #     X_t[:, :, 1:-1] = X_t[:, :, 1:-1] - guidance_scale * gradient
                    # elif guidance_schedule == 'varying':
                    #     # Remember to index t-1 while parallelizing
                    #     X_t[:, :, 1:-1] = X_t[:, :, 1:-1] - (1.4 + (t/self.T)) * gradient
                    # else:
                    #     raise NotImplementedError("This type of guidance schedule is not implemented!!")
                    
                    X_t[:, :, 1:-1] = X_t[:, :, 1:-1] - guidance_schedule[:, t-1, np.newaxis, np.newaxis] * gradient

                # weight = np.clip(np.log(1+((t-2)/self.T)*(np.exp(1) - 1)), 0.005, 1) * 2.2
                # X_t[:, :, 1:-1] = X_t[:, :, 1:-1] - weight * gradient
                # X_t[:, :, 1:-1] = X_t[:, :, 1:-1] - 2. * gradient

            if condition:
                X_t[:, :, 0] = start[:]
                X_t[:, :, -1] = goal[:]

            if not benchmarking:
                print("\rDenoised " + str(t-1) + " timesteps", end="", flush=True)

        # full_info[0, :, :, :] = X_t.copy()

        return X_t.copy() #, full_info.copy()
        
    



