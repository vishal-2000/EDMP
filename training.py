import torch
import torch.nn as nn
import time
import numpy as np
import os

from scripts.diffusion_functions import *
from Models.temporalunet import TemporalUNet
from scripts.nvidia_MPI_dataset import Traj_train_dataset

import wandb


device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------- Edit these parameters -------------- #
# from priors.bezier_curve_N_points_2d import *

traj_len = 50
T = 256 # NUmber of diffusion time steps
epochs = 50000
batch_size = 2048

dataset = Traj_train_dataset(dataset_path='/scratch/jayaram_reddy/mpinet_dataset/train.hdf5', n_diffusion_steps=T)
model_name = "/scratch/jayaram_reddy/mpinet_models/Model_weights/7dof256/" + "TemporalUNetModel" + str(T) + "_N" + str(traj_len)
# --------------------------------------------------- #
model_time_dim = 32
model_dims = (32, 64, 128, 256, 512, 512)
denoiser = TemporalUNet(model_name = model_name, input_dim = 7, time_dim = model_time_dim, dims = model_dims) # state size: 7; trajectory size: 7*50
_ = denoiser.to(device)

optimizer = torch.optim.Adam(denoiser.parameters(), lr = 0.0001)
loss_fn = nn.MSELoss()

# WANDB setup

number = 1
NAME = "7DOF dip 256 both" + str(number)
ID = 'Traj_dip_256_both' + str(number)
run = wandb.init(project='7DOF_denoiser_64', name = NAME, id = ID)

wandb.config.update({
    'max_epochs': epochs,
    'batch_size': 2048,
    'Max number of timesteps in diffusion': T,
    'Trajectory length': traj_len,
    'State dimension': 7,
    'model type': "Temporal UNet",
    'model time dim': model_time_dim,
    'model dims': model_dims,
    'Dataset size': '3M + 3M'
})

for e in range(epochs):
            
    denoiser.train(True)

    start = time.time()
    X, Y_true, t = dataset.generate_training_batch(batch_size=batch_size)
    # print(f"Sample gen: {time.time() - start}")
    # start = time.time()
    
    X = X.to(device)
    Y_true = Y_true.to(device)
    t = t.to(device)

    Y_pred = denoiser(X, t)
    # print(f"Denoise: {time.time() - start}")
    # start = time.time()

    optimizer.zero_grad()

    loss = loss_fn(Y_pred, Y_true)

    loss.backward()
    # print(f"Loss backpropagate: {time.time() - start}")
    start = time.time()

    optimizer.step()
    print(f"Optimizer Step: {time.time() - start}")
    start = time.time()

    denoiser.losses = np.append(denoiser.losses, loss.item()) # Time consuming step (Takes about 30 seconds on avg, where as all the other steps take about 5 secs at max) - Unnecessary
    print(f"Denoiser loss apend: {time.time() - start}")

    print(f"\rEpoch number = {e}, Current Loss: {loss.item()}")

    wandb.log({'epoch': e, 'loss': loss.item()}) 

    denoiser.save()
    if e % 1000 == 0:
        denoiser.save_checkpoint(e)
    
