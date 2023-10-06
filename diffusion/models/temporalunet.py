import torch
import torch.nn as nn
import numpy as np
import os
import torchvision.transforms.functional as tvtf

from diffusion.models.blocks import *

class TemporalUNet(nn.Module):

    def __init__(self, model_name, input_dim, time_dim, device, dims = (32, 64, 128, 256)):

        super(TemporalUNet, self).__init__()

        dims = [input_dim, *dims]  # length of dims is 5

        # Initial Time Embedding:
        self.time_embedding = TimeEmbedding(time_dim, device)

        # Down Sampling:
        self.down_samplers = nn.ModuleList([])
        for i in range(len(dims) - 2):      # Loops 0, 1, 2
            self.down_samplers.append(DownSampler(dims[i], dims[i+1], time_dim))
        self.down_samplers.append(DownSampler(dims[-2], dims[-1], time_dim, is_last = True))  # 3 -> 4

        # Middle Block:
        self.middle_block = MiddleBlock(dims[-1], time_dim)

        # Up Sampling:
        self.up_samplers = nn.ModuleList([])
        for i in range(len(dims) - 1, 1, -1):  # Loops 4, 3, 2  since the last one is a seperate convolution
            self.up_samplers.append(UpSampler(dims[i-1], dims[i], time_dim))

        # Final Convolution:
        self.final_conv = nn.Sequential(Conv1dBlock(dims[1], dims[1], kernel_size = 5),
                                        nn.Conv1d(dims[1], input_dim, kernel_size = 1))
        
        self.model_name = model_name
        if not os.path.exists(model_name):
            os.mkdir(model_name)
            self.losses = np.array([])
        else:
            self.load()

        _ = self.to(device)

    def forward(self, x, t):
        """
        x => Tensor of size (batch_size, traj_len*2)
        t => Integer representing the diffusion timestep of x
        """
        
        # Get the time embedding from t:
        time_emb = self.time_embedding(t)

        # Down Sampling Layers:
        h_list = []
        for i in range(len(self.down_samplers)):
            x, h = self.down_samplers[i](x, time_emb)
            h_list.append(h)

        # Middle Layer:
        x = self.middle_block(x, time_emb)

        # Up Sampling Layers:
        for i in range(len(self.up_samplers)):
            h_temp = h_list.pop()
            # print(f"Shape of x: {x.shape}\t Shape of h_list: {h_temp.shape}")
            x = self.up_samplers[i](x, h_temp, time_emb)   # How does pop work and not h_list[i]
            if x.shape[2] == 8 or x.shape[2] == 14 or x.shape[2] == 26 or x.shape[2] == 8: # Upsampling doubles the dimensions of the input. So, we are manually cropping the extra size to match the size of it's corresponding h (context/residual from the downsampling layer)
                x = tvtf.crop(x, 0, 0, x.shape[1], x.shape[2] - 1)

        # Final Convolution
        out = self.final_conv(x)

        return out

    def save(self):

        torch.save(self.state_dict(), self.model_name + "/weights_latest.pt")
        np.save(self.model_name + "/losses.npy", self.losses)

    def save_checkpoint(self, checkpoint):
        
        torch.save(self.state_dict(), self.model_name + "/weights_" + str(checkpoint) + ".pt")
        np.save(self.model_name + "/latest_checkpoint.npy", checkpoint)
    
    def load(self):

        self.losses = np.load(self.model_name + "/losses.npy")
        self.load_state_dict(torch.load(self.model_name + "/weights_latest.pt"))
        print("Loaded Model at " + str(self.losses.size) + " epochs")

    def load_checkpoint(self, checkpoint):

        _ = input("Press Enter if you are running the model for inference, or Ctrl+C\n(Never load a checkpoint for training! This will overwrite progress)")
        
        latest_checkpoint = np.load(self.model_name + "/latest_checkpoint.npy")
        self.load_state_dict(torch.load(self.model_name + "/weights_" + str(checkpoint) + ".pt"))
        self.losses = np.load(self.model_name + "/losses.npy")[:checkpoint]

    

            