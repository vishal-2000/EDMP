import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

class TrajectoryInspector:
    def __init__(self, pkl_path):
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"File {pkl_path} not found.")
        
        self.pkl_path = pkl_path
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Loaded data from {pkl_path}")
        
    def plot_gradient_norm(self):
        """
        Extracts gradient history from the pickle data and plots Norm vs Time.
        """
        if 'gradient_history' not in self.data:
            print("No 'gradient_history' found in this pickle file.")
            return
        
        grad_hist = self.data['gradient_history']
        
        if not grad_hist:
            print("Gradient history is empty.")
            return

        # Sort by timestep (descending usually, but we want plot 0..T or T..0)
        # Diffusion goes T -> 0. Let's plot T on x-axis reversed or just step index.
        # Plotting Step Number (x) vs Gradient Norm (y). 
        # Since steps go 250 -> 0, let's sort them so x-axis is 250, ..., 0 or 250->0 order.
        # Usually easier to read if X axis is "Time Remaining" or "Diffusion Step". 
        
        timesteps = sorted(grad_hist.keys(), reverse=True) # 250, 249... 0
        norms = [grad_hist[t] for t in timesteps]
        
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, norms, marker='o', linestyle='-')
        plt.xlabel('Diffusion Step (t)')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm vs. Diffusion Step')
        plt.grid(True)
        plt.gca().invert_xaxis() # Show T -> 0 as process progresses left to right? 
                                 # Or just standard X axis. In diffusion T->0 is "progress".
                                 # Let's keep T on X axis, but maybe invert it so it feels like "Time passes".
                                 # Or just keep it 250 -> 0 (standard x axis decreasing).
        
        plt.show()
        print("Plot displayed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect EDMP trajectory pickle files.")
    parser.add_argument('-f', '--file', type=str, required=True, help="Path to .pkl file")
    
    args = parser.parse_args()
    
    inspector = TrajectoryInspector(args.file)
    inspector.plot_gradient_norm()
