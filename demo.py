import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from data_processing import *
from graph_NN_simulator import *
import yaml

'''
This code is a PyTorch implementation of a graph neural network (GNN) simulator for particle dynamics, specifically designed to simulate the motion of particles in a 2D space. 
The GNN simulator is based on the Interaction Network architecture and is trained using a dataset of particle trajectories. 
The code started off as basically a copy from a student capstone project 
https://colab.research.google.com/drive/1hirUfPgLU35QCSQSZ7T2lZMyFMOaK_OF?usp=sharing#scrollTo=ZoB4_A6YJ7FP
which is also described in this Medium article: Simulating Complex Physics with Graph Networks: step by step
https://medium.com/stanford-cs224w/simulating-complex-physics-with-graph-networks-step-by-step-177354cb9b05
'''

TYPE_TO_COLOR = {
    3: "black",  # Boundary particles.
    0: "green",  # Rigid solids.
    7: "magenta",  # Goop.
    6: "gold",  # Sand.
    5: "blue",  # Water.
}

def visualize_prepare(ax, particle_type, position, metadata):
    bounds = metadata["bounds"]
    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0)
    points = {type_: ax.plot([], [], "o", ms=2, color=color)[0] for type_, color in TYPE_TO_COLOR.items()}
    return ax, position, points


def visualize_pair(particle_type, position_pred, position_gt, metadata):
    two_frames = True
    if position_gt.shape[1] < position_pred.shape[1]:
        position_gt = position_gt[:position_pred.shape[1],:]
    if position_gt.shape[1] < 100:
        two_frames = False
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if two_frames:
        plot_info = [
            visualize_prepare(axes[0], particle_type, position_gt, metadata),
            visualize_prepare(axes[1], particle_type, position_pred, metadata),
        ]
        axes[0].set_title("Ground truth")
        axes[1].set_title("Prediction")
    else:
        plot_info = [
            visualize_prepare(axes[0], particle_type, position_pred, metadata),
            visualize_prepare(axes[1], particle_type, position_pred, metadata),
        ]
        axes[0].set_title("Prediction")
        axes[1].set_title("Prediction")

    def update(step_i):
        outputs = []
        for _, position, points in plot_info:
            for type_, line in points.items():
                mask = particle_type == type_
                line.set_data(position[mask, step_i, 0], position[mask, step_i, 1])
            outputs.append(line)
        return outputs

    return animation.FuncAnimation(fig, update, frames=np.arange(0, position_gt.size(1)), interval=1, blit=True)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    params = config["training"]
    model_params = config["model"]
    data_path = config['data']['data_path']
    model_path = config['data']['model_path']

    # build model
    simulator = LearnedSimulator(hidden_size=model_params["hidden_size"], 
                                    n_mp_layers=model_params["n_mp_layers"], 
                                    window_size=model_params["window_size"])
    simulator = simulator.to(device)

    checkpoint = torch.load(os.path.join(model_path, "2025-04-01_21_35_checkpoint_175000.pt"))
    simulator.load_state_dict(checkpoint["model"])
    rollout_dataset = RolloutDataset(data_path, "valid")
    simulator.eval()

    rollout_data = rollout_dataset[1]
    rollout_out = rollout(simulator, rollout_data, rollout_dataset.metadata, params["noise"])

    anim = visualize_pair(rollout_data["particle_type"], rollout_out, rollout_data["position"], rollout_dataset.metadata)
    plt.show()