import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from data_processing import *
import yaml
import importlib
import matplotlib.patches as patches

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    model_version = config["model"]["model_version"]
graph_model = importlib.import_module(f"{model_version}.graph_NN_simulator")

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

def visualize_prepare(ax, particle_type, position, metadata, margin=0.0):
    bounds = metadata["bounds"]
    ax.set_xlim(bounds[0][0]-margin, bounds[0][1]+margin)
    ax.set_ylim(bounds[1][0]-margin, bounds[1][1]+margin)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1.0)
    rect = patches.Rectangle((bounds[0][0], bounds[1][0]), bounds[0][1]-bounds[0][0], bounds[1][1]-bounds[1][0], 
                             linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    points = {type_: ax.plot([], [], "o", ms=2, color=color)[0] for type_, color in TYPE_TO_COLOR.items()}
    return ax, position, points


def visualize_pair(particle_type, position_pred, position_gt, metadata, frames = 2):
    if position_gt.shape[1] < position_pred.shape[1]:
        position_gt = position_gt[:position_pred.shape[1],:]
    if frames==2:
        fig, axes = plt.subplots(1, frames, figsize=(5*frames, 5))
        plot_info = [
            visualize_prepare(axes[0], particle_type, position_gt, metadata),
            visualize_prepare(axes[1], particle_type, position_pred, metadata),
        ]
        axes[0].set_title("Ground truth")
        axes[1].set_title("Prediction")
    else:
        fig, axes = plt.subplots(1, frames, figsize=(7*frames, 5))
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.01)
        plot_info = [
            visualize_prepare(axes, particle_type, position_pred, metadata),
        ]

    def update(step_i):
        outputs = []
        for _, position, points in plot_info:
            for type_, line in points.items():
                mask = particle_type == type_
                line.set_data(position[mask, step_i, 0], position[mask, step_i, 1])
            outputs.append(line)
        return outputs

    return animation.FuncAnimation(fig, update, frames=np.arange(0, position_gt.size(1), 3), interval=0.1, blit=True)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    params = config["training"]
    model_params = config["model"]
    data_path = config['data']['data_path']
    model_path = config['data']['model_path']
    rollout_path = config['data']['rollout_path']

    # build model
    simulator = graph_model.LearnedSimulator(hidden_size=model_params["hidden_size"], 
                                    n_mp_layers=model_params["n_mp_layers"], 
                                    window_size=model_params["window_size"])
    simulator = simulator.to(device)

    checkpoint = torch.load(os.path.join(model_path, "2025-04-07_16_46_checkpoint_62000.pt"))
    # checkpoint2 = torch.load(os.path.join(model_path, "2025-04-06_20_44_checkpoint_5000.pt"))
    # print(checkpoint["model"].keys())
    # for key in checkpoint["model"].keys():
    #     if key.startswith("node_in1") or key.startswith("node_out1") or key.startswith("edge_in1") or key.startswith("layers1"):
    #         checkpoint["model"][key] = checkpoint2["model"][key]
    
    simulator.load_state_dict(checkpoint["model"])
    name = "poster"
    rollout_dataset = RolloutDataset(data_path, name)
    simulator.eval()

    rollout_data = rollout_dataset[0]
    rollout_start = 0
    rollout_out = rollout(simulator, rollout_data, rollout_dataset.metadata, params["noise"], rollout_start=rollout_start, no_leak=True, rollout_length=100)
    length_ = rollout_out.size(1)
    cropped_rollout_data_pos = rollout_data["position"][:,rollout_start:rollout_start+length_,:]

    first_frame = cropped_rollout_data_pos[:,0,:].unsqueeze(1).repeat(1, 100, 1)
    cropped_rollout_data_pos = torch.cat([first_frame, cropped_rollout_data_pos], dim=1)

    first_frame = rollout_out[:,0,:].unsqueeze(1).repeat(1, 100, 1)
    rollout_out = torch.cat([first_frame, rollout_out], dim=1)

    anim = visualize_pair(rollout_data["particle_type"], rollout_out, cropped_rollout_data_pos, rollout_dataset.metadata, frames = 1)
    anim.save(os.path.join(rollout_path, name+".gif"), writer=PillowWriter(fps=30))  # adjust fps as needed
    plt.show()