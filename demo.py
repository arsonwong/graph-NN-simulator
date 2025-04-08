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

def create_animation(data_path, rollout_data_name, rollout_path, example_index=0, rollout_start=0, rollout_length=500, modifiers=None, show=True, suffix=None):
    rollout_dataset = RolloutDataset(data_path, rollout_data_name)
    rollout_data = rollout_dataset[example_index]
    rollout_out = rollout(simulator, rollout_data, rollout_dataset.metadata, params["noise"], rollout_start=rollout_start, no_leak=True, rollout_length=rollout_length, modifiers=modifiers)
    length_ = rollout_out.size(1)
    cropped_rollout_data_pos = rollout_data["position"][:,rollout_start:rollout_start+length_,:]

    first_frame = cropped_rollout_data_pos[:,0,:].unsqueeze(1).repeat(1, 100, 1)
    cropped_rollout_data_pos = torch.cat([first_frame, cropped_rollout_data_pos], dim=1)

    first_frame = rollout_out[:,0,:].unsqueeze(1).repeat(1, 100, 1)
    rollout_out = torch.cat([first_frame, rollout_out], dim=1)

    anim = visualize_pair(rollout_data["particle_type"], rollout_out, cropped_rollout_data_pos, rollout_dataset.metadata, frames = 1)
    name = rollout_data_name
    if suffix is not None:
        name += "_" + suffix
    anim.save(os.path.join(rollout_path, name+".gif"), writer=PillowWriter(fps=30))  # adjust fps as needed
    if show:
        plt.show()

def compile_gifs():
    names = ['fadeaway','poster','grate','bullet','trash','valid']
    indices = [0,0,0,0,0,1]
    viscosity_multiplier = [1, 1/1.4, 1,3,1]
    integrity_multiplier = [1,1,0.5,1000,1]
    gravity_multiplier = [1,1,1,1,0]
    scenarios = ["normal","fluid","sandy","blob","zero gravity"]
    for i, name in enumerate(names):
        for j, scenario in enumerate(scenarios):
            modifiers = {"viscosity_multiplier":viscosity_multiplier[j],
                         "integrity_multiplier":integrity_multiplier[j],
                         "gravity_multiplier":gravity_multiplier[j]}
            create_animation(data_path, name, rollout_path, example_index=indices[i], modifiers=modifiers, show=False, suffix=scenario, rollout_length=500)

if __name__ == '__main__':
    name = "valid"
    example_index=67

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    load_model_path = config["data"]['load_model_path']
    params = config["training"]
    model_params = config["model"]
    data_path = config['data']['data_path']
    model_path = config['data']['model_path']
    rollout_path = config['data']['rollout_path']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    simulator = graph_model.LearnedSimulator(hidden_size=model_params["hidden_size"], 
                                    n_mp_layers=model_params["n_mp_layers"], 
                                    window_size=model_params["window_size"])
    simulator = simulator.to(device)

    checkpoint = torch.load(os.path.join(model_path, load_model_path))
    simulator.eval()
    simulator.load_state_dict(checkpoint["model"])
    create_animation(data_path, name, rollout_path, example_index=example_index)
    
    
    
    

    