
import torch
import numpy as np
import torch_geometric as pyg
import pickle
from tqdm import tqdm
from graph_NN_simulator import KINEMATIC_PARTICLE_ID, preprocess, rotate, read_metadata

'''
This code is a PyTorch implementation of a graph neural network (GNN) simulator for particle dynamics, specifically designed to simulate the motion of particles in a 2D space. 
The GNN simulator is based on the Interaction Network architecture and is trained using a dataset of particle trajectories. 
The code started off as basically a copy from a student capstone project 
https://colab.research.google.com/drive/1hirUfPgLU35QCSQSZ7T2lZMyFMOaK_OF?usp=sharing#scrollTo=ZoB4_A6YJ7FP
which is also described in this Medium article: Simulating Complex Physics with Graph Networks: step by step
https://medium.com/stanford-cs224w/simulating-complex-physics-with-graph-networks-step-by-step-177354cb9b05
'''


class OneStepDataset(pyg.data.Dataset):
    def __init__(self, data_path, split, window_length=7, noise_std=0.0, return_pos=False, random_rotation=False):
        super().__init__()

        # load dataset from the disk
        metadata = read_metadata(data_path)
        self.metadata = metadata
        with open(f"{data_path}/{split}.pkl", "rb") as f:
            examples = pickle.load(f)
        self.data = examples
        self.window_length = window_length
        self.noise_std = noise_std
        self.return_pos = return_pos
        self.random_rotation = random_rotation

        # cut particle trajectories according to time slices
        self.windows = np.array([])
        for i, example in enumerate(examples):
            position_seq = example["position_seq"]
            target_position = example["target_position"]
            # target position is just one extended from the last, so add it
            position_seq = np.concatenate([position_seq, target_position[:, np.newaxis, :]], axis=1)
            self.data[i]["position_seq"] = position_seq
            position_seq_size = position_seq.shape[1]
            num_windows = position_seq_size-window_length+1
            entry = np.array([i*np.ones((num_windows,)), np.arange(num_windows)]).T
            if i == 0:
                self.windows = entry
            else:
                self.windows = np.vstack([self.windows,entry])
        self.windows = self.windows.astype(int)

    def len(self):
        return len(self.windows)

    def get(self, idx):
        # load corresponding data for this time slice
        window = self.windows[idx]
        which_example = window[0]
        index = window[1]
        particle_type = self.data[which_example]["particle_type"].copy()
        particle_type = torch.from_numpy(particle_type)
        position_seq = self.data[which_example]["position_seq"][:,index:index+self.window_length,:].copy()
        target_position = position_seq[:, -1]
        position_seq = position_seq[:, :-1]
        target_position = torch.from_numpy(target_position)
        position_seq = torch.from_numpy(position_seq)
        
        # construct the graph
        with torch.no_grad():
            rotation = 0
            if self.random_rotation:
                rotation = np.random.randint(0, 4)
            graph = preprocess(particle_type, position_seq, target_position, self.metadata, self.noise_std, rotation=rotation)
        return graph

class RolloutDataset(pyg.data.Dataset):
    def __init__(self, data_path, split):
        super().__init__()

        # load dataset from the disk
        metadata = read_metadata(data_path)
        self.metadata = metadata
        with open(f"{data_path}/{split}.pkl", "rb") as f:
            examples = pickle.load(f)
        self.data = examples

    def len(self):
        return len(self.data)

    def get(self, idx):
        particle_type = self.data[idx]["particle_type"].copy()
        particle_type = torch.from_numpy(particle_type)
        position = self.data[idx]["position_seq"].copy()
        position = torch.from_numpy(position)
        data = {"particle_type": particle_type, "position": position}
        return data

def rollout(model, data, metadata, noise_std, rollout_start=0, rollout_length=None, rotation=0):
    device = next(model.parameters()).device
    model.eval()
    window_size = model.window_size + 1
    total_time = data["position"].size(1)-rollout_start
    if rollout_length is not None:
        total_time = rollout_length
    traj = data["position"][:,rollout_start:rollout_start+window_size,:]
    # traj = traj.permute(1, 0, 2)
    particle_type = data["particle_type"]
    boundary = torch.tensor(metadata["bounds"])
    obstacle_particle_indices = torch.where(particle_type == KINEMATIC_PARTICLE_ID)[0]

    rotation = 0
    real_pos = data["position"]
    if rotation != 0:
        center = 0.5 * (boundary[:,0] + boundary[:,1])
        traj = traj - center
        traj = rotate(traj, rotation)
        traj = traj + center
        real_pos = real_pos - center
        real_pos = rotate(real_pos, rotation)       
        real_pos = real_pos + center

    # for i in range(window_size-1):
    #     traj[:,i] = traj[:, -1]
    for time in tqdm(range(total_time - window_size)):
        with torch.no_grad():
            graph = preprocess(particle_type, traj[:, -window_size:], None, metadata, 0.0)
            graph = graph.to(device)
            acceleration = model(graph).cpu()
            acceleration = acceleration * torch.sqrt(torch.sum(torch.tensor(metadata["acc_std"]) ** 2) + noise_std ** 2)

            recent_position = traj[:, -1]
            recent_velocity = recent_position - traj[:, -2]
            new_velocity = recent_velocity + acceleration
            new_position = recent_position + new_velocity
            
            # obstacle particles just follow the ground truth positions
            if len(obstacle_particle_indices) > 0:
                new_position[obstacle_particle_indices,:] = real_pos[obstacle_particle_indices, time+window_size+rollout_start,:]

            new_position[:,0] = torch.maximum(new_position[:,0], boundary[0,0])
            new_position[:,1] = torch.maximum(new_position[:,1], boundary[1,0])
            new_position[:,0] = torch.minimum(new_position[:,0], boundary[0,1])
            new_position[:,1] = torch.minimum(new_position[:,1], boundary[1,1])
            traj = torch.cat((traj, new_position.unsqueeze(1)), dim=1)

    return traj


def oneStepMSE(simulator, dataloader, metadata, noise, sets_to_test=500):
    """Returns two values, loss and MSE"""
    total_loss = 0.0
    total_mse = 0.0
    batch_count = 0
    simulator.eval()
    with torch.no_grad():
        scale = torch.sqrt(torch.sum(torch.tensor(metadata["acc_std"]) ** 2) + noise ** 2).cuda()
        for data in tqdm(dataloader,desc="Validating"):
            data = data.cuda()
            particle_type = data.x
            obstacle_particle_indices = torch.where(particle_type == KINEMATIC_PARTICLE_ID)[0]
            pred = simulator(data)
            pred[obstacle_particle_indices,:] = data.y[obstacle_particle_indices,:]
            # mask out loss on kinematic particles
            mse = ((pred - data.y) * scale) ** 2
            mse = mse.sum(dim=-1).mean()
            loss = ((pred - data.y) ** 2).mean()
            total_mse += mse.item()
            total_loss += loss.item()
            batch_count += 1
            if batch_count >= sets_to_test:
                break
    return total_loss / batch_count, total_mse / batch_count

def rolloutMSE(simulator, dataset, metadata, noise, sets_to_test=[0], starting_points=None, rollout_length=None):
    """Returns two values, loss and MSE"""
    total_loss = 0.0
    total_mse = 0.0
    batch_count = 0
    simulator.eval()
    with torch.no_grad():
        scale = torch.sqrt(torch.sum(torch.tensor(metadata["acc_std"]) ** 2) + noise ** 2)
        for i, rollout_data in enumerate(tqdm(dataset[sets_to_test],desc="Roll out")):
            rollout_start = 0
            if starting_points is not None:
                length_ = 100
                if rollout_length is not None:
                    length_ = rollout_length
                rollout_start = starting_points[i]
            rollout_out = rollout(simulator, rollout_data, dataset.metadata, noise, rollout_start=rollout_start, rollout_length=rollout_length)
            length_ = rollout_out.size(1)
            mse = (rollout_out - rollout_data["position"][:,rollout_start:rollout_start+length_,:]) ** 2
            mse = mse.sum(dim=-1).mean()
            loss = ((rollout_out - rollout_data["position"][:,rollout_start:rollout_start+length_,:]) / scale) ** 2
            loss = loss.sum(dim=-1).mean()
            total_mse += mse.item()
            total_loss += loss.item()
            batch_count += 1
    return total_loss / batch_count, total_mse / batch_count