import os
import torch
import json
import numpy as np
import torch_geometric as pyg
import pickle
from tqdm import tqdm

'''
This code is a PyTorch implementation of a graph neural network (GNN) simulator for particle dynamics, specifically designed to simulate the motion of particles in a 2D space. 
The GNN simulator is based on the Interaction Network architecture and is trained using a dataset of particle trajectories. 
The code started off as basically a copy from a student capstone project 
https://colab.research.google.com/drive/1hirUfPgLU35QCSQSZ7T2lZMyFMOaK_OF?usp=sharing#scrollTo=ZoB4_A6YJ7FP
which is also described in this Medium article: Simulating Complex Physics with Graph Networks: step by step
https://medium.com/stanford-cs224w/simulating-complex-physics-with-graph-networks-step-by-step-177354cb9b05
'''

def read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())
    
def preprocess(particle_type, position_seq, target_position, metadata, noise_std):
    def generate_noise(position_seq, noise_std):
        """Generate noise for a trajectory"""
        velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]
        time_steps = velocity_seq.size(1)
        velocity_noise = torch.randn_like(velocity_seq) * (noise_std / time_steps ** 0.5)
        velocity_noise = velocity_noise.cumsum(dim=1)
        position_noise = velocity_noise.cumsum(dim=1)
        position_noise = torch.cat((torch.zeros_like(position_noise)[:, 0:1], position_noise), dim=1)
        return position_noise

    """Preprocess a trajectory and construct the graph"""
    # apply noise to the trajectory
    position_noise = generate_noise(position_seq, noise_std)
    position_seq = position_seq + position_noise

    # calculate the velocities of particles
    recent_position = position_seq[:, -1]
    velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]

    # construct the graph based on the distances between particles
    n_particle = recent_position.size(0)
    edge_index = pyg.nn.radius_graph(recent_position, metadata["default_connectivity_radius"], loop=True, max_num_neighbors=n_particle)
    
    # node-level features: velocity, distance to the boundary
    normal_velocity_seq = (velocity_seq - torch.tensor(metadata["vel_mean"])) / torch.sqrt(torch.tensor(metadata["vel_std"]) ** 2 + noise_std ** 2)
    boundary = torch.tensor(metadata["bounds"])
    distance_to_lower_boundary = recent_position - boundary[:, 0]
    distance_to_upper_boundary = boundary[:, 1] - recent_position
    distance_to_boundary = torch.cat((distance_to_lower_boundary, distance_to_upper_boundary), dim=-1)
    distance_to_boundary = torch.clip(distance_to_boundary / metadata["default_connectivity_radius"], -1.0, 1.0)

    # edge-level features: displacement, distance
    dim = recent_position.size(-1)
    edge_displacement = (torch.gather(recent_position, dim=0, index=edge_index[0].unsqueeze(-1).expand(-1, dim)) -
                   torch.gather(recent_position, dim=0, index=edge_index[1].unsqueeze(-1).expand(-1, dim)))
    edge_displacement /= metadata["default_connectivity_radius"]
    edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)

    # ground truth for training
    if target_position is not None:
        last_velocity = velocity_seq[:, -1]
        next_velocity = target_position + position_noise[:, -1] - recent_position
        acceleration = next_velocity - last_velocity
        acceleration = (acceleration - torch.tensor(metadata["acc_mean"])) / torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise_std ** 2)
    else:
        acceleration = None

    # return the graph with features
    graph = pyg.data.Data(
        x=particle_type,
        edge_index=edge_index,
        edge_attr=torch.cat((edge_displacement, edge_distance), dim=-1),
        y=acceleration,
        pos=torch.cat((velocity_seq.reshape(velocity_seq.size(0), -1), distance_to_boundary), dim=-1)
    )

    return graph

class OneStepDataset(pyg.data.Dataset):
    def __init__(self, data_path, split, window_length=7, noise_std=0.0, return_pos=False):
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
            graph = preprocess(particle_type, position_seq, target_position, self.metadata, self.noise_std)
        if self.return_pos:
          return graph, position_seq[:, -1]
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

def rollout(model, data, metadata, noise_std, rollout_length=None):
    device = next(model.parameters()).device
    model.eval()
    window_size = model.window_size + 1
    total_time = data["position"].size(1)
    if rollout_length is not None:
        total_time = rollout_length
    traj = data["position"][:,:window_size,:]
    # traj = traj.permute(1, 0, 2)
    particle_type = data["particle_type"]
    boundary = torch.tensor(metadata["bounds"])

    for time in tqdm(range(total_time - window_size),desc="rollout"):
        with torch.no_grad():
            graph = preprocess(particle_type, traj[:, -window_size:], None, metadata, 0.0)
            graph = graph.to(device)
            acceleration = model(graph).cpu()
            acceleration = acceleration * torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise_std ** 2) + torch.tensor(metadata["acc_mean"])

            recent_position = traj[:, -1]
            recent_velocity = recent_position - traj[:, -2]
            new_velocity = recent_velocity + acceleration
            new_position = recent_position + new_velocity
            new_position[:,0] = torch.maximum(new_position[:,0], boundary[0,0])
            new_position[:,1] = torch.maximum(new_position[:,1], boundary[1,0])
            new_position[:,0] = torch.minimum(new_position[:,0], boundary[0,1])
            new_position[:,1] = torch.minimum(new_position[:,1], boundary[1,1])
            traj = torch.cat((traj, new_position.unsqueeze(1)), dim=1)

    return traj


def oneStepMSE(simulator, dataloader, metadata, noise):
    """Returns two values, loss and MSE"""
    total_loss = 0.0
    total_mse = 0.0
    batch_count = 0
    simulator.eval()
    with torch.no_grad():
        scale = torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise ** 2).cuda()
        for data in tqdm(dataloader,desc="Validating"):
            data = data.cuda()
            pred = simulator(data)
            mse = ((pred - data.y) * scale) ** 2
            mse = mse.sum(dim=-1).mean()
            loss = ((pred - data.y) ** 2).mean()
            total_mse += mse.item()
            total_loss += loss.item()
            batch_count += 1
    return total_loss / batch_count, total_mse / batch_count

def rolloutMSE(simulator, dataset, metadata, noise, sets_to_test=1):
    """Returns two values, loss and MSE"""
    total_loss = 0.0
    total_mse = 0.0
    batch_count = 0
    simulator.eval()
    with torch.no_grad():
        scale = torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise ** 2)
        for rollout_data in tqdm(dataset[:sets_to_test],desc="Roll out"):
            rollout_out = rollout(simulator, rollout_data, dataset.metadata, noise)
            mse = (rollout_out - rollout_data["position"]) ** 2
            mse = mse.sum(dim=-1).mean()
            loss = ((rollout_out - rollout_data["position"]) / scale) ** 2
            loss = loss.sum(dim=-1).mean()
            total_mse += mse.item()
            total_loss += loss.item()
            batch_count += 1
        print(scale)
    return total_loss / batch_count, total_mse / batch_count