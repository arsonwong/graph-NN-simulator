import torch
import torch_geometric as pyg
import math
import torch_scatter
import yaml
import os
import json

'''
This code is a PyTorch implementation of a graph neural network (GNN) simulator for particle dynamics, specifically designed to simulate the motion of particles in a 2D space. 
The GNN simulator is based on the Interaction Network architecture and is trained using a dataset of particle trajectories. 
The code started off as basically a copy from a student capstone project 
https://colab.research.google.com/drive/1hirUfPgLU35QCSQSZ7T2lZMyFMOaK_OF?usp=sharing#scrollTo=ZoB4_A6YJ7FP
which is also described in this Medium article: Simulating Complex Physics with Graph Networks: step by step
https://medium.com/stanford-cs224w/simulating-complex-physics-with-graph-networks-step-by-step-177354cb9b05
'''

'''
2025-04-02 "physics" to help learn faster
1. Acceleration due to boundary is the same for each of the 4 walls, just rotated
So, we don't distinguish the 4 walls, just return the closest distance and direction to a wall (done)
We let the model learn the rotational symmetry by randomly rotating the examples to make may duplicates (done)
2. Expand edge feature to include relative velocities: this will help the model learn any damping (done)
3. Not sure if this helps, but we can log direction and magnitudes, rather than displacement and magnitude
Moreover for distances, use 1/(distance+tiny_bias) to make things more "impenetratable"
There will be higher sense of urgency to move away from each other (done)
'''

'''
2025-04-03 more physics
acceleration due to neighbours + gravity + walls = 
function (relative velocity, relative position, down direction, proximity to wall, wall direction, abs position [since walls don't move])

change to

acceleration due to neighbours = function (relative velocitIES, relative position)
where (relative velocitIES, relative position) is passed to the edge
and the node just embeds a blank thing at the start (no information inside)

Also, separate force due to obstacles and force due to swarm neighbours

acceleration due to gravity = constant * down direction

acceleration due to walls = some perpend magnitude (proximity to wall, abs velocitIES in relation to wall direction) * wall direction
+ some parallel magnitude (proximity to wall, abs velocitIES in relation to wall direction) * (parallel to wall direction)
so that friction can be captured

total acceleration = acceleration due to neighbours + acceleration due to gravity + acceleration due to walls'
'''

KINEMATIC_PARTICLE_ID = 3

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
    max_neighbours = config["model"]["max_neighbours"]

def read_metadata(data_path):
  with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
    return json.loads(fp.read())

def rotate(tensor, rotation):
    if rotation == 1: # CCW
        tensor = tensor[..., [1, 0]]
        tensor[...,0] = -tensor[..., 0]
    elif rotation == 2: # 180
        tensor = -tensor
    elif rotation == 3: # CW
        tensor = tensor[..., [1, 0]]
        tensor[...,1] = -tensor[..., 1]
    return tensor
    
def preprocess(particle_type, position_seq, target_position, metadata, noise_std, rotation=0):
    def generate_noise(position_seq, noise_std):
        """Generate noise for a trajectory"""
        velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]
        time_steps = velocity_seq.size(1)
        velocity_noise = torch.randn_like(velocity_seq) * (noise_std / time_steps ** 0.5)
        velocity_noise = velocity_noise.cumsum(dim=1)
        position_noise = velocity_noise.cumsum(dim=1)
        position_noise = torch.cat((torch.zeros_like(position_noise)[:, 0:1], position_noise), dim=1)
        return position_noise
    
    boundary = torch.tensor(metadata["bounds"])

    down_direction = torch.zeros((position_seq.shape[0],metadata["dim"]))
    down_direction[:,1] = -1.0
    if rotation != 0:
        down_direction = rotate(down_direction, rotation)
        center = 0.5 * (boundary[:,0] + boundary[:,1])
        position_seq = position_seq - center
        position_seq = rotate(position_seq, rotation)
        position_seq = position_seq + center
        if target_position is not None:
            target_position = target_position - center
            target_position = rotate(target_position, rotation) 
            target_position = target_position + center
        
    """Preprocess a trajectory and construct the graph"""
    # apply noise to the trajectory
    position_noise = generate_noise(position_seq, noise_std)
    # obstacle particles are not allowed to move
    obstacle_particle_indices = torch.where(particle_type == KINEMATIC_PARTICLE_ID)[0]
    position_noise[obstacle_particle_indices, :] = 0.0
    position_seq = position_seq + position_noise

    # calculate the velocities of particles
    recent_position = position_seq[:, -1]
    velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]

    # construct the graph based on the distances between particles, but end up making advacency matrix symmetrical
    n_particle = recent_position.size(0)
    edge_index = pyg.nn.radius_graph(recent_position, metadata["default_connectivity_radius"], loop=False, max_num_neighbors=max_neighbours)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index2 = pyg.nn.radius_graph(recent_position, metadata["default_connectivity_radius"], loop=False)
    opposite_particles = torch.where(particle_type[edge_index2[0,:]] != particle_type[edge_index2[1,:]])[0]
    edge_index2 = edge_index2[:,opposite_particles]
    edge_index = torch.cat([edge_index, edge_index2], dim=1)
    edge_index = torch.unique(edge_index, dim=1)

    find_ = torch.where((particle_type[edge_index[0,:]] != KINEMATIC_PARTICLE_ID) & (particle_type[edge_index[1,:]] != KINEMATIC_PARTICLE_ID))[0]
    swarm_edges = edge_index[:,find_]
    find_ = torch.where((particle_type[edge_index[0,:]] == KINEMATIC_PARTICLE_ID) | (particle_type[edge_index[1,:]] == KINEMATIC_PARTICLE_ID))[0]
    non_swarm_edges = edge_index[:,find_]
    find_ = torch.where((particle_type[edge_index[0,:]] != KINEMATIC_PARTICLE_ID) & (particle_type[edge_index[1,:]] == KINEMATIC_PARTICLE_ID))[0]
    non_swarm_edges_with_moving_particle = edge_index[:,find_]

    nodes_with_opp_neighbour = edge_index2.flatten()
    nodes_with_opp_neighbour = torch.unique(nodes_with_opp_neighbour)
    has_opp_neighbour = torch.zeros(n_particle, dtype=torch.bool)
    has_opp_neighbour[nodes_with_opp_neighbour] = True   
    has_opp_neighbour[obstacle_particle_indices] = False 
    
    # node-level features: velocity, distance to the boundary
    normal_velocity_seq = velocity_seq / torch.sqrt(torch.sum(torch.tensor(metadata["vel_std"]) ** 2) + noise_std ** 2)
    distance_to_lower_boundary = recent_position - boundary[:, 0]
    distance_to_upper_boundary = boundary[:, 1] - recent_position
    distance_to_boundary = torch.cat((distance_to_lower_boundary, distance_to_upper_boundary), dim=-1)
    norm_distance_to_boundary = torch.clip(distance_to_boundary / metadata["default_connectivity_radius"], -1.0, 1.0)
    min_, _ = torch.min(torch.abs(norm_distance_to_boundary), dim=-1)
    particles_far_from_wall = torch.where((particle_type == KINEMATIC_PARTICLE_ID) | (min_ >= 1))[0]
    particles_close_to_wall = torch.where((particle_type != KINEMATIC_PARTICLE_ID) & (min_ < 1))[0]

    # left wall, lower wall, right wall, upper wall
    # 1 if touching, 0 at default_connectivity_radius or beyond
    norm_inv_distance_to_boundary = (1/(torch.abs(norm_distance_to_boundary) + 0.1) - 1/1.1)/(1/0.1-1/1.1)

    # edge-level features: displacement, distance
    dim = recent_position.size(-1)
    edge_displacement1 = (torch.gather(recent_position, dim=0, index=swarm_edges[0].unsqueeze(-1).expand(-1, dim)) -
                   torch.gather(recent_position, dim=0, index=swarm_edges[1].unsqueeze(-1).expand(-1, dim)))
    edge_displacement1 /= metadata["default_connectivity_radius"]
    inv_edge_displacement1 = torch.sign(edge_displacement1)*(1/(torch.abs(edge_displacement1) + 0.1) - 1/1.1)/(1/0.1-1/1.1)
    depth = normal_velocity_seq.shape[1]
    normal_relative_velocities1 = (torch.gather(normal_velocity_seq, dim=0, index=swarm_edges[0].view(-1, 1, 1).expand(-1, depth, 2)) -
                torch.gather(normal_velocity_seq, dim=0, index=swarm_edges[1].view(-1, 1, 1).expand(-1, depth, 2)))

    edge_displacement2 = (torch.gather(recent_position, dim=0, index=non_swarm_edges[0].unsqueeze(-1).expand(-1, dim)) -
                   torch.gather(recent_position, dim=0, index=non_swarm_edges[1].unsqueeze(-1).expand(-1, dim)))
    edge_displacement2 /= metadata["default_connectivity_radius"]
    inv_edge_displacement2 = torch.sign(edge_displacement2)*(1/(torch.abs(edge_displacement2) + 0.1) - 1/1.1)/(1/0.1-1/1.1)
    depth = normal_velocity_seq.shape[1]
    normal_relative_velocities2 = (torch.gather(normal_velocity_seq, dim=0, index=non_swarm_edges[0].view(-1, 1, 1).expand(-1, depth, 2)) -
                torch.gather(normal_velocity_seq, dim=0, index=non_swarm_edges[1].view(-1, 1, 1).expand(-1, depth, 2)))
    
    edge_displacement3 = (torch.gather(recent_position, dim=0, index=non_swarm_edges_with_moving_particle[1].unsqueeze(-1).expand(-1, dim)) -
                   torch.gather(recent_position, dim=0, index=non_swarm_edges_with_moving_particle[0].unsqueeze(-1).expand(-1, dim)))
    edge_displacement3 /= metadata["default_connectivity_radius"]
    edge_distance3 = edge_displacement3.norm(dim=1, keepdim=True)
    edge_direction3 = torch.zeros_like(edge_displacement3)
    find_ = torch.where(edge_distance3 > 0)[0]
    edge_direction3[find_,:] = edge_displacement3/edge_distance3
    inv_edge_distance3 = (1/(torch.abs(edge_distance3) + 0.1) - 1/1.1)/(1/0.1-1/1.1)
    recent_velocity = normal_velocity_seq[:,-1]
    unit_x = torch.tensor([1,0]).unsqueeze(0)
    unit_y = torch.tensor([0,1]).unsqueeze(0)

    find_ = non_swarm_edges_with_moving_particle[0,:]
    approach_speed = recent_velocity[find_,0].unsqueeze(1)*edge_direction3[:,0].unsqueeze(1) + recent_velocity[find_,1].unsqueeze(1)*edge_direction3[:,1].unsqueeze(1)
    approach_speed = torch.clip(approach_speed,0,1000)
    
    # ground truth for training
    if target_position is not None:
        last_velocity = velocity_seq[:, -1]
        next_velocity = target_position + position_noise[:, -1] - recent_position
        acceleration = next_velocity - last_velocity
        acceleration = acceleration / torch.sqrt(torch.sum(torch.tensor(metadata["acc_std"]) ** 2) + noise_std ** 2)
    else:
        acceleration = None

    # return the graph with features

    graph = pyg.data.Data(
        x=particle_type,
        edge_index=swarm_edges,
        # anti-symmetric
        edge_attr=torch.cat((edge_displacement1, inv_edge_displacement1, normal_relative_velocities1.flatten(start_dim=1)), dim=-1),
        edge_index2=non_swarm_edges,
        edge_attr2=torch.cat((edge_displacement2, inv_edge_displacement2, normal_relative_velocities2.flatten(start_dim=1)), dim=-1),
        edge_index3 = non_swarm_edges_with_moving_particle,
        y=acceleration,
        pos=normal_velocity_seq.flatten(start_dim=1), # captures any air drag
        aux = {'has_opp_neighbour':has_opp_neighbour, 'particles_close_to_wall': particles_close_to_wall, 
               'down_direction':down_direction,'particles_far_from_wall':particles_far_from_wall,'recent_velocity':recent_velocity,
               'acceleration_scale': torch.sqrt(torch.sum(torch.tensor(metadata["acc_std"]) ** 2) + noise_std ** 2),
               'velocity_scale': torch.sqrt(torch.sum(torch.tensor(metadata["vel_std"]) ** 2) + noise_std ** 2), 
               'norm_inv_distance_to_boundary':norm_inv_distance_to_boundary, 'norm_distance_to_boundary':norm_distance_to_boundary, 'recent_position':recent_position, 
               'normal_velocity_seq': normal_velocity_seq, 'unit_x': unit_x, 'unit_y': unit_y,
               'approach_speed': approach_speed, 'inv_edge_distance3': inv_edge_distance3, 'edge_direction3': edge_direction3}
    )

    return graph

class MLP(torch.nn.Module):
    """Multi-Layer perceptron"""
    def __init__(self, input_size, hidden_size, output_size, layers, layernorm=True):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.input_size = input_size
        for i in range(layers):
            self.layers.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == layers - 1 else hidden_size,
            ))
            if i != layers - 1:
                self.layers.append(torch.nn.ReLU())
        if layernorm and output_size>1:
            self.layers.append(torch.nn.LayerNorm(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class InteractionNetwork(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper: 
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, hidden_size, layers):
        super().__init__()
        self.lin_edge = MLP(hidden_size * 3, hidden_size, hidden_size, layers)
        self.lin_node = MLP(hidden_size * 2, hidden_size, hidden_size, layers)

    def forward(self, x, edge_index, edge_feature):
        edge_out, aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)
        node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
        edge_out = edge_feature + edge_out
        node_out = x + node_out
        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature):
        x = torch.cat((x_i, x_j, edge_feature), dim=-1)
        x = self.lin_edge(x)
        return x

    def aggregate(self, inputs, index, dim_size=None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return (inputs, out)
    
class LearnedSimulator(torch.nn.Module):
    """Graph Network-based Simulators(GNS)"""
    def __init__(
        self,
        hidden_size=128,
        n_mp_layers=10, # number of GNN layers
        num_particle_types=9,
        particle_type_dim=16, # embedding dimension of particle types
        dim=2, # dimension of the world, typical 2D or 3D
        window_size=5, # the model looks into W frames before the frame to be predicted
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.dim = dim
        self.embed_type2 = torch.nn.Embedding(num_particle_types, particle_type_dim)
        self.node_in1 = MLP(dim+1, hidden_size//2, hidden_size//2, 3)
        self.node_in2 = MLP(particle_type_dim+dim, hidden_size//2, hidden_size//2, 3)
        self.node_in3 = MLP(dim*(window_size), hidden_size//2, hidden_size//2, 3)
        self.edge_in1 = MLP(dim*(window_size+2), hidden_size//2, hidden_size//2, 3)
        self.node_out1 = MLP(hidden_size//2, hidden_size//2, dim, 3, layernorm=False)
        self.edge_in2 = MLP(dim*(window_size+2), hidden_size//2, hidden_size//2, 3)
        self.node_out2 = MLP(hidden_size//2, hidden_size//2, dim, 3, layernorm=False)
        self.wall_in = MLP(dim*window_size + 2 + dim, hidden_size//2, dim, 6, layernorm=False)
        self.n_mp_layers = n_mp_layers // 2
        self.layers1 = torch.nn.ModuleList([InteractionNetwork(
            hidden_size//2, 3
        ) for _ in range(n_mp_layers)])
        self.layers2 = torch.nn.ModuleList([InteractionNetwork(
            hidden_size//2, 3
        ) for _ in range(n_mp_layers)])

        self.gravity = torch.nn.Parameter(torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embed_type2.weight)

    def forward(self, data: pyg.data.Data) -> torch.Tensor:
        acceleration_scale = data.aux['acceleration_scale']
        velocity_scale = data.aux['velocity_scale']
        recent_velocity = data.aux['recent_velocity']
        unit_x = data.aux['unit_x']
        unit_y = data.aux['unit_y']
        if acceleration_scale.shape != torch.Size([]):
            acceleration_scale = acceleration_scale[0]
            velocity_scale = velocity_scale[0]
            unit_x = unit_x[0]
            unit_y = unit_y[0]

        #acceleration due to gravity = constant * down direction, with bias term that's the true value
        down_direction = data.aux['down_direction']
        g = 5.5339e-05/acceleration_scale*60/94
        out = g*down_direction

        # no drag force
        # last particle is a reference for blank
        node_feature1 = torch.zeros((data.x.shape[0]+1, self.dim+1), device=data.x.device)
        node_feature1 = self.node_in1(node_feature1)
        edge_feature1 = self.edge_in1(data.edge_attr)
        
        # stack of GNN layers
        for i in range(self.n_mp_layers):
            node_feature1, edge_feature1 = self.layers1[i](node_feature1, data.edge_index, edge_feature=edge_feature1)

        # post-processing
        # acceleration due to neighbours
        swarm_acceleration = self.node_out1(node_feature1)
        blank_ = swarm_acceleration[-1,:]
        swarm_acceleration = swarm_acceleration[0:-1,:]
        swarm_acceleration -= blank_
        mean_swarm_acceleration = torch.mean(swarm_acceleration,axis=0)
        swarm_acceleration -= mean_swarm_acceleration
        recent_position = data.aux['recent_position']
        center = torch.mean(recent_position,axis=0)
        displacement = recent_position-center
        distance = displacement.norm(dim=1, keepdim=True)
        total_distance = torch.sum(distance)
        moment = swarm_acceleration[:,1]*displacement[:,0]-swarm_acceleration[:,0]*displacement[:,1]
        total_angular_momentum = torch.sum(moment)
        counter_moment = torch.zeros_like(swarm_acceleration)
        counter_moment[:,0] = displacement[:,1]*total_angular_momentum/total_distance
        counter_moment[:,1] = -displacement[:,0]*total_angular_momentum/total_distance
        swarm_acceleration += counter_moment
        out += swarm_acceleration

        #bias at obstacle particle - just try to stop the particle
        obstacle_bias = torch.zeros((data.x.shape[0],2), device=data.x.device)
        approach_speed = data.aux['approach_speed']
        inv_edge_distance3 = data.aux['inv_edge_distance3']
        edge_direction3 = data.aux['edge_direction3']
        vals = inv_edge_distance3 * approach_speed * edge_direction3 * velocity_scale / acceleration_scale * 60/94 * 10 
        idx = data.edge_index3[0,:]
        obstacle_bias.index_add_(0, idx, vals)
        obstacle_bias = torch.sign(obstacle_bias) * torch.min(torch.abs(obstacle_bias), torch.abs(recent_velocity * velocity_scale / acceleration_scale * 60/94))
        out -= obstacle_bias

        # left wall, lower wall, right wall, upper wall
        norm_inv_distance_to_boundary = data.aux['norm_inv_distance_to_boundary']
        norm_distance_to_boundary = data.aux['norm_distance_to_boundary']
        normal_velocity_seq = data.aux['normal_velocity_seq']

        # bias at wall - just stops the particle from going through the wall
        wall_bias = torch.zeros((data.x.shape[0],2), device=data.x.device)
        find_ = torch.where(norm_inv_distance_to_boundary[:,0].unsqueeze(1) > 1e-7)[0]
        if len(find_) > 0:
            find2_ = torch.where(norm_distance_to_boundary[find_,0] < 0)[0]
            if len(find2_) > 0:
                wall_bias[find_[find2_],:] += torch.clip(-recent_velocity[find_[find2_],0],0,1000).unsqueeze(1) * -unit_x * velocity_scale / acceleration_scale * 60/94
            find2_ = torch.where(norm_distance_to_boundary[find_,0] >= 0)[0]
            if len(find2_) > 0:
                wall_bias[find_[find2_],:] += norm_inv_distance_to_boundary[find_[find2_],0].unsqueeze(1) * torch.clip(-recent_velocity[find_[find2_],0],0,1000).unsqueeze(1) * -unit_x * velocity_scale / acceleration_scale * 60/94           

        find_ = torch.where(norm_inv_distance_to_boundary[:,1].unsqueeze(1) > 1e-7)[0]
        if len(find_) > 0:
            find2_ = torch.where(norm_distance_to_boundary[find_,1] < 0)[0]          
            if len(find2_) > 0:
                wall_bias[find_[find2_],:] += torch.clip(-recent_velocity[find_[find2_],1],0,1000).unsqueeze(1) * -unit_y * velocity_scale / acceleration_scale * 60/94
            find2_ = torch.where(norm_distance_to_boundary[find_,1] >= 0)[0]
            if len(find2_) > 0:
                wall_bias[find_[find2_],:] += norm_inv_distance_to_boundary[find_[find2_],1].unsqueeze(1) * torch.clip(-recent_velocity[find_[find2_],1],0,1000).unsqueeze(1) * -unit_y * velocity_scale / acceleration_scale * 60/94             

        find_ = torch.where(norm_inv_distance_to_boundary[:,2].unsqueeze(1) > 1e-7)[0]
        if len(find_) > 0:
            find2_ = torch.where(norm_distance_to_boundary[find_,2] < 0)[0]
            if len(find2_) > 0:
                wall_bias[find_[find2_],:] += torch.clip(recent_velocity[find_[find2_],0],0,1000).unsqueeze(1) * unit_x * velocity_scale / acceleration_scale * 60/94
            find2_ = torch.where(norm_distance_to_boundary[find_,2] >= 0)[0]
            if len(find2_) > 0:
                wall_bias[find_[find2_],:] += norm_inv_distance_to_boundary[find_[find2_],2].unsqueeze(1) * torch.clip(recent_velocity[find_[find2_],0],0,1000).unsqueeze(1) * unit_x * velocity_scale / acceleration_scale * 60/94

        find_ = torch.where(norm_inv_distance_to_boundary[:,3].unsqueeze(1) > 1e-7)[0]
        if len(find_) > 0:
            find2_ = torch.where(norm_distance_to_boundary[find_,3] < 0)[0]
            if len(find2_) > 0:
                wall_bias[find_[find2_],:] += torch.clip(recent_velocity[find_[find2_],1],0,1000).unsqueeze(1) * unit_y * velocity_scale / acceleration_scale * 60/94
            find2_ = torch.where(norm_distance_to_boundary[find_,3] >= 0)[0]
            if len(find2_) > 0:
                wall_bias[find_[find2_],:] += norm_inv_distance_to_boundary[find_[find2_],3].unsqueeze(1) * torch.clip(recent_velocity[find_[find2_],1],0,1000).unsqueeze(1) * unit_y * velocity_scale / acceleration_scale * 60/94

        out -= wall_bias

        # next force gets a chance to experience the current force, so that it may try to counteract it
        node_feature2 = torch.cat((self.embed_type2(data.x), out), dim=-1)
        find_ = torch.where(data.x != KINEMATIC_PARTICLE_ID)[0]
        blank_x = (torch.ones((1), device=data.x.device) * data.x[find_[0]]).to(torch.int64)
        blank_node_feature2 = torch.cat((self.embed_type2(blank_x), torch.zeros((1, 2), device=data.x.device)),dim=-1)
        node_feature2 = torch.cat((node_feature2, blank_node_feature2),dim=0)
        node_feature2 = self.node_in2(node_feature2)
        edge_feature2 = self.edge_in2(data.edge_attr2)

        # stack of GNN layers
        for i in range(self.n_mp_layers):
            node_feature2, edge_feature2 = self.layers2[i](node_feature2, data.edge_index2, edge_feature=edge_feature2)

        obstacle_acceleration = self.node_out2(node_feature2)
        blank_ = obstacle_acceleration[-1,:] # subtract the blank
        obstacle_acceleration = obstacle_acceleration[0:-1,:]
        obstacle_acceleration -= blank_
        find_ = torch.where(data.aux['has_opp_neighbour']==False)[0]
        obstacle_acceleration[find_,:] = 0.0

        out += obstacle_acceleration

        wall_accel = torch.zeros((data.x.shape[0],2), device=data.x.device)
        # wall force
        find_ = torch.where(norm_inv_distance_to_boundary[:,0].unsqueeze(1) > 1e-7)[0]
        if len(find_) > 0:
            # wall gets to feel the other forces so that it gets a chance to repel them
            input = torch.cat((-out[find_,0].unsqueeze(1), -out[find_,1].unsqueeze(1), -normal_velocity_seq[find_,:,0],-normal_velocity_seq[find_,:,1],norm_inv_distance_to_boundary[find_,0].unsqueeze(1),norm_distance_to_boundary[find_,0].unsqueeze(1)), dim=-1)
            wall_out = self.wall_in(input)
            wall_accel[find_] += wall_out[:,0].unsqueeze(1) * -unit_x + wall_out[:,1].unsqueeze(1) * -unit_y

        find_ = torch.where(norm_inv_distance_to_boundary[:,1].unsqueeze(1) > 1e-7)[0]
        if len(find_) > 0:
            # wall gets to feel the other forces so that it gets a chance to repel them
            input = torch.cat((-out[find_,1].unsqueeze(1), out[find_,0].unsqueeze(1),-normal_velocity_seq[find_,:,1],normal_velocity_seq[find_,:,0],norm_inv_distance_to_boundary[find_,1].unsqueeze(1),norm_distance_to_boundary[find_,1].unsqueeze(1)), dim=-1)
            wall_out = self.wall_in(input)
            wall_accel[find_] += wall_out[:,0].unsqueeze(1) * -unit_y + wall_out[:,1].unsqueeze(1) * unit_x

        find_ = torch.where(norm_inv_distance_to_boundary[:,2].unsqueeze(1) > 1e-7)[0]
        if len(find_) > 0:
            # wall gets to feel the other forces so that it gets a chance to repel them
            input = torch.cat((out[find_,0].unsqueeze(1), out[find_,1].unsqueeze(1),normal_velocity_seq[find_,:,0],normal_velocity_seq[find_,:,1],norm_inv_distance_to_boundary[find_,2].unsqueeze(1),norm_distance_to_boundary[find_,2].unsqueeze(1)), dim=-1)
            wall_out = self.wall_in(input)
            wall_accel[find_] += wall_out[:,0].unsqueeze(1) * unit_x + wall_out[:,1].unsqueeze(1) * unit_y

        find_ = torch.where(norm_inv_distance_to_boundary[:,3].unsqueeze(1) > 1e-7)[0]
        if len(find_) > 0:
             # wall gets to feel the other forces so that it gets a chance to repel them
            input = torch.cat((out[find_,1].unsqueeze(1), -out[find_,0].unsqueeze(1),normal_velocity_seq[find_,:,1],-normal_velocity_seq[find_,:,0],norm_inv_distance_to_boundary[find_,3].unsqueeze(1),norm_distance_to_boundary[find_,3].unsqueeze(1)), dim=-1)
            wall_out = self.wall_in(input)
            wall_accel[find_] += wall_out[:,0].unsqueeze(1) * unit_y + wall_out[:,1].unsqueeze(1) * -unit_x

        out += wall_accel

        is_at_wall_or_obstacle = torch.zeros((data.x.shape[0]), device=data.x.device)
        find_ = data.aux['has_opp_neighbour']
        particles_close_to_wall = data.aux['particles_close_to_wall']
        is_at_wall_or_obstacle[find_] = 1
        is_at_wall_or_obstacle[particles_close_to_wall] = 1
        node_feature1 = torch.zeros((data.x.shape[0], self.dim+1), device=data.x.device)
        node_feature1[:,0:2] = -obstacle_bias-wall_bias+obstacle_acceleration+wall_accel
        node_feature1[:,2] = is_at_wall_or_obstacle
        blank_node_feature1 = torch.zeros((1, self.dim+1), device=data.x.device)
        node_feature1 = torch.cat((node_feature1,blank_node_feature1),dim=0)
        node_feature1 = self.node_in1(node_feature1)
        edge_feature1 = self.edge_in1(data.edge_attr)
        
        # stack of GNN layers
        for i in range(self.n_mp_layers):
            node_feature1, edge_feature1 = self.layers1[i](node_feature1, data.edge_index, edge_feature=edge_feature1)
        swarm_acceleration_mod = self.node_out1(node_feature1)
        blank_ = swarm_acceleration_mod[-1,:] # subtract the blank
        swarm_acceleration_mod = swarm_acceleration_mod[0:-1,:]
        swarm_acceleration_mod -= blank_
        swarm_acceleration_mod -= mean_swarm_acceleration
        swarm_acceleration_mod += counter_moment
        out += swarm_acceleration_mod - swarm_acceleration

        return out
