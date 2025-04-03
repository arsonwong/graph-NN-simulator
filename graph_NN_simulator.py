import torch
import torch_geometric as pyg
import math
import torch_scatter

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

acceleration due to gravity = constant * down direction

acceleration due to walls = some perpend magnitude (proximity to wall, abs velocitIES in relation to wall direction) * wall direction
+ some parallel magnitude (proximity to wall, abs velocitIES in relation to wall direction) * (parallel to wall direction)
so that friction can be captured

total acceleration = acceleration due to neighbours + acceleration due to gravity + acceleration due to walls'
'''

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
        self.window_size = window_size
        self.embed_type = torch.nn.Embedding(num_particle_types, particle_type_dim)
        self.node_in = MLP(particle_type_dim, hidden_size, hidden_size, 3)
        self.edge_in = MLP(dim*(window_size+1) + 1, hidden_size, hidden_size, 3)
        self.node_out = MLP(hidden_size, hidden_size, dim, 3, layernorm=False)
        self.wall_in = MLP(dim*window_size + 1, hidden_size, dim, 3)
        self.n_mp_layers = n_mp_layers
        self.layers = torch.nn.ModuleList([InteractionNetwork(
            hidden_size, 3
        ) for _ in range(n_mp_layers)])

        self.gravity = torch.nn.Parameter(torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embed_type.weight)

    def forward(self, data: pyg.data.Data) -> torch.Tensor:
        # pre-processing
        # node feature: combine categorial feature data.x and contiguous feature data.pos.
        node_feature = self.embed_type(data.x)
        node_feature = self.node_in(node_feature)
        edge_feature = self.edge_in(data.edge_attr)
        # stack of GNN layers
        for i in range(self.n_mp_layers):
            node_feature, edge_feature = self.layers[i](node_feature, data.edge_index, edge_feature=edge_feature)
        # post-processing
        # acceleration due to neighbours
        out = self.node_out(node_feature)

        wall_in_parameters = data.aux['wall_in_parameters']
        direction_to_boundary = data.aux['direction_to_boundary']
        direction_parallel_boundary = data.aux['direction_parallel_boundary']
        down_direction = data.aux['down_direction']
        # acceleration due to walls

        #acceleration due to gravity = constant * down direction
        out += self.gravity*down_direction

        #acceleration due to walls = some magnitude (proximity to wall, relative velocitIES) * wall direction
        wall_out = self.wall_in(wall_in_parameters)
        out +=  wall_out[:,0].unsqueeze(1) * direction_to_boundary + wall_out[:,1].unsqueeze(1) * direction_parallel_boundary

        return out
