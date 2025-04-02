import os
import torch
import torch_geometric as pyg
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from data_processing import *
from graph_NN_simulator import *
import yaml
from datetime import datetime

'''
This code is a PyTorch implementation of a graph neural network (GNN) simulator for particle dynamics, specifically designed to simulate the motion of particles in a 2D space. 
The GNN simulator is based on the Interaction Network architecture and is trained using a dataset of particle trajectories. 
The code started off as basically a copy from a student capstone project 
https://colab.research.google.com/drive/1hirUfPgLU35QCSQSZ7T2lZMyFMOaK_OF?usp=sharing#scrollTo=ZoB4_A6YJ7FP
which is also described in this Medium article: Simulating Complex Physics with Graph Networks: step by step
https://medium.com/stanford-cs224w/simulating-complex-physics-with-graph-networks-step-by-step-177354cb9b05
'''


def train(params, simulator, train_loader, valid_loader, metadata, valid_rollout_dataset, obstacle_bias=0.0, visualize=False, prefix=""):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(simulator.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / 5e6))

    # recording loss curve
    train_loss_list = []
    eval_loss_list = []
    onestep_mse_list = []
    rollout_loss_list = []
    total_step = 0

    if visualize:
        plt.ion()  # Turn on interactive mode
        fig, axes = plt.subplots(1,3, figsize=(16, 4))

    for i in range(params["epoch"]):
        simulator.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {i}")
        total_loss = 0
        batch_count = 0
        for data in progress_bar:
            optimizer.zero_grad()
            data = data.to(device)
            has_opp_neighbour = data.aux
            pred = simulator(data)
            particle_type = data.x
            obstacle_particle_indices = torch.where(particle_type == KINEMATIC_PARTICLE_ID)[0]
            find_ = torch.where(has_opp_neighbour)[0]
            pred[find_,:] *= 1.0+obstacle_bias
            data.y[find_,:] *= 1.0+obstacle_bias
            pred[obstacle_particle_indices,:] = data.y[obstacle_particle_indices,:]
            loss = loss_fn(pred, data.y)

            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({"loss": loss.item(), "avg_loss": total_loss / batch_count, "lr": optimizer.param_groups[0]["lr"]})
            total_step += 1

            if visualize and total_step % 10 == 0:
                train_loss_list.append((total_step, loss.item()))

                # Clear and update the plot
                for ax in axes:
                    ax.clear()

                if train_loss_list:
                    axes[0].scatter(*zip(*train_loss_list))
                axes[0].set_xlabel("Iterations")
                axes[0].set_ylabel("Loss")
                axes[0].set_title("Train Loss")
                axes[0].set_yscale("log")  # Log scale for y-axis

                # Eval loss subplot
                if eval_loss_list:
                    axes[1].scatter(*zip(*eval_loss_list))
                axes[1].set_xlabel("Iterations")
                axes[1].set_ylabel("Loss")
                axes[1].set_title("Eval Loss")
                axes[1].set_yscale("log")  # Log scale for y-axis

                # Rollout loss subplot
                if rollout_loss_list:
                    axes[2].scatter(*zip(*rollout_loss_list))
                axes[2].set_xlabel("Iterations")
                axes[2].set_ylabel("Loss")
                axes[2].set_title("Rollout Loss")
                axes[2].set_yscale("log")  # Log scale for y-axis
                
                plt.draw()
                plt.pause(0.01)  # Brief pause to allow GUI update
            
            if total_step % 100 == 0:
                with open(os.path.join(training_stats_path,f"{prefix}_train_loss.txt"), "a") as file:
                    file.write(f"{total_step},{loss.item()}\n") 

            # save model
            if total_step % params["save_interval"] == 0:
                torch.save(
                    {
                        "model": simulator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    os.path.join(model_path, f"{prefix}_checkpoint_{total_step}.pt")
                )

            # evaluation
            if total_step % params["eval_interval"] == 0:
                simulator.eval()
                eval_loss, onestep_mse = oneStepMSE(simulator, valid_loader, metadata, params["noise"])
                eval_loss_list.append((total_step, eval_loss))
                onestep_mse_list.append((total_step, onestep_mse))
                tqdm.write(f"\nEval: Loss: {eval_loss}, One Step MSE: {onestep_mse}")
                with open(os.path.join(training_stats_path,f"{prefix}_eval_loss.txt"), "a") as file:
                    file.write(f"{total_step},{eval_loss}\n") 
                simulator.train()

            # do rollout on valid set
            if total_step % params["rollout_interval"] == 0:
                simulator.eval()
                rollout_loss, rollout_mse = rolloutMSE(simulator, valid_rollout_dataset, metadata, params["noise"])
                rollout_loss_list.append((total_step, rollout_loss))
                tqdm.write(f"\nEval: Rollout Loss: {rollout_loss}, Rollout MSE: {rollout_mse}")
                with open(os.path.join(training_stats_path,f"{prefix}_rollout_loss.txt"), "a") as file:
                    file.write(f"{total_step},{rollout_loss}\n") 
                simulator.train()

    if visualize:
        plt.ioff()  # Turn off interactive mode when training is done
        plt.show()  # Final plot after training
    
    torch.save(
        {
            "model": simulator.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        os.path.join(model_path, f"{prefix}_checkpoint_{total_step}.pt")
    )
    
    return train_loss_list, eval_loss_list, onestep_mse_list, rollout_mse_list

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    params = config["training"]
    model_params = config["model"]
    data_path = config['data']['data_path']
    model_path = config['data']['model_path']
    training_stats_path = config['data']['training_stats_path']
    session_name = config["training"]['session_name']
    visualize = config["training"]['visualize']
    load_model_path = config["training"]['load_model_path']
    obstacle_bias = config["training"]['obstacle_bias']

    if len(session_name)>0:
        session_name += "_"

    session_name += datetime.now().strftime("%Y-%m-%d_%H_%M")

    # load dataset
    train_dataset = OneStepDataset(data_path, "train", noise_std=params["noise"], random_rotation=True)
    valid_dataset = OneStepDataset(data_path, "valid", noise_std=params["noise"])
    train_loader = pyg.loader.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, pin_memory=True, num_workers=1)
    valid_loader = pyg.loader.DataLoader(valid_dataset, batch_size=params["batch_size"], shuffle=True, pin_memory=True, num_workers=1)
    valid_rollout_dataset = RolloutDataset(data_path, "valid")

    # build model
    simulator = LearnedSimulator(hidden_size=model_params["hidden_size"], 
                                    n_mp_layers=model_params["n_mp_layers"], 
                                    window_size=model_params["window_size"])
    simulator = simulator.to(device)

    if len(load_model_path)>0:
        checkpoint = torch.load(os.path.join(model_path, load_model_path))
        simulator.load_state_dict(checkpoint["model"])

    # train the model
    metadata = read_metadata(data_path)
    train_loss_list, eval_loss_list, onestep_mse_list, rollout_mse_list = train(params, simulator, train_loader, valid_loader, metadata, valid_rollout_dataset, 
                                                                                obstacle_bias=obstacle_bias, prefix=session_name, visualize=visualize)
