import os
import torch
import torch_geometric as pyg
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from data_processing import *
import yaml
from datetime import datetime
import re
import importlib

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


def train(params, simulator, train_loader, valid_loader, metadata, valid_rollout_dataset, 
          optimizer_state_dict = None, scheduler_state_dict = None,
          obstacle_bias=0.0, wall_bias=0.0, visualize=False, prefix="", v=0, total_steps_start=0):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(simulator.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / 5e6))

    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    if scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    # recording loss curve
    train_loss_list = []
    eval_loss_list = []
    onestep_mse_list = []
    rollout_loss_list = []
    total_step = total_steps_start

    file_path = os.path.join(training_stats_path, f"{prefix}_train_loss.txt")
    if os.path.exists(file_path):
        data = np.loadtxt(file_path, delimiter=",")  # Load two-column CSV data
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)  # or data = data[None, :]
        for row in data:
            if row[0] >= total_steps_start:
                break
            train_loss_list.append((row[0], row[1], row[2], row[3]))
        with open(file_path, "w") as file:
            for row in train_loss_list:
                file.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n") 


    file_path = os.path.join(training_stats_path, f"{prefix}_eval_loss.txt")
    if os.path.exists(file_path):
        data = np.loadtxt(file_path, delimiter=",")  # Load two-column CSV data
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)  # or data = data[None, :]
        for row in data:
            if row[0] >= total_steps_start:
                break
            eval_loss_list.append((row[0], row[1]))
        with open(file_path, "w") as file:
            for row in eval_loss_list:
                file.write(f"{row[0]},{row[1]}\n") 

    file_path = os.path.join(training_stats_path, f"{prefix}_rollout_loss.txt")
    if os.path.exists(file_path):
        data = np.loadtxt(file_path, delimiter=",")  # Load two-column CSV data
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)  # or data = data[None, :]
        for row in data:
            if row[0] >= total_steps_start:
                break
            rollout_loss_list.append((row[0], row[1]))
        with open(file_path, "w") as file:
            for row in rollout_loss_list:
                file.write(f"{row[0]},{row[1]}\n") 

    if visualize:
        plt.ion()  # Turn on interactive mode
        fig, axes = plt.subplots(1,3, figsize=(16, 4))

    epoch_passed = int(np.floor(total_steps_start/float(len(train_loader))))
    residual = total_steps_start - epoch_passed*len(train_loader)
    loss_overall = 0.0
    loss_overall_counter = 0
    loss_close_to_wall = 0.0
    loss_close_to_wall_counter = 0
    loss_with_obstacle = 0.0    
    loss_with_obstacle_counter = 0
    
    for i in range(epoch_passed,params["epoch"]):
        simulator.train()
        if i==epoch_passed:
            progress_bar = tqdm(train_loader, desc=f"Epoch {i}", initial=residual)
        else:
            progress_bar = tqdm(train_loader, desc=f"Epoch {i}")
        batch_count = 0

        for data in progress_bar:
            optimizer.zero_grad()
            data = data.to(device)
            has_opp_neighbour = data.aux['has_opp_neighbour']
            pred = simulator(data)
            particle_type = data.x
            obstacle_particle_indices = torch.where(particle_type == KINEMATIC_PARTICLE_ID)[0]
            pred[obstacle_particle_indices,:] = data.y[obstacle_particle_indices,:]
            loss_overall += loss_fn(pred, data.y).item()*data.y.shape[0]
            loss_overall_counter += data.y.shape[0]
            is_close_to_wall = torch.zeros((data.y.shape[0],1),dtype=torch.bool).to(device)
            find_ = data.aux['particles_close_to_wall']
            is_close_to_wall[find_] = True
            if len(find_) > 0:
                loss_close_to_wall += loss_fn(pred[find_,:], data.y[find_,:]).item()*len(find_)
                loss_close_to_wall_counter += len(find_)
            find_ = torch.where((has_opp_neighbour==True) & (is_close_to_wall==False))[0]
            if len(find_) > 0:
                loss_with_obstacle += loss_fn(pred[find_,:], data.y[find_,:]).item()*len(find_)
                loss_with_obstacle_counter += len(find_)
            pred[data.aux['particles_close_to_wall'],:] *= 1.0+wall_bias
            data.y[data.aux['particles_close_to_wall'],:] *= 1.0+wall_bias
            pred[find_,:] *= 1.0+obstacle_bias
            data.y[find_,:] *= 1.0+obstacle_bias
            loss = loss_fn(pred, data.y)

            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_count += 1
            progress_bar.set_postfix({"loss": loss_overall/loss_overall_counter, "lr": optimizer.param_groups[0]["lr"]})
            total_step += 1

            if visualize and total_step % 10 == 0:
                train_loss_list.append((total_step, loss_overall/loss_overall_counter, loss_with_obstacle/loss_with_obstacle_counter, loss_close_to_wall/loss_close_to_wall_counter))
                loss_overall = 0.0
                loss_overall_counter = 0
                loss_close_to_wall = 0.0
                loss_close_to_wall_counter = 0
                loss_with_obstacle = 0.0    
                loss_with_obstacle_counter = 0

                # Clear and update the plot
                for ax in axes:
                    ax.clear()

                if train_loss_list:
                    train_loss_list_ = np.array(train_loss_list)
                    axes[0].scatter(train_loss_list_[:, 0], train_loss_list_[:, 1], c = 'black', label='Overall Loss')
                    axes[0].scatter(train_loss_list_[:, 0], train_loss_list_[:, 2], c = 'red', label='Obstacle Loss')
                    axes[0].scatter(train_loss_list_[:, 0], train_loss_list_[:, 3], c = 'blue', label='Close to Wall Loss')
                    axes[0].legend()
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
                    file.write(f"{train_loss_list_[-1, 0]},{train_loss_list_[-1, 1]},{train_loss_list_[-1, 2]},{train_loss_list_[-1, 3]}\n") 

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
                starting_points = [100,0,200,500,400,50,300,150,250,532,352,361,265,312,123,4,153,132,152,423,292]
                rollout_loss, rollout_mse = rolloutMSE(simulator, valid_rollout_dataset, metadata, params["noise"],
                                                       sets_to_test=list(range(1, 21)), starting_points=starting_points, rollout_length=20)
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
    wall_bias = config["training"]['wall_bias']

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
    simulator = graph_model.LearnedSimulator(hidden_size=model_params["hidden_size"], 
                                    n_mp_layers=model_params["n_mp_layers"], 
                                    window_size=model_params["window_size"])
    simulator = simulator.to(device)

    total_steps_start = 0
    checkpoint = {      "model": None,
                        "optimizer": None,
                        "scheduler": None }
    if len(load_model_path)>0:
        checkpoint = torch.load(os.path.join(model_path, load_model_path))
        simulator.load_state_dict(checkpoint["model"])
        # "2025-04-03_10_25_checkpoint_10000.pt"
        print(load_model_path)
        match = re.match(r"(.+?)_checkpoint_(\d+)\.pt", load_model_path)
        if match:
            session_name = match.group(1)  # "2025-04-03_10_25"
            total_steps_start = int(match.group(2))+1

    # train the model
    metadata = read_metadata(data_path)
    train_loss_list, eval_loss_list, onestep_mse_list, rollout_mse_list = train(params, simulator, train_loader, valid_loader, metadata, valid_rollout_dataset, 
                                                                                optimizer_state_dict=checkpoint["optimizer"],
                                                                                scheduler_state_dict=checkpoint["scheduler"],
                                                                                wall_bias=wall_bias, obstacle_bias=obstacle_bias, prefix=session_name, visualize=visualize, total_steps_start=total_steps_start)
