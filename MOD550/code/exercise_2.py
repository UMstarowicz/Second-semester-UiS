'''
Urszula Maria Starowicz
283591
MOD550

Assignement 2
Deadline 18.02.2025

I have done this task on data with matadata save in data_metadata_oryginal.json.

'''

# === Standard Library ===
import json
import timeit as it

# === Third-Party Libraries ===
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.preprocessing import StandardScaler

# === Local/First-Party Imports ===
from mse_vanilla import mean_squared_error as vanilla_mse
from mse_numpy import mean_squared_error as numpy_mse

# Task 1: Fix mse_scaling_2 and make a check for the same results from three methods.
observed = [2, 4, 6, 8]
predicted = [2.5, 3.5, 5.5, 7.5]

factory = {
    'mse_vanilla': vanilla_mse,
    'mse_numpy': numpy_mse,
    'mse_sk': sk_mse  # Uncommented
}

# Store computed MSE values
mse_vanilla = vanilla_mse(observed, predicted)
mse_numpy = numpy_mse(observed, predicted)
mse_sklearn = sk_mse(observed, predicted)

for talker, worker in factory.items():
    exec_time = it.timeit(lambda: worker(observed, predicted), number=100) / 100
    mse = worker(observed, predicted)
    print(f"Mean Squared Error, {talker}: {mse}, Average execution time: {exec_time:.8f} seconds")

# Corrected test condition
if mse_vanilla == mse_numpy == mse_sklearn:
    print('Test successful')
else:
    print('Test failed')

# Task 2: Make a function that makes data: 1d oscillatory function with and without noise.
#Print f'Data generated: {n_points}, {range}, {other_info}'  (ask if you are uncertain on what this means)

def oscillatory_data_gen(n_points=100, x_range=(0, 10), noise_level=0.0, seed=None, 
                         save_metadata=True, metadata_filename="data_metadata.json"):
    """
    Generates oscillatory data (sinusoidal with optional noise) and saves metadata for reproducibility.

    Parameters:
        n_points (int): Number of data points.
        x_range (tuple): Range of x values (start, end).
        noise_level (float): Standard deviation of Gaussian noise.
        seed (int, optional): Random seed for reproducibility.
        save_metadata (bool): Whether to save metadata.
        metadata_filename (str): Filename to save metadata.

    Returns:
        x (numpy array): Generated x values.
        y (numpy array): Corresponding y values with noise.
    """
    if seed is not None:
        np.random.seed(seed)  # Set seed for reproducibility

    x = np.linspace(x_range[0], x_range[1], n_points)
    y_clean = np.sin(x)

    noise = np.random.normal(0, noise_level, n_points)
    y = y_clean + noise

    metadata = {
        "n_points": n_points,
        "x_range": x_range,
        "noise_level": noise_level,
        "seed": seed
    }

    if save_metadata:
        with open(metadata_filename, "w") as f:
            json.dump(metadata, f, indent=4)

    print(f'Data generated: {n_points} points, Range: {x_range}, Noise level: {noise_level}, Seed: {seed}')
    print(f'Metadata saved to {metadata_filename}')

    return x, y

# Example usage
x, y = oscillatory_data_gen(200, (0, 20), 0.5, seed=42, save_metadata=True)

# Task 3: Use clustering (pick the method you prefer) to group data and print the variance
# as a function of the number of clusters.  
# Print the info about your clustering method and its parameters, plot the variance vs n.of cluster 

def clusers(data, clusters_n=10):
    variances = []
    print("Clustering method: K-Means")
    print(f'Number of clusters: {clusters_n}')

    for i in range(1, clusters_n+1):
        kmeans = KMeans(n_clusters=i, n_init=10, max_iter=300, random_state=123)
        kmeans.fit(data.reshape(-1, 1))
        variance = kmeans.inertia_
        variances.append(variance)
        print(f'Clusters: {i}, Variance: {variance:.4f}')

clusers(y,10)

# Task 4: Use LR,  NN and PINNS to make a regression of such data.  
#Print 'Task completed {regression_method}'

def LR_model(x, y):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)  # Normalize x

    lr_model = LinearRegression()
    lr_model.fit(x_scaled, y)
    lr_predict = lr_model.predict(x_scaled)

    error_lr = np.abs(y - lr_predict)
    print("Task completed: Linear Regression")
    return lr_predict, error_lr

def NN_model(x, y, max_iter=2000):
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)  # Normalize x

    nn_model = MLPRegressor(
        hidden_layer_sizes=(20, 20), activation='relu', solver='adam',
        max_iter=1, warm_start=True, learning_rate_init=0.01, random_state=42
    )

    nn_loss = []

    for i in range(1, max_iter + 1):
        nn_model.partial_fit(x_scaled, y)

        if i % 100 == 0:
            loss = np.mean((nn_model.predict(x_scaled) - y) ** 2)
            nn_loss.append(loss)

    nn_predict = nn_model.predict(x_scaled)
    nn_error = np.abs(y - nn_predict)

    print("Task completed: Neural Network")
    return nn_predict, nn_loss, nn_error

def PINN_model(x, y, hidden_layers=(20, 20), activation=nn.Tanh(), lr=0.01, epochs=2000):
    class PINN(nn.Module):
        def __init__(self):
            super(PINN, self).__init__()
            layers = []
            input_dim = 1
            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(activation)
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    x_torch = torch.tensor(x.reshape(-1, 1), dtype=torch.float32)
    y_torch = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    pinn_model = PINN()
    optimizer = optim.Adam(pinn_model.parameters(), lr=lr)
    loss_function = nn.MSELoss()

    pinn_loss = []
    pinn_errors = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_predict = pinn_model(x_torch)
        loss = loss_function(y_predict, y_torch)
        loss.backward()
        optimizer.step()

        # Store loss at specific intervals
        if epoch % 100 == 0:
            pinn_loss.append(loss.item())
        if epoch % 10 == 0:
            pinn_errors.append(loss.item())
    print("Task completed: PINN model")
    return pinn_model(x_torch).detach().numpy(), pinn_loss, pinn_errors

lr_predict, lr_error = LR_model(x,y)
nn_predict, nn_loss, nn_error = NN_model(x,y)
pinn_predict, pinn_loss, pinn_error = PINN_model(x,y)

# Task 5: Plot the solution as a function of the number of executed iterations (NN and PINNs).  
# Plot the regression function as a function of iteration numbers (tip. use a while loop with a pause function) 

plt.figure(figsize=(10, 5))
plt.plot(range(0, 2000, 100), nn_loss, label='NN Loss', marker='o')
plt.plot(range(0, 2000, 100), pinn_loss, label='PINN Loss', marker='s')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss vs Iterations")
plt.show()

# Task 6: Plot the error in respect to the truth as a function of the executed iteration (LR, NN and PINNs).

# Create a 3x2 grid of subplots
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

# 1) Error Convergence Over Iterations
axes[0, 0].plot(range(0, len(pinn_error) * 10, 10), pinn_error, label='PINN Error', color='green')
axes[0, 0].plot(range(0, len(nn_error) * 10, 10), nn_error, label='NN Error', color='red')
axes[0, 0].plot(range(0, len(lr_error) * 10, 10), lr_error, label='LR Error', color='blue')
axes[0, 0].set_xlabel("Iteration Number")
axes[0, 0].set_ylabel("Mean Squared Error (MSE)")
axes[0, 0].legend()
axes[0, 0].set_title("Error Convergence Over Iterations")

# 2) Linear Regression Fit
axes[0, 1].plot(x, y, label="True Function", color="black")
axes[0, 1].plot(x, lr_predict, label="Linear Regression", linestyle="--", color="red")
axes[0, 1].fill_between(x, y, lr_predict, color="gray", alpha=0.3, label="Error")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("Function Value")
axes[0, 1].set_title("Linear Regression Fit")
axes[0, 1].legend()
axes[0, 1].grid(True)

# 3) Neural Network Loss Over Epochs
axes[1, 0].plot(range(len(nn_loss)), nn_loss, label="NN Loss", color='blue')
axes[1, 0].set_xlabel("Epochs")
axes[1, 0].set_ylabel("Loss")
axes[1, 0].set_title("Neural Network Loss Over Epochs")
axes[1, 0].legend()
axes[1, 0].grid(True)

# 4) Neural Network Fit
axes[1, 1].plot(x, y, label="True Function", color="black")
axes[1, 1].plot(x, nn_predict, label="Neural Network", linestyle="--", color="red")
axes[1, 1].fill_between(x, y, nn_predict.squeeze(), color="gray", alpha=0.3, label="Error")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("Function Value")
axes[1, 1].set_title("Neural Network Fit")
axes[1, 1].legend()
axes[1, 1].grid(True)

# 5) PINN Loss Over Epochs
axes[2, 0].plot(range(len(pinn_loss)), pinn_loss, label="PINN Loss", color='green')
axes[2, 0].set_xlabel("Epochs")
axes[2, 0].set_ylabel("Loss")
axes[2, 0].set_title("PINN Loss Over Epochs")
axes[2, 0].legend()
axes[2, 0].grid(True)

# 6) PINN Fit
axes[2, 1].plot(x, y, label="True Function", color="black")
axes[2, 1].plot(x, pinn_predict, label="PINN", linestyle="--", color="red")
axes[2, 1].fill_between(x, y, pinn_predict.squeeze(), color="gray", alpha=0.3, label="Error")
axes[2, 1].set_xlabel("x")
axes[2, 1].set_ylabel("Function Value")
axes[2, 1].set_title("PINN Fit")
axes[2, 1].legend()
axes[2, 1].grid(True)

plt.tight_layout()
plt.show()

# Task 7: Assume to not know the truth any longer, select a method to monitor the progress of the calculations and plot its outcome (LR, NN and PINNs).

mse_lr = sk_mse(y, lr_predict)

plt.figure(figsize=(10, 5))

# Linear Regression MSE (constant since it's a closed-form solution)
plt.axhline(y=mse_lr, color='r', linestyle='--', label="LR MSE")

# Neural Network Loss
plt.plot(nn_loss, label="NN Loss", color='blue')

# PINN Loss
plt.plot(pinn_loss, label="PINN Loss", color='green')

plt.xlabel("Epochs")
plt.ylabel("Loss / MSE")
plt.title("Monitoring Progress: LR, NN, and PINN Loss")
plt.legend()
plt.grid()
plt.show()

# Task 8: Run the reinforcement learning script.

# GridWorld Environment
class GridWorld:
    """GridWorld environment with obstacles and a goal.
    The agent starts at the top-left corner and has to reach the bottom-right corner.
    The agent receives a reward of -1 at each step, a reward of -0.01 at each step in an obstacle, and a reward of 1 at the goal.

    Args:
        size (int): The size of the grid.
        num_obstacles (int): The number of obstacles in the grid.

    Attributes:
        size (int): The size of the grid.
        num_obstacles (int): The number of obstacles in the grid.
        obstacles (list): The list of obstacles in the grid.
        state_space (numpy.ndarray): The state space of the grid.
        state (tuple): The current state of the agent.
        goal (tuple): The goal state of the agent.

    Methods:
        generate_obstacles: Generate the obstacles in the grid.
        step: Take a step in the environment.
        reset: Reset the environment.
    """
    def __init__(self, size=5, num_obstacles=5):
        self.size = size
        self.num_obstacles = num_obstacles
        self.obstacles = [(0, 4), (4, 3), (1, 3), (1, 0), (3, 2)]
        self.state_space = np.zeros((self.size, self.size))
        self.state = (0, 0)
        self.goal = (self.size-1, self.size-1)

    def step(self, action):
        """
        Take a step in the environment.
        The agent takes a step in the environment based on the action it chooses.

        Args:
            action (int): The action the agent takes.
                0: up
                1: right
                2: down
                3: left

        Returns:
            state (tuple): The new state of the agent.
            reward (float): The reward the agent receives.
            done (bool): Whether the episode is done or not.
        """
        x, y = self.state

        if action == 0:  # up
            x = max(0, x-1)
        elif action == 1:  # right
            y = min(self.size-1, y+1)
        elif action == 2:  # down
            x = min(self.size-1, x+1)
        elif action == 3:  # left
            y = max(0, y-1)
        self.state = (x, y)
        if self.state in self.obstacles:
         #   self.state = (0, 0)
            return self.state, -1, True
        if self.state == self.goal:
            return self.state, 1, True
        return self.state, -0.1, False

    def reset(self):
        """
        Reset the environment.
        The agent is placed back at the top-left corner of the grid.

        Args:
            None

        Returns:
            state (tuple): The new state of the agent.
        """
        self.state = (0, 0)
        return self.state

# Q-Learning
class QLearning:
    """
    Q-Learning agent for the GridWorld environment.

    Args:
        env (GridWorld): The GridWorld environment.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        episodes (int): The number of episodes to train the agent.

    Attributes:
        env (GridWorld): The GridWorld environment.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        episodes (int): The number of episodes to train the agent.
        q_table (numpy.ndarray): The Q-table for the agent.

    Methods:
        choose_action: Choose an action for the agent to take.
        update_q_table: Update the Q-table based on the agent's experience.
        train: Train the agent in the environment.
        save_q_table: Save the Q-table to a file.
        load_q_table: Load the Q-table from a file.
    """
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.1, episodes=10):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((self.env.size, self.env.size, 4))

    def choose_action(self, state):
        """
        Choose an action for the agent to take.
        The agent chooses an action based on the epsilon-greedy policy.

        Args:
            state (tuple): The current state of the agent.

        Returns:
            action (int): The action the agent takes.
                0: up
                1: right
                2: down
                3: left
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1, 2, 3])  # exploration
        else:
            return np.argmax(self.q_table[state])  # exploitation

    def update_q_table(self, state, action, reward, new_state):
        """
        Update the Q-table based on the agent's experience.
        The Q-table is updated based on the Q-learning update rule.

        Args:
            state (tuple): The current state of the agent.
            action (int): The action the agent takes.
            reward (float): The reward the agent receives.
            new_state (tuple): The new state of the agent.

        Returns:
            None
        """
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
            self.alpha * (reward + self.gamma * np.max(self.q_table[new_state]))

    def train(self):
        """
        Train the agent in the environment.
        The agent is trained in the environment for a number of episodes.
        The agent's experience is stored and returned.

        Args:
            None

        Returns:
            rewards (list): The rewards the agent receives at each step.
            states (list): The states the agent visits at each step.
            starts (list): The start of each new episode.
            steps_per_episode (list): The number of steps the agent takes in each episode.
        """
        rewards = []
        states = []  # Store states at each step
        starts = []  # Store the start of each new episode
        steps_per_episode = []  # Store the number of steps per episode
        steps = 0  # Initialize the step counter outside the episode loop
        episode = 0
        while episode < self.episodes:
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(state)
                new_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, new_state)
                state = new_state
                total_reward += reward
                states.append(state)  # Store state
                steps += 1  # Increment the step counter
                if done and state == self.env.goal:  # Check if the agent has reached the goal
                    starts.append(len(states))  # Store the start of the new episode
                    rewards.append(total_reward)
                    steps_per_episode.append(steps)  # Store the number of steps for this episode
                    steps = 0  # Reset the step counter
                    episode += 1
        return rewards, states, starts, steps_per_episode

for i in range(1):
    env = GridWorld(size=5, num_obstacles=5)
    agent = QLearning(env)

    # Train the agent and get rewards
    rewards, states, starts, steps_per_episode = agent.train()  # Get starts and steps_per_episode as well

    # Visualize the agent moving in the grid
    fig, ax = plt.subplots(figsize=(8, 6))

    def update(i):
        """
        Update the grid with the agent's movement.

        Args:
            i (int): The current step.

        Returns:
            None
        """
        ax.clear()
        # Calculate the cumulative reward up to the current step
        cumulative_reward = sum(rewards[:i+1])
        # Find the current episode
        current_episode = next((j for j, start in enumerate(starts) if start > i), len(starts)) - 1
        # Calculate the number of steps since the start of the current episode
        if current_episode < 0:
            steps = i + 1
        else:
            steps = i - starts[current_episode] + 1
        ax.set_title(f"Episode: {current_episode+1}, Number of Steps to Reach Target: {steps}")
        grid = np.zeros((env.size, env.size))
        for obstacle in env.obstacles:

            grid[obstacle] = -1
        grid[env.goal] = 1
        grid[states[i]] = 0.5  # Use states[i] instead of env.state
        ax.imshow(grid, cmap='magma')

    ani = animation.FuncAnimation(fig, update, frames=range(len(states)), repeat=False)
    print(f"Environment number {i+1}")
    for i, steps in enumerate(steps_per_episode, 1):
        print(f"Episode {i}: {steps} Number of Steps to Reach Target ")
    print()

# GridWorld Environment
class GridWorld:
    def __init__(self, size=5, num_obstacles=5):
        self.size = size
        self.num_obstacles = num_obstacles
        self.obstacles = [(0, 4), (4, 3), (1, 3), (1, 0), (3, 2)]
        self.state = (0, 0)
        self.goal = (self.size-1, self.size-1)

    def step(self, action):
        x, y = self.state
        if action == 0:  # up
            x = max(0, x-1)
        elif action == 1:  # right
            y = min(self.size-1, y+1)
        elif action == 2:  # down
            x = min(self.size-1, x+1)
        elif action == 3:  # left
            y = max(0, y-1)
        self.state = (x, y)
        if self.state in self.obstacles:
            return self.state, -1, True
        if self.state == self.goal:
            return self.state, 1, True
        return self.state, -0.1, False

    def reset(self):
        self.state = (0, 0)
        return self.state

# Q-Learning Agent
class QLearning:
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.1, episodes=500):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((self.env.size, self.env.size, 4))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1, 2, 3])
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, new_state):
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
            self.alpha * (reward + self.gamma * np.max(self.q_table[new_state]))

    def train(self):
        steps_per_episode = []
        for episode in range(self.episodes):
            state = self.env.reset()
            done = False
            steps = 0
            while not done:
                action = self.choose_action(state)
                new_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, new_state)
                state = new_state
                steps += 1
                if done and state == self.env.goal:
                    steps_per_episode.append(steps)
                    break
        return steps_per_episode

# Test different learning rates
learning_rates = np.linspace(0.1, 1.0, 10)
convergence_steps = []

for alpha in learning_rates:
    env = GridWorld(size=5, num_obstacles=5)
    agent = QLearning(env, alpha=alpha, episodes=500)
    steps_per_episode = agent.train()
    convergence_steps.append(np.mean(steps_per_episode[-50:]))  # Average over last 50 episodes

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(learning_rates, convergence_steps, marker='o', linestyle='-')
plt.xlabel("Learning Rate (alpha)")
plt.ylabel("Avg Steps to Converge")
plt.title("Effect of Learning Rate on Convergence Speed")
plt.grid(True)
plt.show()
