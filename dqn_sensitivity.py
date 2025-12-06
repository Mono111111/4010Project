import numpy as np
import matplotlib.pyplot as plt
from environment.gym_env import SnakeGymEnv
from agents.dqn import DQN_Agent

# Sensitivity configurations
episodes = 800

test_params = {
    "gamma": [0.8, 0.9, 0.95, 0.99],
    "batch_size": [32, 64, 128],
    "lr": [1e-4, 5e-4, 1e-3],
    "epsilon_start": [1.0, 0.8, 0.5],
    "epsilon_decay_steps": [20000, 50000, 80000],
}

# Base hyperparameters
base_params = {
    "gamma": 0.95,
    "batch_size": 64,
    "lr": 1e-3,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 50000,
}

# Run a single DQN training session
def run_experiment(gamma, batch_size, lr,
                   epsilon_start, epsilon_decay_steps,
                   epsilon_end=0.05):

    env = SnakeGymEnv(render_mode=None, max_steps=2000)

    agent = DQN_Agent(
        env,
        gamma=gamma,
        batch_size=batch_size,
        lr=lr,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
    )

    history = agent.train(max_episode=episodes)
    env.close()

    return np.array(history["episode_rewards"])

# Plot curves & save figs
def plot_sensitivity(param_name, values, fixed_params, title_label):
    plt.figure(figsize=(10, 6))

    for val in values:
        print(f"  Testing {param_name} = {val}")

        params = fixed_params.copy()
        params[param_name] = val

        rewards = run_experiment(**params)

        # Moving avg
        window_size = min(50, len(rewards))
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode="valid")

        plt.plot(moving_avg, label=f"{param_name} = {val}", linewidth=2)

    plt.title(f"DQN Sensitivity – {title_label}", fontsize=15, fontweight="bold")
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Moving Average of Reward", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    filename = f"dqn_sensitivity_{param_name}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"  ✓ Saved {filename}\n")

# Run all tests
def run_all_tests():

    param_labels = {
        "gamma": "Discount Factor (γ)",
        "batch_size": "Batch Size",
        "lr": "Learning Rate",
        "epsilon_start": "Initial Exploration Rate (ε₀)",
        "epsilon_decay_steps": "Exploration Decay Steps",
    }

    print("DQN Hyperparameter Sensitivity")
    print(f"Base Config:\n{base_params}\n")

    for param_name in test_params:
        print(f"Testing: {param_labels[param_name]}")
        plot_sensitivity(param_name, test_params[param_name],
                         base_params, param_labels[param_name])

    print("✓ All DQN sensitivity plots completed!")

if __name__ == "__main__":
    run_all_tests()
