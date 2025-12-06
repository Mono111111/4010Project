import numpy as np
import matplotlib.pyplot as plt
from environment.gym_env import SnakeGymEnv

# Choose algorithm: "qlearning" or "sarsa"
MODE = "qlearning"  # change to "sarsa" to test SARSA agent

if MODE == "qlearning":
    from agents.q_learning import QLearningAgent as Agent
elif MODE == "sarsa":
    from agents.sarsa import SARSA_Agent as Agent
else:
    raise ValueError("MODE must be either 'qlearning' or 'sarsa'")

# Experiment configurations
episodes = 500
test_params = {
    "alpha": [ 0.05, 0.1, 0.2, 0.5],
    "gamma": [0.8, 0.9, 0.95, 0.99],
    "epsilon": [0.2, 0.5, 1.0],
    "epsilon_min": [0.01, 0.05, 0.1],
    "epsilon_decay": [0.98, 0.99, 0.995, 0.999],
}

# base parameters for other hyperparameters
base_params = {
    "alpha": 0.1,
    "gamma": 0.9,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995
}

# Function: run training with configs
def run_experiment(alpha, gamma, epsilon, epsilon_min, epsilon_decay):
    """Run a single training experiment with given hyperparameters"""
    env = SnakeGymEnv(render_mode=None)
    agent = Agent(
        env,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        episodes=episodes,
    )
    
    # call training method based on mode
    if MODE == "qlearning":
        rewards = agent.train()
    else:
        rewards = agent.SARSA()
    
    env.close()  # close environment
    return np.array(rewards)

# Generic plotting helper
def plot_sensitivity(param_name, values, fixed_params, title_label):
    """Plot sensitivity analysis for a single parameter"""
    plt.figure(figsize=(10, 6))
    
    for val in values:
        print(f"  Testing {param_name}={val}...")
        # copy fixed params and update the one being tested
        params = fixed_params.copy()
        params[param_name] = val
        
        # run experiment
        rewards = run_experiment(**params)
        
        # calculate moving average
        window_size = min(50, len(rewards))
        moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")
        
        # draw plot
        plt.plot(moving_avg, label=f"{param_name}={val}", linewidth=2)
    
    plt.title(f"{MODE.upper()} - Sensitivity to {title_label}", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Reward (Moving Avg)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # save figure
    filename = f"{MODE}_sensitivity_{param_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved plot: {filename}\n")

# Run all sensitivity tests
def run_all_tests():
    """Run sensitivity analysis for all hyperparameters"""
    print(f"\n{'='*60}")
    print(f"  {MODE.upper()} Hyperparameter Sensitivity Analysis")
    print(f"{'='*60}")
    print(f"Episodes per run: {episodes}")
    print(f"Base parameters: {base_params}")
    print(f"{'='*60}\n")
    
    # test parameter labels for better plot titles
    param_labels = {
        "alpha": "Learning Rate (α)",
        "gamma": "Discount Factor (γ)",
        "epsilon": "Initial Exploration Rate (ε₀)",
        "epsilon_min": "Minimum Exploration Rate (ε_min)",
        "epsilon_decay": "Exploration Decay Rate"
    }
    
    for param_name in test_params.keys():
        print(f"Testing {param_labels[param_name]}...")
        plot_sensitivity(
            param_name, 
            test_params[param_name], 
            base_params, 
            param_labels[param_name]
        )
    
    print(f"{'='*60}")
    print("✓ All sensitivity plots saved successfully!")
    print(f"{'='*60}\n")

# Main
if __name__ == "__main__":
    run_all_tests()