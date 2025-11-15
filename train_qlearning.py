import numpy as np
import matplotlib.pyplot as plt
from agents.q_learning import QLearningAgent
from environment.gym_env import SnakeGymEnv

# 
# Switch mode here: "train" or "test"
# 
MODE = "test"

def train_and_plot():
    print("Starting Q-Learning training")
    
    env = SnakeGymEnv(render_mode=None)
    agent = QLearningAgent(
        env,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        episodes=1000,
    )
    
    rewards = agent.train()
    agent.save("q_learning_agent.pkl")
    
    # Plot episode reward curve
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.6, label="Episode Reward")
    
    # Plot moving average
    window = 50
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, 
                 color='red', linewidth=2, label=f"Moving Avg ({window} episodes)")
    
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning Training Reward Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("q_learning_rewards.png")
    print("Reward curve saved to q_learning_rewards.png")
    plt.show()

def test_trained():
    print("Testing trained Q-Learning agent")
    
    env = SnakeGymEnv(render_mode="human")
    agent = QLearningAgent(env)
    agent.load("q_learning_agent.pkl")
    
    # No exploration during testing
    agent.epsilon = 0.0
    
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    
    print("Starting test episode")
    
    while not done:
        state = agent.get_state()
        q_values = agent.get_q_values(state)
        action = int(np.argmax(q_values))
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        
        env.render()
        env.game.clock.tick(10)  # Control speed, around 10 FPS
    
    print(f"\n Test finished!")
    print(f"   Total Reward: {total_reward:.2f}")
    print(f"   Final Score: {info['score']}")
    print(f"   Steps: {steps}")
    print(f"   Energy: {info['energy']}")

if __name__ == "__main__":
    if MODE == "train":
        train_and_plot()
    elif MODE == "test":
        test_trained()
    else:
        print("MODE must be 'train' or 'test'")