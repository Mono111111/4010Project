# compare_algorithms.py
import numpy as np
import matplotlib.pyplot as plt
from environment.gym_env import SnakeGymEnv
from agents.q_learning import QLearningAgent
from agents.sarsa import SARSA_Agent, get_state, argmax_Q
from agents.dqn import DQN_Agent
from agents.agent import Greedy_Agent
import time

class AlgorithmComparison:
    def __init__(self, episodes=500, eval_episodes=50, max_steps=2000,
                 alpha=0.1, gamma=0.95, epsilon=1.0, 
                 epsilon_min=0.05, epsilon_decay=0.995):
        self.episodes = episodes
        self.eval_episodes = eval_episodes
        self.max_steps = max_steps
        
        # Unified hyperparameters for fair comparison
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.results = {}
        
        print("\n" + "="*60)
        print("  UNIFIED HYPERPARAMETERS (Fair Comparison)")
        print("="*60)
        print(f"Learning Rate (α):       {self.alpha}")
        print(f"Discount Factor (γ):     {self.gamma}")
        print(f"Initial Epsilon (ε):     {self.epsilon}")
        print(f"Min Epsilon:             {self.epsilon_min}")
        print(f"Epsilon Decay:           {self.epsilon_decay}")
        print(f"Episodes:                {self.episodes}")
        print(f"Max Steps per Episode:   {self.max_steps}")
        print("="*60)
    
    def train_qlearning(self):
        """Train Q-Learning agent"""
        print("\n" + "="*60)
        print(" Training Q-Learning Agent...")
        print("="*60)
        
        env = SnakeGymEnv(render_mode=None, max_steps=self.max_steps)
        agent = QLearningAgent(
            env,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            episodes=self.episodes,
        )
        
        start_time = time.time()
        rewards = agent.train()
        training_time = time.time() - start_time
        
        # Save model
        agent.save("q_learning_agent.pkl")
        
        self.results['Q-Learning'] = {
            'rewards': rewards,
            'training_time': training_time,
            'q_table_size': len(agent.q_table)
        }
        
        print(f" Q-Learning training completed in {training_time:.2f}s")
        print(f" Q-table size: {len(agent.q_table)} states")
        
        return agent
    
    def train_sarsa(self):
        """Train SARSA agent"""
        print("\n" + "="*60)
        print(" Training SARSA Agent...")
        print("="*60)
        
        env = SnakeGymEnv(render_mode=None, max_steps=self.max_steps)
        agent = SARSA_Agent(
            env,
            alpha=self.alpha,           # Use unified hyperparameters
            gamma=self.gamma,           # Use unified hyperparameters
            epsilon=self.epsilon,       # Use unified hyperparameters
            episodes=self.episodes,
            epsilon_min=self.epsilon_min,  # Use unified hyperparameters
            epsilon_decay=self.epsilon_decay,  # Use unified hyperparameters
        )
        
        start_time = time.time()
        episode_rewards = agent.SARSA()
        training_time = time.time() - start_time
        
        # Save model
        agent.save("sarsa_q_table.pkl")
        
        self.results['SARSA'] = {
            'rewards': episode_rewards,
            'training_time': training_time,
            'q_table_size': len(agent.Q)
        }
        
        print(f" SARSA training completed in {training_time:.2f}s")
        print(f" Q-table size: {len(agent.Q)} states")
        
        return agent
    
    def train_dqn(self):
        print("\n" + "="*60)
        print(" Training DQN Agent...")
        print("="*60)
    
        env = SnakeGymEnv(render_mode=None, max_steps=self.max_steps)

        # Use the same hyperparameters as your standalone train_dqn.py
        agent = DQN_Agent(
            env,
            gamma=0.99,                 # more stable long-term reward propagation
            lr=1e-3,
            batch_size=64,
            replay_capacity=100000,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=50_000,  # faster decay → starts exploiting earlier
            max_steps_per_episode=self.max_steps,
            hidden_dim=128,
            target_update_freq=500,     # sync target net more often
        )

        start_time = time.time()
        history = agent.train(max_episode=self.episodes)
        training_time = time.time() - start_time

        # Save trained model
        agent.save("dqn_snake.pt")

        # Count total parameters
        total_params = sum(p.numel() for p in agent.q_net.parameters())

        # Store results for later comparison
        self.results['DQN'] = {
            'rewards': history['episode_rewards'],
            'steps': history['episode_steps'],
            'training_time': training_time,
            'network_params': total_params
        }

        print(f" DQN training completed in {training_time:.2f}s")
        print(f" Network parameters: {total_params:,}")
        print(f" Final average reward (last 50 eps): {np.mean(history['episode_rewards'][-50:]):.2f}")

        return agent

    def evaluate_agent(self, agent_type, model_path):
        """Evaluate trained agent"""
        print(f"\n Evaluating {agent_type}...")
        
        env = SnakeGymEnv(render_mode=None, max_steps=self.max_steps)
        
        if agent_type == 'Q-Learning':
            agent = QLearningAgent(env)
            agent.load(model_path)
            agent.epsilon = 0.0  # Greedy policy
        elif agent_type == 'SARSA':  # SARSA
            agent = SARSA_Agent(env)
            agent.load(model_path)
            agent.epsilon = 0.0
        elif agent_type == 'DQN':
            agent = DQN_Agent(env, max_steps_per_episode=self.max_steps)
            agent.load(model_path)
        
        eval_rewards = []
        eval_steps = []
        eval_scores = []
        eval_energies = []
        
        for ep in range(self.eval_episodes):
            obs, info = env.reset(seed=ep)
            terminated = False
            truncated = False
            total_reward = 0.0
            step = 0
            
            while not (terminated or truncated):
                if agent_type == 'Q-Learning':
                    state = agent.get_state()
                    q_values = agent.get_q_values(state)
                    action = int(np.argmax(q_values))
                elif agent_type == 'SARSA':  # SARSA
                    state = get_state(env)
                    action = int(argmax_Q(agent.Q, state, agent.n_actions))
                elif agent_type == 'DQN':
                    state = agent.get_state()
                    action = agent._epsilon_greedy(state, greedy=True)
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step += 1
            
            eval_rewards.append(total_reward)
            eval_steps.append(step)
            eval_scores.append(info['score'])
            eval_energies.append(info['energy'])
        
        eval_results = {
            'rewards': eval_rewards,
            'steps': eval_steps,
            'scores': eval_scores,
            'energies': eval_energies,
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_steps': np.mean(eval_steps),
            'avg_score': np.mean(eval_scores),
            'max_score': np.max(eval_scores),
            'success_rate': sum(1 for s in eval_scores if s >= 50) / self.eval_episodes * 100
        }
        
        self.results[agent_type]['evaluation'] = eval_results
        
        print(f"   Avg Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"   Avg Steps: {eval_results['avg_steps']:.1f}")
        print(f"   Avg Score: {eval_results['avg_score']:.1f}")
        print(f"   Max Score: {eval_results['max_score']}")
        print(f"   Success Rate (score≥50): {eval_results['success_rate']:.1f}%")
        
        return eval_results
    
    def plot_comparison(self):
        """Plot comprehensive comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # 1. Training Rewards Curve
        ax1 = axes[0, 0]
        colors = {'Q-Learning': 'blue', 'SARSA': 'green', 'DQN': 'red'}
        for name in ['Q-Learning', 'SARSA', 'DQN']:
            if name in self.results:
                rewards = self.results[name]['rewards']
                ax1.plot(rewards, alpha=0.3, color=colors[name], label=f'{name} (raw)')
                
                # Moving average
                window = 50
                if len(rewards) >= window:
                    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    ax1.plot(range(window-1, len(rewards)), moving_avg, 
                            color=colors[name], linewidth=2, label=f'{name} (MA-{window})')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Training Reward Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Evaluation Rewards Distribution
        ax2 = axes[0, 1]
        eval_data = []
        labels = []
        for name in ['Q-Learning', 'SARSA', 'DQN']:
            if name in self.results and 'evaluation' in self.results[name]:
                eval_data.append(self.results[name]['evaluation']['rewards'])
                labels.append(name)
        
        if eval_data:
            bp = ax2.boxplot(eval_data, labels=labels, patch_artist=True)
            for patch, name in zip(bp['boxes'], labels):
                patch.set_facecolor(colors[name])
                patch.set_alpha(0.6)
            ax2.set_ylabel('Reward')
            ax2.set_title('Evaluation Reward Distribution')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Scores Comparison
        ax3 = axes[1, 0]
        scores_data = []
        for name in ['Q-Learning', 'SARSA', 'DQN']:
            if name in self.results and 'evaluation' in self.results[name]:
                scores_data.append(self.results[name]['evaluation']['scores'])
        
        if scores_data:
            bp = ax3.boxplot(scores_data, labels=labels, patch_artist=True)
            for patch, name in zip(bp['boxes'], labels):
                patch.set_facecolor(colors[name])
                patch.set_alpha(0.6)
            ax3.set_ylabel('Score')
            ax3.set_title('Game Score Distribution')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Summary Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = " Algorithm Comparison Summary\n\n"
        
        for name in ['Q-Learning', 'SARSA', 'DQN']:
            if name in self.results:
                summary_text += f"{'='*40}\n"
                summary_text += f"{name}:\n"
                summary_text += f"{'='*40}\n"
                summary_text += f"Training Time: {self.results[name]['training_time']:.2f}s\n"
                
                if name == 'DQN':
                    summary_text += f"Network Params: {self.results[name]['network_params']:,}\n"
                else:
                    summary_text += f"Q-table Size: {self.results[name].get('q_table_size', 'N/A')} states\n"
                
                if 'evaluation' in self.results[name]:
                    eval_res = self.results[name]['evaluation']
                    summary_text += f"\nEvaluation ({self.eval_episodes} episodes):\n"
                    summary_text += f"  Avg Reward: {eval_res['avg_reward']:.2f} ± {eval_res['std_reward']:.2f}\n"
                    summary_text += f"  Avg Steps: {eval_res['avg_steps']:.1f}\n"
                    summary_text += f"  Avg Score: {eval_res['avg_score']:.1f}\n"
                    summary_text += f"  Max Score: {eval_res['max_score']}\n"
                    summary_text += f"  Success Rate: {eval_res['success_rate']:.1f}%\n"
                summary_text += "\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=300)
        print("\n Comparison plot saved to 'algorithm_comparison.png'")
        plt.show()
    
    def print_detailed_comparison(self):
        """Print detailed comparison table"""
        print("\n" + "="*100)
        print(" DETAILED ALGORITHM COMPARISON")
        print("="*100)
        
        print(f"\n{'Metric':<30} {'Q-Learning':<15} {'SARSA':<15} {'DQN':<15} {'Winner':<10}")
        print("-"*100)
        
        metrics = [
            ('Training Time (s)', 'training_time', 'lower'),
            ('Q-table Size', 'q_table_size', 'lower'),
            ('Avg Reward', 'evaluation.avg_reward', 'higher'),
            ('Reward Std Dev', 'evaluation.std_reward', 'lower'),
            ('Avg Steps', 'evaluation.avg_steps', 'higher'),
            ('Avg Score', 'evaluation.avg_score', 'higher'),
            ('Max Score', 'evaluation.max_score', 'higher'),
            ('Success Rate (%)', 'evaluation.success_rate', 'higher'),
        ]
        
        algorithm_name = ['Q-Learning', 'SARSA', 'DQN']
        
        for metric_name, metric_key, better in metrics:
            values = {}
            for name in algorithm_name:
                if name in self.results:
                    if metric_key == 'model_size':
                        # Special handling for model size
                        if name == 'DQN':
                            val = f"{self.results[name]['network_params']:,} params"
                        else:
                            val = f"{self.results[name].get('q_table_size', 'N/A')} states"
                        values[name] = val
                    else:
                        keys = metric_key.split('.')
                        val = self.results[name]
                        for k in keys:
                            if isinstance(val, dict) and k in val:
                                val = val[k]
                            else:
                                val = None
                                break
                        values[name] = val
            
            # Format values
            q_val = values.get('Q-Learning')
            s_val = values.get('SARSA')
            d_val = values.get('DQN')
            
            #Winner
            winner = "N/A"
            numeric_vals = {}
            for algo, v in [('Q-Learning', q_val), ('SARSA', s_val), ('DQN', d_val)]:
                if isinstance(v, (int, float)):
                    numeric_vals[algo] = v
            if numeric_vals:
                if better == 'higher':
                    winner = max(numeric_vals, key=numeric_vals.get)
                else:
                    winner = min(numeric_vals, key=numeric_vals.get)

            # Format display
            if metric_key == 'model_size':
                #just for ensure
                # Model size is already formatted as string
                q_str = str(q_val) if q_val is not None else "N/A"
                s_str = str(s_val) if s_val is not None else "N/A"
                d_str = str(d_val) if d_val is not None else "N/A"
                print(f"{metric_name:<30} {str(q_val):<20} {str(s_val):<20} {str(d_val):<20}{winner:<10}")
            elif all(isinstance(v, (int, float)) for v in [q_val, s_val, d_val] if v is not None):
                # Numeric comparison
                q_str = f"{q_val:.2f}" if q_val is not None else "N/A"
                s_str = f"{s_val:.2f}" if s_val is not None else "N/A"
                d_str = f"{d_val:.2f}" if d_val is not None else "N/A"
                print(f"{metric_name:<30} {q_str:<20} {s_str:<20} {d_str:<20}{winner:<10}")
            else:
                q_str = str(q_val) if q_val is not None else "N/A"
                s_str = str(s_val) if s_val is not None else "N/A"
                d_str = str(d_val) if d_val is not None else "N/A"
                print(f"{metric_name:<30} {q_str:<20} {s_str:<20} {d_str:<20}{winner:<10}")
        
        print("="*100)

def main():
    print(" Starting Algorithm Comparison: Q-Learning vs SARSA vs DQN")
    print("="*60)
    
    # Use unified hyperparameters for fair comparison
    comparison = AlgorithmComparison(
        episodes=500,
        eval_episodes=50,
        max_steps=2000,
        alpha=0.1,          # Unified learning rate
        gamma=0.95,         # Unified discount factor
        epsilon=1.0,        # Unified initial exploration
        epsilon_min=0.05,   # Unified minimum exploration
        epsilon_decay=0.995 # Unified exploration decay
    )
    
    # Train both agents
    comparison.train_qlearning()
    comparison.train_sarsa()
    comparison.train_dqn()
    
    # Evaluate both agents
    comparison.evaluate_agent('Q-Learning', 'q_learning_agent.pkl')
    comparison.evaluate_agent('SARSA', 'sarsa_q_table.pkl')
    comparison.evaluate_agent('DQN', 'dqn_snake.pt')
    
    # Print detailed comparison
    comparison.print_detailed_comparison()
    
    # Plot comparison
    comparison.plot_comparison()
    
    print("\n Comparison completed!")
    print(" Note: Both algorithms used identical hyperparameters for fair comparison.")

if __name__ == "__main__":
    main()