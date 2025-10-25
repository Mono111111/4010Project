import time
from environment.gym_env import SnakeGymEnv, ACTION2DIR

def action_name(a: int) -> str:
    """Pretty-print action id to name."""
    return ["UP", "DOWN", "LEFT", "RIGHT"][int(a)]

if __name__ == "__main__":
    # You can tweak max_steps to see truncation in action.
    env = SnakeGymEnv(render_mode="human", max_steps=500)

    episodes = 3           # run a few episodes for the demo, then exit
    print_every = 10       # print every N steps
    step_delay = 0.08      # seconds per step to slow down the random agent

    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=ep)  # seed with episode id for reproducibility in the demo
        print(f"\n=== Episode {ep} started ===")
        print(f"Initial info: score={info['score']}, energy={info['energy']}, "
              f"mystery_foods={info.get('mystery_foods', 0)}")

        done = False
        steps_in_ep = 0
        total_reward = 0.0

        while True:
            time.sleep(step_delay)  # slow down so humans can see what's happening

            # random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            steps_in_ep += 1
            total_reward += reward

            # occasional logging
            if steps_in_ep % print_every == 0 or terminated or truncated:
                print(
                    f"[EP{ep} STEP {steps_in_ep:4d}] "
                    f"action={action_name(action):5s}  "
                    f"reward={reward:+.1f}  "
                    f"score={info['score']:3d}  energy={info['energy']:3d}  "
                    f"term={terminated} trunc={truncated}"
                )

            if terminated or truncated:
                print(f"--- Episode {ep} ended --- "
                      f"steps={steps_in_ep}, total_reward={total_reward:.1f}, "
                      f"terminated={terminated}, truncated={truncated}")
                break

    print("\nDemo finished.")
