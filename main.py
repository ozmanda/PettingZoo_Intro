import glob
import os
import time

from pettingzoo.butterfly import knights_archers_zombies_v10
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy
import supersuit as ss

env = knights_archers_zombies_v10.env(render_mode="human")

#* *args and **kwargs are used to add additional arguments to the function, where *args is used to add non-keyword arguments (variables without names) 
#*  and **kwargs is used to add keyword arguments (variables with names).
def train(env_fn: knights_archers_zombies_v10, steps: int = 10000, seed: int | None = 0, **env_kwargs):
    env = env_fn.parallel_env(**env_kwargs)
    env = ss.black_death_v3(env)    # black death wrapper keeps number of agents constant even when they die

    # pre-processing using supersuit if the observation state is visual
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        # reduce colour channels: full -> grayscales the observation (computationally expensive), R/G/B: keeps this colour channel (faster and generally sufficient)
        env = ss.color_reduction_v0(env, mode="B")
        # resize from 512px to 84px
        env = ss.resize_v1(env, x_size=84, y_size=84)
        # stacks 3 frames together to capture motion: 1D arrays (vectors) are conatenated to longer 1D arrays, 2D and 3D arrays are stacked to "taller" arrays
        env = ss.frame_stack_v1(env, num_frames=3)

    env.reset(seed=seed)
    print(f'Starting training on {str(env.metadata["name"])}.')

    # takes PettingZoo ParallelEnv with the assumptions of no agent death / generation, homogeneous action and obs spaces -> gym vector environment, where each
    #   "environment" represents one agent
    env = ss.pettingzoo_env_to_vec_env_v1(env)   

    # creates num_vec_envs copies of the vec_env environment concatenated together and runs them on num_cpus.
    env = ss.concat_vec_envs_v1(vec_env=env, num_vec_envs=8, num_cpus=1, base_class='stable_baselines3')

    # Setup tensorboard logs
    logpath = './logs/'

    # use a CNN policy for visual observations and otherwise an MLP policy
    #  CnnPolicy: alias of stable_baselines3.common.policies.ActorCriticCnnPolicy
    #  MlpPolicy: alias of stable_baselines3.common.policies.ActorCriticPolicy
    model = PPO(CnnPolicy if visual_observation else MlpPolicy, 
                env, 
                verbose=3, 
                batch_size=256,
                tensorboard_log=logpath)
    
    model.learn(total_timesteps=steps)
    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")
    
    env.close()


def eval(env_fn: knights_archers_zombies_v10, num_games: int = 100, render_mode: str | None = None, **env_kwargs): 
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    # Pre-process using SuperSuit (see above)
    visual_observation = not env.unwrapped.vector_state
    if visual_observation:
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

    print(f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})")

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)
    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward

if __name__ == "__main__":
    env_fn = knights_archers_zombies_v10

    # Set vector_state to false in order to use visual observations (significantly longer training time)
    env_kwargs = dict(max_cycles=100, max_zombies=4, vector_state=True)

    # Train a model (takes ~5 minutes on a laptop CPU)
    train(env_fn, steps=100000, seed=0, **env_kwargs)

    # Evaluate 10 games (takes ~10 seconds on a laptop CPU)
    eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    # Watch 2 games (takes ~10 seconds on a laptop CPU)
    eval(env_fn, num_games=2, render_mode="human", **env_kwargs)