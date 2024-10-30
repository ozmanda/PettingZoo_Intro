import os
import ray
import supersuit as ss 
from ray import tune    # experiment runner
from pettingzoo.butterfly import cooperative_pong_v5
from ray.tune.registry import register_env
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig

ray.init(num_cpus=6)

config = {}
env_name = "cooperative_pong"

def env_creator(env_config):
    env = cooperative_pong_v5.parallel_env(render_mode=env_config.get("render_mode", "human"))
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)   # ray only has default models for speicific sizes, for example 84x84
    env = ss.frame_stack_v1(env, 3)
    return env

register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
config['env_config'] = {}

config = (
    PPOConfig()
    .environment(env=env_name)
    .framework("torch")
    .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            num_sgd_iter=10,
    )
)

savepath = f'./ray_results/{env_name}_test'
if not os.path.exists(savepath):
    os.makedirs(savepath)

tune.run(
    'PPO',
    config=config.to_dict(),
    name='PPO_cooperative_pong',
)
