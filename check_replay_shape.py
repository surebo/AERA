from smac.env import StarCraft2Env
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--map',type=str, default='3m', help='the map of the game')
parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
args = parser.parse_args()
env = StarCraft2Env(map_name=args.map,
                    step_mul=args.step_mul, 
                    difficulty=args.difficulty, 
                    game_version='latest', 
                    replay_dir=args.replay_dir)

env_info = env.get_env_info()

obs_shape = env_info['obs_shape']
state_shape = env_info['state_shape']
n_actions = env_info['n_actions']
n_agents = env_info['n_agents']
episode_limit = env_info['episode_limit']

print("===================================================================\n",
    f"智能体数量：{n_agents} | 观察维度: {obs_shape} | 状态维度 ： {state_shape}\n",
    "===================================================================\n",
      "|观察 'o':   |",f"[5000,{episode_limit},{n_agents},{obs_shape}]      \n" 
      "-------------------------------------------------------------------\n",
      "|动作 'u':   |",f"[5000,{episode_limit},{n_agents},1]      \n",
      "-------------------------------------------------------------------\n",
      "|状态 's':   |",f"[5000,{episode_limit},{state_shape}]      \n",
      "-------------------------------------------------------------------\n",
      "|奖励 'r':   |",f"[5000,{episode_limit},1]      \n",
      "-------------------------------------------------------------------\n",
      "|'o_next':   |",f"[5000,{episode_limit},{n_agents},{obs_shape}]      \n",
      "-------------------------------------------------------------------\n",
      "|'s_next':   |",f"[5000,{episode_limit},{state_shape}]      \n",
      "-------------------------------------------------------------------\n",
      "|'u_onehot'   |",f"[5000,{episode_limit},{n_agents},{n_actions}]     \n",
      "-------------------------------------------------------------------\n",
      "|填充 'padded':|",f"[5000,{episode_limit},1]     \n",
)
