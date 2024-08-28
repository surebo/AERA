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

for key in env_info:
    print(str(key)+":"+ str(env_info[key]))
