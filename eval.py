from xml.etree import ElementTree as ET
import importlib.util
import sys
import os
import requests
import argparse

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Mario")

    parser.add_argument(
        "--token",
        default="",
        type=str,
    )

    args = parser.parse_args()
    return args


args = parse_arguments()

# retrive submission meta info
xml_file_path = 'meta.xml'

tree = ET.parse(xml_file_path)
root = tree.getroot()

sub_name = ""

for book in root.findall('info'):
    sub_name =  book.find('name').text

# initializing agent
agent_path = sub_name + "_hw2_test.py"
module_name = agent_path.replace('/', '.').replace('.py', '')
spec = importlib.util.spec_from_file_location(module_name, agent_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
Agent = getattr(module, 'Agent')

os.environ["SDL_AUDIODRIVER"] = "dummy"

# evaluating
import time
from tqdm import tqdm

total_reward = 0
total_time = 0
agent = Agent()
time_limit = 120

for episode in tqdm(range(50), desc="Evaluating"):
    obs = env.reset()
    start_time = time.time()
    episode_reward = 0
    
    while True:
        action = agent.act(obs) 

        obs, reward, done, info = env.step(action)
        episode_reward += reward

        if time.time() - start_time > time_limit:
            print(f"Time limit reached for episode {episode}")
            break

        if done:
            break

    end_time = time.time()
    total_reward += episode_reward
    total_time += (end_time - start_time)

env.close()

score = total_reward / 50
print(f"Final Score: {score}")

# push to leaderboard
params = {
    'act': 'add',
    'name': sub_name,
    'score': str(score),
    'token': args.token
}
url = 'http://project.aseart.com/s/FOB2023/deep/action.php'

response = requests.get(url, params=params)
if response.ok:
    print('Success:', response.text)
else:
    print('Error:', response.status_code)