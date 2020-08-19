import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding
import numpy as np


class TESTenv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        print('Init Printing')

    def step(self, action):
        print('step')

    def reset(self):
        print('reset')

    def render(self, mode='human'):
        print('render')

    def close(self):
        print('close')
