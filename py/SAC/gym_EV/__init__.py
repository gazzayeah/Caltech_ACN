from gym.envs.registration import register
#from .envs import *
register(
    id='EV-v0',
    entry_point='gym_EV.envs:EVEnv',
)