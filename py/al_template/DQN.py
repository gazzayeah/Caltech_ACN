import numpy as np

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Input, Activation, Concatenate
from keras.optimizers import *

from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


class MujocoProcessor(WhiteningNormalizerProcessor):
    def process_action(self, action):
        return np.clip(action, -1., 1.)


def DQN_train(env):
    nb_actions = env.action_space.shape[0]

    # Next, we build a very simple model.
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(400))
    actor.add(Activation('relu'))
    actor.add(Dense(300))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('tanh'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Dense(400)(flattened_observation)
    x = Activation('relu')(x)
    x = Concatenate()([x, action_input])
    x = Dense(300)(x)
    x = Activation('relu')(x)
    x = Dense(5)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.1)
    agent = DQNAgent(
        critic,
        policy=None,
        test_policy=None,
        enable_double_dqn=True,
        enable_dueling_network=False,
        dueling_type='avg',
        nb_actions=nb_actions,
        memory=memory
    )
    agent.compile([Adam(lr=1e-3), Adam(lr=1e-3)], metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    # agent.fit(env, nb_steps=50000, visualize=False, verbose=1, nb_max_episode_steps=200)

    # After training is done, we save the final weights.
    agent.save_weights('dqn_{}_weights.h5f'.format('EV_Charging'), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    # agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)
