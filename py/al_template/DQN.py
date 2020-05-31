from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy


def DQN_train(env):
    nb_actions = env.action_space.shape[0]

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(5))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    policy = BoltzmannQPolicy()
    sarsa = SARSAAgent(model=model, nb_actions=5, nb_steps_warmup=10, policy=policy)
    sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    sarsa.fit(env, nb_steps=50000, visualize=False, verbose=2)

    # After training is done, we save the final weights.
    sarsa.save_weights('cem_{}_params.h5f'.format('EV_Charging'), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    # cem.test(env, nb_episodes=5, visualize=True)
