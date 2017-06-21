import gym
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras import backend as K
from keras.layers.core import Lambda

env = gym.make('CartPole-v0')
action_space = 2 # {0 , 1}

# BUILDING GRAPH
state = Input(shape=(4,), name='state')
layer1 = Dense(5)(state)
layer2 = Dense(3)(layer1)
Q = Dense(2, name='Q', activation='softmax')(layer2)

def output_shape(input_shape):
    shape = list(input_shape)
    shape[-1] = 1
    return tuple(shape)

def Max(x):
    return K.max(x, keepdims=True)

q = Lambda(Max)(Q)


def Argmax(x):
    return tf.cast(K.argmax(x, axis=1), tf.float32)
action = Lambda(Argmax, output_shape=output_shape)(Q)


model = Model(inputs=state, outputs=[q, action])
model.compile(optimizer='Adam', loss='mse', loss_weights=[1,0])

env.reset()
# monitor_dir = '/home/prath/Desktop/cartpole_keras2'
# env = gym.wrappers.Monitor(env, monitor_dir, force=True)

ep_r = 0
avg_r = 0
count = 0
while avg_r < 197:
    for n in range(100):
        env.reset()

        state_curr = np.zeros((1, 4))
        state_new = np.zeros((1, 4))

        state_curr[0, 0:4], _, done, _ = env.step(0)
        _, action_curr = model.predict(state_curr, batch_size=1)

        ep_r = 0
        while not done:
            state_new[0,0:4], reward, done, _ = env.step(int(action_curr))
            Q_new, action_new = model.predict(state_new)
            model.fit(state_curr,
                      y = [Q_new + reward, action_new],
                      epochs=1,
                      verbose=0)
            state_curr = state_new
            action_curr = action_new
            ep_r += 1
        avg_r += ep_r
        print(ep_r)
    avg_r /= 100
    print(avg_r)
