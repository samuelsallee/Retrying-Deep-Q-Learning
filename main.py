import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Input, Dense
from keras.optimizer_v1 import Adam
from keras.callbacks import TensorBoard
from collections import deque
import numpy as np
import time
import Game_Sam
import random

tf.compat.v1.disable_eager_execution()

DISCOUNT = 0.999
REPLAY_MEMORY_SIZE = 500_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = 30  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 50_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = epsilon/3000
MIN_EPSILON = 0.01

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

random.seed(int(time.time()))
np.random.seed(int(time.time()))
tf.random.set_seed(int(time.time()))

ACTION_SPACE = 9

# Memory fraction, used mostly when training multiple agents
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
# backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

if not os.path.isdir('logs'):
    os.makedirs('logs')

if not os.path.isdir('logs/' + MODEL_NAME):
    os.makedirs('logs/' + MODEL_NAME)

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_dir = 'C:/Users/samsa/PycharmProjects/pythonProject8/logs/'
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir + MODEL_NAME)
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    # Added because of version
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


def create_model(input_size, layer1_width, layer2_width, output_size):
    model = Sequential([
        Input(input_size),
        Dense(layer1_width, activation='relu'),
        Dense(layer1_width, activation='relu'),
        Dense(layer2_width, activation='relu'),
        Dense(layer2_width, activation='relu'),
        Dense(output_size, activation='linear')
    ])
    model.compile(loss="mse", optimizer=Adam(lr=0.001))
    return model

def get_score_mem(timestamp):
    return timestamp[-1]
# Agent class
class DQNAgent:
    def __init__(self, input_size, layer1_width, layer2_width, output_size, min_chkpt_size):

        # Main model
        self.model = create_model(input_size, layer1_width, layer2_width, output_size)
        # self.model = load_model('models/2x64___765.00max__208.00avg___78.00min__1646024275.model')

        # Target network
        self.target_model = create_model(input_size, layer1_width, layer2_width, output_size)
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.sort_replay_at = int(REPLAY_MEMORY_SIZE * .5)
        self.sort_replay_ctr = 0

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir=f"C:/Users/samsa/PycharmProjects/pythonProject8/logs/{MODEL_NAME}")

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        self.checkpoint_ctr = 0
        self.checkpoint_replay = deque(maxlen=1000)
        self.min_chkpt_size = min_chkpt_size
        self.current_run = deque(maxlen=65)





    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)

    def train_chkpt(self):
        if len(self.checkpoint_replay) < self.min_chkpt_size:
            return

        minibatch0 = random.sample(self.checkpoint_replay, MINIBATCH_SIZE)
        rang = min([len(x) for x in minibatch0])
        for i in range(1, rang-1):
            minibatch = [x[-i] for x in minibatch0]
            current_states = np.array([transition[0] for transition in minibatch])
            current_qs_list = self.model.predict(current_states)

            new_current_states = np.array([transition[3] for transition in minibatch])
            future_qs_list = self.target_model.predict(new_current_states)

            X = []
            y = []

            # Now we need to enumerate our batches
            for index, (current_state, action, reward, new_current_state, done, score) in enumerate(minibatch):

                # If not a terminal state, get new q from future states, otherwise set it to 0
                # almost like with Q Learning, but we use just part of equation here
                if not done:
                    max_future_q = np.max(future_qs_list[index])
                    new_q = reward + DISCOUNT * max_future_q
                else:
                    new_q = reward

                # Update Q value for given state
                current_qs = current_qs_list[index]
                current_qs[action] = new_q

                # And append to our training data
                X.append(current_state)
                y.append(current_qs)

            # Fit on all samples as one batch, log only on terminal state
            self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

            # Update target network counter every episode
        self.target_model.set_weights(self.model.get_weights())

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        self.current_run.append(transition)
        if transition[2] > 0:
            self.checkpoint_replay.append(self.current_run)
            self.current_run = deque(maxlen=65)

        '''
        self.sort_replay_ctr += 1
        if self.sort_replay_ctr == self.sort_replay_at:
            self.replay_memory = sorted(self.replay_memory, key=get_score_mem)
            self.replay_memory = deque(self.replay_memory, maxlen=REPLAY_MEMORY_SIZE)
            self.sort_replay_ctr = 0'''

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done, score) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        state = state[np.newaxis, :]
        return self.model.predict(state)


agent = DQNAgent(len(Game_Sam.state), 128, 64, ACTION_SPACE, 100)


high = -10
scores = []
ep_rewards = []
epsilon += EPSILON_DECAY

for episode in range(1, EPISODES + 1):
    agent.tensorboard.step = episode
    if epsilon > MIN_EPSILON:
        epsilon -= EPSILON_DECAY
    else:
        epsilon = MIN_EPSILON
    Game_Sam.crashed = False
    score = 0
    Game_Sam.car.reset()
    current_state = Game_Sam.state
    agent.epsilon = epsilon
    step = 1
    while not Game_Sam.crashed:
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(np.array(current_state)))
        else:
            # Get random action
            action = np.random.randint(0, 5)

        Game_Sam.event = action

        Game_Sam.run()
        new_state = Game_Sam.state
        score += Game_Sam.reward
        agent.update_replay_memory((current_state, action, Game_Sam.reward, new_state, Game_Sam.crashed, score))
        agent.train(Game_Sam.crashed, step)
        if Game_Sam.reward > 0:
            # agent.train_chkpt()
            pass
        step += 1
        current_state = new_state

    scores.append(score)
    ep_rewards.append(score)
    if not episode % AGGREGATE_STATS_EVERY:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                       epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward > MIN_REWARD:
            MIN_REWARD = min_reward
            agent.model.save(
                f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    avg_score = np.mean(scores[max(0, episode - 100):(episode + 1)])
    print('episode:', episode, ' score:', score, ' average score:', avg_score, " Epsilon:", epsilon,
          " ending epsilon:", agent.epsilon)




