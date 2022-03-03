import pygame as py
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Input
from keras.optimizer_v1 import Adam
from collections import deque
import random
import numpy as np
import time

tf.compat.v1.disable_eager_execution()  # allows the older version of the Adam optimizer to work


BOX_SIZE = 10
NUM_SQUARES = 10
WIDTH = NUM_SQUARES * BOX_SIZE + BOX_SIZE
HEIGHT = NUM_SQUARES * BOX_SIZE + BOX_SIZE
RENDER = True

MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
DISCOUNT = .99
MAX_REPLAY_BUFFER_LEN = 100_000
UPDATE_TARGET_EVERY = 5
NUM_GAMES = 10
epsilon = 0
EPS_DECAY = epsilon / 5000
AGGREGATE_STATS_EVERY = 100
MIN_REWARD = -30

if RENDER:
    py.init()
    screen = py.display.set_mode((WIDTH, HEIGHT))


# a class for both players and targets
class Box:
    def __init__(self, x, y, max_x, max_y):
        self.x, self.y = x, y
        self.max_x, self.max_y = max_x, max_y

    def move_up(self):
        if self.y > 0:
            self.y -= 1

    def move_down(self):
        if self.y < self.max_y:
            self.y += 1

    def move_left(self):
        if self.x > 0:
            self.x -= 1

    def move_right(self):
        if self.x < self.max_x:
            self.x += 1

    def do_nothing(self):
        pass


def build_model():
    model = Sequential([
        Input(4),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(5)
    ])

    model.compile(optimizer=Adam(lr=.001), loss='mse', metrics=['accuracy'])
    return model


class Agent:
    def __init__(self, max_mem):
        # self.model = build_model()
        self.model = load_model('models/John_Fully_Trained')
        self.target_model = build_model()
        self.target_model.set_weights(self.model.get_weights())

        self.memory_replay = deque(maxlen=max_mem)

        self.target_update_counter = 0

    def remember(self, transition):
        self.memory_replay.append(transition)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        return np.argmax( self.model.predict(state))

    def train(self, terminal_state):
        if len(self.memory_replay) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.memory_replay, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

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
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


def draw(player, target):
    screen.fill((0, 0, 0))
    py.draw.rect(screen, (0, 0, 255), (player.x * 10, player.y * 10,10,10), width=10)
    py.draw.rect(screen, (255, 0, 0), (target.x * 10, target.y * 10,10,10), width=10)

    py.display.update()



agent = Agent(MAX_REPLAY_BUFFER_LEN)
scores = []
for episode in range(1, NUM_GAMES + 1):
    playerx = random.randint(0, NUM_SQUARES)
    playery = random.randint(0, NUM_SQUARES)
    player = Box(playerx, playery, NUM_SQUARES, NUM_SQUARES)

    targetx, targety = random.randint(0, NUM_SQUARES), random.randint(0, NUM_SQUARES)
    while targetx == playerx and targety == playery:
        targetx, targety = random.randint(0, NUM_SQUARES), random.randint(0, NUM_SQUARES)
    target = Box(targetx, targety, NUM_SQUARES, NUM_SQUARES)


    done = False
    state = np.array([player.x, player.y, target.x, target.y])
    score = 0


    while not done:
        if RENDER:
            draw(player,target)
            time.sleep(.25)


        reward = -1
        if random.random() > epsilon:
            action = agent.choose_action(state)
        else:
            action = random.randint(0,4)

        if action == 0:
            player.do_nothing()
        elif action == 1:
            player.move_up()
        elif action == 2:
            player.move_down()
        elif action == 3:
            player.move_left()
        elif action == 4:
            player.move_right()
        else:
            print("Something went wrong. Invalid action taken")

        if player.x == target.x and player.y == target.y:
            reward = 10
            done = True

        score += reward

        new_state = np.array([player.x, player.y, target.x, target.y])
        transition = (state, action, reward, new_state, done)
        agent.remember(transition)

        state = new_state

        agent.train(done)
    scores.append(score)
    if not episode % AGGREGATE_STATS_EVERY:
        average_reward = sum(scores[-AGGREGATE_STATS_EVERY:]) / len(scores[-AGGREGATE_STATS_EVERY:])
        min_reward = min(scores[-AGGREGATE_STATS_EVERY:])
        max_reward = max(scores[-AGGREGATE_STATS_EVERY:])

        # Save model, but only when min reward is greater or equal a set value
        if min_reward > MIN_REWARD:
            MIN_REWARD = min_reward
            agent.model.save(
                f'models/John__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    avg_score = np.mean(scores[max(0, episode - 100):(episode + 1)])
    print('episode:', episode, ' score:', score, ' average score:', avg_score, " Epsilon:", epsilon)

    epsilon -= EPS_DECAY

average_reward = sum(scores[-AGGREGATE_STATS_EVERY:]) / len(scores[-AGGREGATE_STATS_EVERY:])
min_reward = min(scores[-AGGREGATE_STATS_EVERY:])
max_reward = max(scores[-AGGREGATE_STATS_EVERY:])

# Save model, but only when min reward is greater or equal a set value
agent.model.save(
        f'models/John__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__END.model')
