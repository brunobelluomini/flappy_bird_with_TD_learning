import numpy as np
from collections import defaultdict
import json
import os


def make_epsilon_greedy_policy(action_space_size, Q_state, epsilon):
    policy_state = np.ones(action_space_size) * epsilon / action_space_size
    policy_state[np.argmax(Q_state)] = 1 - epsilon + (epsilon / action_space_size)

    return policy_state


def choose_action_from_policy(action_space, policy):
    available_actions = np.arange(action_space)

    return np.random.choice(available_actions, p=policy)


def update_Q_values(alpha, reward, gamma, Q_current_action, Q_next_action):
    return Q_current_action + alpha * (reward + gamma * Q_next_action - Q_current_action)


class Sarsa:


    def __init__(self, action_space, force_training=False):
        self.action_space = action_space
        self.Q = self.load_q_values(force_training)
        self.alpha = 0.70
        self.gamma = 1.00
        self.epsilon = 0.15


    def learn(self, env):
        state = env.get_state()
        epsilon_greedy_policy = make_epsilon_greedy_policy(self.action_space, self.Q[state], self.epsilon)
        action = choose_action_from_policy(self.action_space, epsilon_greedy_policy)

        while True:
            next_state, reward, done = env.step(action)
            epsilon_greedy_policy = make_epsilon_greedy_policy(self.action_space, self.Q[next_state], self.epsilon)
            next_action = choose_action_from_policy(self.action_space, epsilon_greedy_policy)
            Q_current_action = self.Q[state][action]
            Q_next_action = self.Q[next_state][next_action]
            self.Q[next_state][next_action] = update_Q_values(
                self.alpha,
                reward,
                self.gamma,
                Q_current_action,
                Q_next_action
            )
            state = next_state
            action = next_action

            if done:
                break


    def load_q_values(self, force_training):
        if force_training or not os.path.exists('sarsa_q_values.json'):
            return defaultdict(lambda: [0 for action in range(self.action_space)])
        else:
            q_values_file = open('sarsa_q_values.json', 'r')
            q_values = json.load(q_values_file)
            q_values_file.close()

            return defaultdict(lambda: [0 for action in range(self.action_space)], q_values)


    def save_q_values(self):
        q_values_file = open('sarsa_q_values.json', 'w')
        json.dump(dict(self.Q), q_values_file)
        q_values_file.close()
        

class QLearning:


    def __init__(self, action_space, force_training=False):
        self.action_space = action_space
        self.Q = self.load_q_values(force_training)
        self.alpha = 0.70
        self.gamma = 1.00
        self.epsilon = 0.15


    def learn(self, env):
        state = env.get_state()

        while True:
            epsilon_greedy_policy = make_epsilon_greedy_policy(self.action_space, self.Q[state], self.epsilon)
            action = choose_action_from_policy(self.action_space, epsilon_greedy_policy)
            next_state, reward, done = env.step(action)
            max_Q_a = np.max([self.Q[next_state][next_action] for next_action in np.arange(self.action_space)])
            self.Q[state][action] = update_Q_values(
                self.alpha,
                reward,
                self.gamma,
                self.Q[state][action],
                max_Q_a
            )
            state = next_state

            if done:
                break


    def load_q_values(self, force_training):
        if force_training or not os.path.exists('q_learning_q_values.json'):
            return defaultdict(lambda: [0 for action in range(self.action_space)])
        else:
            q_values_file = open('q_learning_q_values.json', 'r')
            q_values = json.load(q_values_file)
            q_values_file.close()

            return defaultdict(lambda: [0 for action in range(self.action_space)], q_values)


    def save_q_values(self):
        q_values_file = open('q_learning_q_values.json', 'w')
        json.dump(dict(self.Q), q_values_file)
        q_values_file.close()


class ExpectedSarsa:


    def __init__(self, action_space, force_training=False):
        self.action_space = action_space
        self.Q = self.load_q_values(force_training)
        self.alpha = 0.70
        self.gamma = 1.00
        self.epsilon = 0.15


    def learn(self, env):
        state = env.get_state()

        while True:
            epsilon_greedy_policy = make_epsilon_greedy_policy(self.action_space, self.Q[state], self.epsilon)
            action = choose_action_from_policy(self.action_space, epsilon_greedy_policy)
            next_state, reward, done = env.step(action)
            next_policy = make_epsilon_greedy_policy(self.action_space, self.Q[next_state], self.epsilon)
            expected_Q = np.dot(self.Q[next_state], next_policy)
            self.Q[state][action] = update_Q_values(
                self.alpha,
                reward,
                self.gamma,
                self.Q[state][action],
                expected_Q
            )
            state = next_state

            if done:
                break


    def act(self, state):
        epsilon_greedy_policy = make_epsilon_greedy_policy(self.action_space, self.Q[state], self.epsilon)
        action = choose_action_from_policy(self.action_space, epsilon_greedy_policy)

        return action


    def load_q_values(self, force_training):
        if force_training or not os.path.exists('expected_sarsa_q_values.json'):
            return defaultdict(lambda: [0 for action in range(self.action_space)])
        else:
            q_values_file = open('expected_sarsa_q_values.json', 'r')
            q_values = json.load(q_values_file)
            q_values_file.close()

            return defaultdict(lambda: [0 for action in range(self.action_space)], q_values)


    def save_q_values(self):
        q_values_file = open('expected_sarsa_q_values.json', 'w')
        json.dump(dict(self.Q), q_values_file)
        q_values_file.close()