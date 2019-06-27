import numpy as np
from collections import defaultdict


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


    def __init__(self, env):
        self.Q = defaultdict(lambda: np.zeros(env.action_space))
        self.env = env
        self.alpha = 0.80
        self.gamma = 0.99
        self.epsilon = 0.10


    def learn(self):
        state = self.env.reset()
        epsilon_greedy_policy = make_epsilon_greedy_policy(self.env.action_space, self.Q[state], self.epsilon)
        action = choose_action_from_policy(self.env.action_space, epsilon_greedy_policy)

        while True:
            next_state, reward, done = self.env.step(action)
            epsilon_greedy_policy = make_epsilon_greedy_policy(self.env.action_space, self.Q[next_state], self.epsilon)
            next_action = choose_action_from_policy(self.env.action_space, epsilon_greedy_policy)
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


class QLearning:


    def __init__(self, env):
        self.Q = defaultdict(lambda: np.zeros(env.action_space))
        self.env = env
        self.alpha = 0.80
        self.gamma = 0.99
        self.epsilon = 0.10


    def learn(self):
        state = self.env.reset()

        while True:
            epsilon_greedy_policy = make_epsilon_greedy_policy(self.env.action_space, self.Q[state], self.epsilon)
            action = choose_action_from_policy(self.env.action_space, epsilon_greedy_policy)
            next_state, reward, done = self.env.step(action)
            max_Q_a = np.max([self.Q[next_state][next_action] for next_action in np.arange(self.env.action_space)])
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


class ExpectedSarsa:


    def __init__(self, env):
        self.Q = defaultdict(lambda: np.zeros(env.action_space))
        self.env = env
        self.alpha = 0.80
        self.gamma = 0.99
        self.epsilon = 0.10


    def learn(self):
        state = self.env.reset()

        while True:
            epsilon_greedy_policy = make_epsilon_greedy_policy(self.env.action_space, self.Q[state], self.epsilon)
            action = choose_action_from_policy(self.env.action_space, epsilon_greedy_policy)
            next_state, reward, done = self.env.step(action)
            next_policy = make_epsilon_greedy_policy(self.env.action_space, self.Q[next_state], self.epsilon)
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