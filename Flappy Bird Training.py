#!/usr/bin/env python
# coding: utf-8

# # Flappy Bird with TD Learning

# In[1]:


from environment import FlappyEnvironment
from agent import Sarsa, QLearning, ExpectedSarsa

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


env = FlappyEnvironment()


# In[4]:


n_episodes = 10000
n_episodes_range = [x+1 for x in range(n_episodes)]


# In[5]:


def plot_training_results(result_df, algorithm_title, figsize=(16,6)):
    plt.subplots(figsize=figsize)
    plt.plot(result_df.index, result_df['score'], label='Scores', color='b')
    plt.plot(result_df.index, result_df['score'].rolling(window=100).mean(), label='Avg Scores every 100 episodes', color='red')
    plt.title(algorithm_title)
    plt.ylabel('Score')
    plt.xlabel('# Episodes')
    plt.legend()
    plt.show()


# ## Sarsa

# In[6]:


sarsa = Sarsa(env.action_space)


# In[7]:


sarsa_scores = []

for i_episode in range(1, n_episodes + 1):
    env = FlappyEnvironment()
    sarsa.learn(env)
    sarsa_scores.append(env.game_score)
    
    if i_episode % 100 == 0:
        print("\rEpisode {}/{} - Max Score {}".format(i_episode, n_episodes, np.array(sarsa_scores).max()), end="")
        sys.stdout.flush()
        
sarsa.save_q_values()
sarsa_df = pd.DataFrame(data=sarsa_scores, index=n_episodes_range, columns=['score'])


# In[8]:


plot_training_results(sarsa_df, "Sarsa")


# ## Q-Learning

# In[9]:


qlearning = QLearning(env.action_space)


# In[10]:


q_learning_scores = []

for i_episode in range(1, n_episodes + 1):
    env = FlappyEnvironment()
    if i_episode % 100 == 0:
        print("\rEpisode {}/{} - Max Score {}".format(i_episode, n_episodes, np.array(q_learning_scores).max()), end="")
        sys.stdout.flush()
        
    qlearning.learn(env)
    q_learning_scores.append(env.game_score)

qlearning.save_q_values()
q_learning_df = pd.DataFrame(data=q_learning_scores, index=n_episodes_range, columns=['score'])


# In[11]:


plot_training_results(q_learning_df, "Q-Learning")


# ## Expected Sarsa

# In[12]:


expected_sarsa = ExpectedSarsa(env.action_space)


# In[13]:


expected_sarsa_scores = []

for i_episode in range(1, n_episodes + 1):
    env = FlappyEnvironment()
    if i_episode % 100 == 0:
        print("\rEpisode {}/{} - Max Score {}".format(i_episode, n_episodes, np.array(expected_sarsa_scores).max()), end="")
        sys.stdout.flush()
        
    expected_sarsa.learn(env)
    expected_sarsa_scores.append(env.game_score)
    
expected_sarsa.save_q_values()
expecte_sarsa_df = pd.DataFrame(data=expected_sarsa_scores, index=n_episodes_range, columns=['score'])


# In[14]:


plot_training_results(expecte_sarsa_df, "Expected Sarsa")


# # Model Comparison

# In[15]:


import matplotlib.pyplot as plt

zoom_at = 1000
plt.subplots(figsize=(16, 8))
plt.plot(range(0, zoom_at), sarsa_scores[:zoom_at], label='Sarsa')
plt.plot(range(0, zoom_at), q_learning_scores[:zoom_at], label='Q-Learning (Sarsamax)')
plt.plot(range(0, zoom_at), expected_sarsa_scores[:zoom_at], label='Expected Sarsa')
plt.xlabel('Episodes')
plt.ylabel("Sum of rewards during episodes")
plt.legend()
plt.show()

