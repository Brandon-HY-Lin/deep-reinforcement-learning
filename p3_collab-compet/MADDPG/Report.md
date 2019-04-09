[maddpg_algorithm]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p3_collab-compet/MADDPG/picures/maddpg_algorithm.png "Algorithm of MADDPG"

[maddpg_digram]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p3_collab-compet/MADDPG/picures/maddpg_diagram.png "Diagram of MADDPG"


# Abstract
This work adopts [MADDPG](https://arxiv.org/abs/1706.02275) to play tennis game which is similar to [Unity's Tennis game](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) and achieve a score of 2.6.

# Introduction
By looking at the highlighted text in [MADDPG](https://arxiv.org/abs/1706.02275) algorithm shown below, the major difference between MADDPG and DDPG is the input shape of critic. Note that, x = \[o1, o2, ..., oN\]. The input size of critic in MADDPG algorithm is __(state_size+action_size)*num_agents__. On the other hand, the input size of actor in MADDPG algorithm is the same as DDPG, i.e. __state_size__.

![Algorithm of MADDPG][maddpg_algorithm]

*Algorthm of MADDPG*


Furthermore, the diagram shown below also illustrate this subtile difference. For N agents, there are N policy-networks, N Q-networks, and only 1 replay buffer.

![Diagram of MADDPG][maddpg_digram]

*Diagram of MADDPG*

# Results


# Appendix

### Hyper-Parameters
param_agent = {		'state_size': 24, 
                    'action_size': 2,
                    'random_seed': random_seed,
                    'lr_critic': 1e-3,
                    'lr_actor': 1e-3,
                    'fc1_units': 256,
                    'fc2_units': 128,
                    'gamma': 0.95,
                    'tau': 1e-2,
                    'max_norm': 1.0,
                    'epsilon_start': 5.0,
                    'epsilon_end': 0.0,
                    'epsilon_decay': 0.99,}

                    'learn_period': 10,
                        'learn_sampling_num':10,
                         'buffer_size': int(1e6), 
                         'batch_size': 256,
                          'random_seed': random_seed}


### Design Patterns
Because each agent needs the other agent to predict its next actions, the [chain-of-responsibility](https://en.wikipedia.org/wiki/Chain-of-responsibility_pattern) design pattern, i.e. broker chain, is used in this project. The Game() class stores 2 agents and each agent also contains Game() class.
