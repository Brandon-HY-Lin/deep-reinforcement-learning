[maddpg_algorithm]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p3_collab-compet/MADDPG/picures/maddpg_algorithm.png "Algorithm of MADDPG"

[maddpg_digram]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p3_collab-compet/MADDPG/picures/maddpg_diagram.png "Diagram of MADDPG"


# Abstract

# Introduction
By looking at the highlighted text in MADDPG algorithm shown below, the major difference between MADDPG and DDPG is the input shape of critic. Note that, x = \[o1, o2, ..., oN\]. The input size of critic in MADDPG algorithm is __(state_size+action_size)*num_agents__. On the other hand, the input size of actor in MADDPG algorithm is the same as DDPG, i.e. __state_size__.

![Algorithm of MADDPG][maddpg_algorithm]

*Algorthm of MADDPG*


Furthermore, the diagram shown below also illustrate this subtile difference. For N agents, there are N policy-networks, N Q-networks, and only 1 replay buffer.

![Diagram of MADDPG][maddpg_digram]

*Diagram of MADDPG*
