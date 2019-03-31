[maddpg_algorithm]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p3_collab-compet/MADDPG/picures/maddpg_algorithm.png "Algorithm of MADDPG"

[maddpg_digram]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p3_collab-compet/MADDPG/picures/maddpg_diagram.png "Diagram of MADDPG"


# Abstract

# Introduction
By looking at the algorithm of MADDPG shown below, the major difference between MADDPG and DDPG is the input shape of critic. Note that, $x = [o_1, o_2, ..., o_N]$. The input size of critic in MADDPG algorithm is **(state_size+action_size)*num_agents**. On the other hand, the input size of actor in MADDPG algorithm is the same as DDPG, i.e. **state_size**.

![Algorithm of MADDPG][maddpg_algorithm]
*Algorthm of MADDPG*


