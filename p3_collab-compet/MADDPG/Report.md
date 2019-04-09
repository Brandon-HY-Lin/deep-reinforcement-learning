[maddpg_algorithm]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p3_collab-compet/MADDPG/picures/maddpg_algorithm.png "Algorithm of MADDPG"

[maddpg_digram]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p3_collab-compet/MADDPG/picures/maddpg_diagram.png "Diagram of MADDPG"

[img_maddpg_version_11]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p3_collab-compet/MADDPG/picures/MADDPG_version_11.png "Score of Version 11"


# Abstract
This work adopts [MADDPG](https://arxiv.org/abs/1706.02275) to play tennis game which is similar to [Unity's Tennis game](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) and achieve a score of 2.7. The average score reaches +0.5 at episode 466.

# Introduction
By looking at the highlighted text in [MADDPG](https://arxiv.org/abs/1706.02275) algorithm shown below, the major difference between MADDPG and DDPG is the input shape of critic. Note that, x = \[o1, o2, ..., oN\]. The input size of critic in MADDPG algorithm is __(state_size+action_size)*num_agents__. On the other hand, the input size of actor in MADDPG algorithm is the same as DDPG, i.e. __state_size__.

![Algorithm of MADDPG][maddpg_algorithm]

*Algorthm of MADDPG*


Furthermore, the diagram shown below also illustrate this subtile difference. For N agents, there are N policy-networks, N Q-networks, and only 1 replay buffer.

![Diagram of MADDPG][maddpg_digram]

*Diagram of MADDPG*

# Results
The average score reaches +0.5 at episode 466. The highest score is 2.7 at episode 937. The stability of score is still an issue despite adding batch normalization.

![Score of MADDPG version 11][img_maddpg_version_11]


# Appendix

### Hyper-Parameters
* state_size : 24
* action_size : 2
* lr_critic : 1e-3 (Adam optimizer)
* lr_actor : 1e-3  (Adam optimizer)
* fc1_units : 256
* fc2_units : 128
* gamma : 0.95     (discount rate of reward)
* tau : 1e-2       (parameter of soft update)
* max_norm : 1.0   (gradient clipping)
* epsilon_start : 5.0     (starting ratio of exploration)
* epsilon_end : 0.0       (ending ratio of exploration)
* epsilon_decay : 0.99    (decay rate of exploration)
* learn_period : 10       (training period)
* learn_sampling_num :10  (number of training in each period)
* buffer_size : int(1e6)
* batch_size : 256


### Design Patterns
Because each agent needs the other agent to predict its next actions, the [chain-of-responsibility](https://en.wikipedia.org/wiki/Chain-of-responsibility_pattern) design pattern, i.e. broker chain, is used in this project. The Game() class stores 2 agents and each agent also contains Game() class.
