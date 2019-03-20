[img_ddpg_version_1]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p2_continuous-control/DDPG/pictures/ddpg_version_1.png "Score of Version 1"


# Abstract
This work adopts [DDPG](https://arxiv.org/abs/1509.02971) with [prioritized experience replay (PER)](https://arxiv.org/abs/1511.05952) to play [Unity's Reacher game](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) and achieves good results. The stability issue of reinforcement learning is greatly reduced by PER. This work achieve average score of 30.0 at episode 24. The final average scores at episode 400 is 38.8. 


# Introduction
In this project, the Unity's Reacher has been used as a testbed for DDPG algorithm. There are 20 agents in this environmet and +0.1 reward is given if agent's hand is in goal location. The maximum time step is 1,000. Hence the maximum total rewards should be +100. The sizes of observation space and action space are 33 and 4, respectively. Both spaces are continous. There are several techniques are applied including gradient clipping, batch normalization, and PER.


# Approches
Six versions are implemented in this work. The techniques used in early version are adopted in the later version which results incremented performances.


### Version 1:
Parameters:
* learning rate for critic and actor are both 1e-3
* actor has 3 fully-connected neural networks with 1st layer size=400, 2nd layer size=300
* critic has 3-layer neural networks. 1st layer size=400 and 2nd layer size=300. The input 2nd layer is concatenated with action space.
* buffer size = 1e6

![Score of Version 1][img_ddpg_version_1]