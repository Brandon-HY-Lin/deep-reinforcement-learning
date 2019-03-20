[img_ddpg_version_1]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p2_continuous-control/DDPG/pictures/ddpg_version_1.png "Score of Version 1"

[img_ddpg_version_2]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p2_continuous-control/DDPG/pictures/ddpg_version_2.png "Score of Version 2"

[img_ddpg_version_3]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p2_continuous-control/DDPG/pictures/ddpg_version_3.png "Score of Version 3"

[img_ddpg_version_4]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p2_continuous-control/DDPG/pictures/ddpg_version_4.png "Score of Version 4"

[img_ddpg_version_5]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p2_continuous-control/DDPG/pictures/ddpg_version_5.png "Score of Version 5"

[img_ddpg_version_6]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p2_continuous-control/DDPG/pictures/ddpg_version_2.png "Score of Version 6"

[ounoise_sigma_02_theta_015]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p2_continuous-control/DDPG/pictures/ounoise_mu_0_sigma_02_theta_015.png "OUnoise"


# Abstract
This work adopts [DDPG](https://arxiv.org/abs/1509.02971) with [prioritized experience replay (PER)](https://arxiv.org/abs/1511.05952) to play [Unity's Reacher game](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) and achieves good results. The stability issue of reinforcement learning is greatly reduced by PER. This work achieve average score of 30.0 at episode 24. The final average scores at episode 400 is 38.8. 


# Introduction
In this project, the Unity's Reacher has been used as a testbed for DDPG algorithm. There are 20 agents in this environmet and +0.1 reward is given if agent's hand is in goal location. The maximum time step is 1,000. Hence the maximum total rewards should be +100. The sizes of observation space and action space are 33 and 4, respectively. Both spaces are continous. There are several techniques are applied including gradient clipping, batch normalization, and PER.


# Approches
Six versions are implemented in this work. The techniques used in early version are adopted in the later version which results incremented performances.


### Version 1:
Vanilla DDPG which updates every time step. Score enters plateau with value of 5. Although the score is very low, it suffers from instability issue.

Parameters:
* learning rate for critic and actor are both 1e-3
* actor has 3 fully-connected neural networks with 1st layer size=400, 2nd layer size=300
* critic has 3-layer neural networks. 1st layer size=400 and 2nd layer size=300. The input 2nd layer is concatenated with action space.
* buffer size = 1e6
* exploration mu, sigma, and theta of OUNoise process is 0, 0.2, and 0.15

![Score of Version 1][img_ddpg_version_1]
The translucent line shows the raw score of every episode. The solid line shows the average score with window=100.


### Version 2: 
To solve the instability issue, gradient-clipping is applied as [Duan et al., 2016](https://arxiv.org/abs/1604.06778) suggests. The result shows that not only instability issue is gone, the score soars to 10. However, the score is still way below the criterion of +30.

Parameters:
* max_norm = 1.0     (by calling torch.clip_grad_norm_(, 1.0))

![Score of Version 2][img_ddpg_version_2]


### Version 3:
In the previous versions, agent learns every time step. This might be the cause of low score. In version 3, agent only learns every 20 time step. In each learning process, 10 random samples are performed. The result soon overshoots +30 criterion despite the raw data jitters.

![Score of Version 3][img_ddpg_version_3]

The translucent line shows the raw score of every episode. The solid line shows the average score with window=100. The average score exceed +30 at episode 212 and it reaches 35.76 at episode 400.


### Version 4:
By inspecting OUNoise, the random noise process keeps jitter upon time step and never decay. Whereas, [DQN](https://www.nature.com/articles/nature14236) exploits the decay property. As a result, version 4 decrease the value of noise every time step.

![OUNoise sigma=0.2 theta=0.15][ounoise_sigma_02_theta_015]

Parameters:
* epsilon_start = 1.0
* epsilon_decay = 1e-6 
* Note: (epsilon_start = epsilon_start - epsilon_decay) on every time step.

![Score of Version 4][img_ddpg_version_4]
The average score exceed +30 at episode 135 and it reaches 37.07 at episode 400 which is greater than version 3 by 1.3 point. Note that it speeds up the convergence. 
