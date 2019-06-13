[maddpg_algorithm]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p3_collab-compet/MADDPG/picures/maddpg_algorithm.png "Algorithm of MADDPG"

[maddpg_digram]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p3_collab-compet/MADDPG/picures/maddpg_diagram.png "Diagram of MADDPG"

[maddpg_ddpg_comparision]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p3_collab-compet/MADDPG/picures/maddpg_v11_ddpg_v6_v7.png "Score of MADDPG version 11, DDPG version 6, and DDPG version 7"

[tennis_simulation_udacity]: https://raw.githubusercontent.com/Brandon-HY-Lin/deep-reinforcement-learning/master/p3_collab-compet/MADDPG/picures/tennis_simulation.gif "Tennis simulation"

# Project: Tennis Game

# Abstract
This work adopts [MADDPG](https://arxiv.org/abs/1706.02275) to play tennis game which is similar to [Unity's Tennis game](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) and achieve a maximal score of 2.7. Its highest average score is +1.2 over 100 episodes. Aside from MADDPG, [DDPG](https://arxiv.org/abs/1509.02971) with [prioritized experience replay (PER)](https://arxiv.org/abs/1511.05952) is also implemented and achieve similar score. The highest score is +2.6 and the best average score is +0.8 (over 100 episodes).

# Introduction

![Tennis game simulation][tennis_simulation_udacity]

Two-player game where agents control rackets to bounce ball over a net. The agents must bounce ball between one another while not dropping or sending ball out of bounds. The environment contains two agent linked to a single Brain. 

* Agent Reward Function (independent):
	* +0.1 To agent when hitting ball over net.
	* -0.1 To agent who let ball hit their ground, or hit ball out of bounds.

* Brains: One Brain with the following observation/action space.
	* Vector Observation space: 8 variables corresponding to position and velocity of ball and racket.
	* Vector Action space: (Continuous) Size of 2, corresponding to movement toward net or away from net, and jumping.
	* Visual Observations: None.

In order to solve this game, the agents must get average score of +0.5 over 100 consecutive episodes. The score of each episode is calculated by taking maximal scores of 2 agents.


# MADDPG Algorithm
By looking at the highlighted text in [MADDPG](https://arxiv.org/abs/1706.02275) algorithm shown below, the major difference between MADDPG and DDPG is the input shape of critic. Note that, x = \[o1, o2, ..., oN\], where o1 is the observation space of agent 1. The input size of critic in MADDPG algorithm is __(state_size+action_size)*num_agents__. On the other hand, the input size of actor in MADDPG algorithm is the same as DDPG, i.e. __state_size__.

![Algorithm of MADDPG][maddpg_algorithm]

*Algorthm of MADDPG*


Furthermore, the diagram shown below also illustrate this subtile difference. For N agents, there are N policy-networks, N Q-networks, and only 1 replay buffer. Note that, each agent's critic needs to get the other agent's next actions. To solve this problem, the [chain-of-responsibility](https://en.wikipedia.org/wiki/Chain-of-responsibility_pattern) design pattern, i.e. broker chain, is used.

![Diagram of MADDPG][maddpg_digram]

*Diagram of MADDPG*



# Results
Three experiments are listed below. Two of them are DDPG, the other is MADDPG. DDPG version 6 and MADDPG version 11 have batch-norm at input of critic and actor netorks. The MADDPG has better result and reaches average score of +0.5 at episode 466. And its highest average score is about +1.2.


![Diagram of MADDPG and DDPG comparison][maddpg_ddpg_comparision]

*Scores of MADDPG and DDPG*

# Conclusion
In this work, two algorithms are implemented and both achieve good scores of over +2.5. During experiments, one key factor is noticed: the performances of both algorithms are highly influenced by the number of training steps. The more it is, the faster the score gets higher. Besides this, stability is still a big issue even if PER is applied as shown in DDPG version.


# Future Works
1. Add PER (Prioritized Experience Replay) to MADDPG. Although shared replay buffer is used by two agents, it worths a try to add replay buffer to one of the agent.
2. Adopt [twin delayed DDPG](https://spinningup.openai.com/en/latest/algorithms/td3.html). This variant has 3 tricks, some of them are applied in current project. So, it woundn't too much efforts to do.


# Appendix

#### Report of DDPG
Check this [link](https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p3_collab-compet/DDPG/Report.md) to see detailed report of DDPG.


#### Hyper-Parameters

* MADDPG
    * state_size : 24
    * action_size : 2
    * lr_critic : 1e-3 (learning rate of critic using Adam optimizer)
    * lr_actor : 1e-3  (learning rate of actor using Adam optimizer)
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


* DDPG
    * state_size : 24
    * action_size : 2
    * lr_actor : 1e-3          (learning rate of actor using Adam optimizer)
    * lr_critic : 1e-3         (learning rate of critic using Adam optimizer)
    * fc1_units : 256
    * fc2_units : 128
    * buffer_size : int(1e6)
    * learn_period : 10         (training period)
    * learn_sampling_num : 20   (number of training in each period)
    * batch_size : 128
    * max_norm : 1.0            (gradient clipping)
    * exploration_sigma : 0.2   (parameter of OU noise process)
    * epsilon_start : 5.0       (starting ratio of exploration)
    * epsilon_end : 0.0         (ending ratio of exploration)
    * epsilon_decay : 0.0       (decay rate of exploration)
