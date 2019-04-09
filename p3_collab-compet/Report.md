[maddpg_algorithm]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p3_collab-compet/MADDPG/picures/maddpg_algorithm.png "Algorithm of MADDPG"

[maddpg_digram]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p3_collab-compet/MADDPG/picures/maddpg_diagram.png "Diagram of MADDPG"


# Abstract
This work adopts [MADDPG](https://arxiv.org/abs/1706.02275) to play tennis game which is similar to [Unity's Tennis game](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) and achieve a score of 2.7. Moreover, [DDPG](https://arxiv.org/abs/1509.02971) with [prioritized experience replay (PER)](https://arxiv.org/abs/1511.05952) is also implemented and achieve similar score.

# Introduction
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
By looking at the highlighted text in [MADDPG](https://arxiv.org/abs/1706.02275) algorithm shown below, the major difference between MADDPG and DDPG is the input shape of critic. Note that, x = \[o1, o2, ..., oN\]. The input size of critic in MADDPG algorithm is __(state_size+action_size)*num_agents__. On the other hand, the input size of actor in MADDPG algorithm is the same as DDPG, i.e. __state_size__.

![Algorithm of MADDPG][maddpg_algorithm]

*Algorthm of MADDPG*


Furthermore, the diagram shown below also illustrate this subtile difference. For N agents, there are N policy-networks, N Q-networks, and only 1 replay buffer. Note that, each agent's critic needs to get the other agent's next actions. To solve this problem, the [chain-of-responsibility](https://en.wikipedia.org/wiki/Chain-of-responsibility_pattern) design pattern, i.e. broker chain, is used.

![Diagram of MADDPG][maddpg_digram]

*Diagram of MADDPG*



# Results


# Conclusion
In this work, 2 algorithm are implemented and both achieve good scores of over +2.5. During experiments, one key factor is noticed: the performances of both algorithms are highly influenced by the number of training steps. The more it is, the faster the score gets higher. Besides this, stability is still a big issue even if PER is applied as shown in DDPG version.


# Future Works
1. Add PER (Prioritized Experience Replay) to MADDPG. Although shared replay buffer is used by two agents, it worths a try to add replay buffer to one of the agent.
2. Adopt [twin delayed DDPG](https://spinningup.openai.com/en/latest/algorithms/td3.html). This variant has 3 tricks, some of them are applied in current project. So, it woundn't too much efforts to do.


# Appendix

#### Report of DDPG
Check this [link](https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p3_collab-compet/DDPG/Report.md) to see detailed report of DDPG.


#### Hyper-Parameters

* MADDPG

* DDPG