[img_comparision_reward_13]: https://github.com/Brandon-HY-Lin/deep-reinforcement-learning/blob/master/p1_navigation/figues/comparison_reward_13.png "Comparison between DQN, DDQN, Deuling DDQN"



# Project 1: Navigation

In this project, I implement different methods to train a simple game. These methods are [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), [DDQN](https://arxiv.org/abs/1509.06461), and [Dueling DDQN](https://arxiv.org/abs/1511.06581). The DDQN method beats other methods and uses only 261 episodes to achieve the average reward of +13 points.


## 1. Introduction
This goal of this project is to train a framework that can play a game automatically. The game is controlled by [Unity Machine Learning Agents (ML-Agents)](https://github.com/Unity-Technologies/ml-agents). Furthermore, the goal of this game is to collect edible bananna (yellow color) while avoiding poisonous bananna (blue color). Collecting a yellow bananna results +1 reward, whereas collecting a blue bananna results -1 reward. I apply 3 existing methods to train this game. There are [DQN (Deep-Q Networks)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), [DDQN (Double Deep-Q Netowrks)](https://arxiv.org/abs/1509.06461), and [Dueling DDQN](https://arxiv.org/abs/1511.06581).


## 2. Implementations
The original pappers of DQN, DDQN, and Dueling DDQN are pixel-based methods. In this project, ML-agent provides an API that facillitate a directed observations of states. There are 37 states in this game. To speedup the training, I use this API instead of pixel-based approach. 
#### 2.1 DQN (Deep Q-Networks)
In this project, DQN has 3 fully-connected layers with output shape 64, 64, and 4. Note that the size of controll space is 4 and it correspoding to the size of the last layer. It uses 396 episodes to achieve a reward of +13.


#### 2.2 DDQN (Double Deep Q-Networks)
The architecture of DDQN is the same as DQN except I tweak the hyperparameters of $\epsilon$-greedy algorithm. The decay rate of $\epsilon$ is lowered from 1.0 to 0.98 on each step. Furthermore, the minimum of $\epsilon$ is raised from 0.01 to 0.02. It uses 261 episodes to achieve +13 reward.

#### 2.3 Dueling DDQN (Dueling Double Deep Q-Networks)
This is a variant of DDQN. The framework is similar to DDQN except that the neural networks. Dueling DDQN alters the fully-connected neural networks to addition of 2 small fully-connected networks. The implementation is based on [Wang's paper](https://arxiv.org/abs/1511.06581). It uses 342 episodes to achieve +13 reward.


# 3. Results
The results of how fast each framework can achieve +13 reward is shown in the following table. The DDQN framework has fewest episodes which is 261. This is pretty surprising that dueling DDQN, the more advanced variant framework, is not the best one. I tried to tweak more parameters in dueling DDQN but it doesn't work well.

![comparison reward 13][img_comparision_reward_13]

**Framework**|**Arch of Neural Networks**|**Hyper-parameters: decay rate of epsilon**|**Hyper-parameters: min of epsilon**|**#Episode to achieve +13 reward**
:-----:|:-----:|:-----:|:-----:|:-----:
DQN|3 Dense Layers|1.0|0.01|396
DDQN|3 Dense Layers|0.98|0.02|261
Dueling DDQN|2 Dense Layers + 1 combined Layer|0.98|0.1|342


The detail experiments results are shown below:

**Framework**|**Arch of Neural Networks**|**Hyper-parameters: decay rate of epsilon**|**Hyper-parameters: min of epsilon**|**#Episode to achieve +13 reward**
:-----:|:-----:|:-----:|:-----:|:-----:
DDQN|3 Dense Layers|0.98|0.02|261
DDQN|3 Dense Layers|0.98|0.1|261
DDQN|3 Dense Layers|0.98|0.01|281
Dueling DDQN|2 Dense Layers + 1 combined Layer|0.98|0.1|342
Dueling DDQN|2 Dense Layers + 1 combined Layer|0.995|0.01|380
DQN|3 Dense Layers|0.995|0.01|396
DDQN|3 Dense Layers|0.995|0.01|398
DDQN|3 Dense Layers|0.98|0.01|479
Dueling DDQN|2 Dense Layers + 1 combined Layer|0.98|0.1|516

