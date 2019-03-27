# Abstract

# Introduction

### Environment
* By looking at the observation spaces of each agent shown below, it's easy to see that values of field 20 of agent_0 and agent_1 are complement.

---
```
Observation of agent 0:
[ 0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.         -7.43639946 -1.5        -0.          0.
  6.83172083  5.97645617 -0.          0.        ]

Observation of agent 1:
[ 0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.         -7.73019552 -1.5         0.          0.
 -6.83172083  5.97645617  0.          0.        ]

Complement fields:
6.83172082901001
-6.83172082901001
```
---

Before implementing MADDPG, I guess DDPG is better than MADDPG.

* Conjecture: let 1 one agent be a quick learner and the other be a slow learner. It might increase scores.