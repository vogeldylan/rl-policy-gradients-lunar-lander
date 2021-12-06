# Reinforcement Learning with Actor-Critic Policy Gradients

Reinforcement learning using actor-critic policy gradients to train a lunar lander. Completed as part of the ETH Zurich Probabilistic AI class. Credit to Prof. Andreas Krause and the PAI teaching team for much of the skeleton code.

# Project Structure

All of the important code is contained in `solution.py`. Contains classes for the policy function neural network (actor) and value function neural network (critic). Also contains classes for the experience buffer, and the agent class which trains the two neural networks and selects the next action.

`lunar_lander.py` contains the lunar lander environment. 

# Running

Simply call `python solution.py` to have the model run. A short video will be created from 10 evaluation runs of the learned policy, along with reporting of the average return per run. 

