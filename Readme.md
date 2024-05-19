# Overview
A proposal of MCTS and self-play for Gomoku.

# About
The goal of this project is to implement and train from scratch a model that can play Gomoku well to some extend. The game board is restricted to 7 by 7 in the implementation. For the larger game board, In the field of Go, artificial intelligence has unequivocally outperform human capabilities. The methodology outlined in the paper ’Mastering the Game of Go with Deep Neural Networks and Tree Search’ is the following: First it proposed the use of supervised learning to train the policy network with the, followed by a combination of reinforcement learning to further train both the policy and value networks. Once the policy and value networks achieve a certain level of accuracy, Monte Carlo Tree Search (MCTS) is employed for optimal decision-making.

# Instructions
Run main.py to play the game. run train.py to train the policy and valuenetwork.  Weights of two networks are also uploaded. 
