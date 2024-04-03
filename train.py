import copy
from gomoku import *
from policynetwork import *
from valuenetwork import *

class OpponentPool:
    def __init__(self):
        self.opponents = []

    def add_opponent(self, opponent_network):
        self.opponents.append(copy.deepcopy(opponent_network))

    def sample_opponent(self):
        import random
        if len(self.opponents) == 0:
            raise ValueError("Opponent pool is empty. Add opponents before sampling.")
        return random.choice(self.opponents)






# Training Loop
OpponentPool = OpponentPool()
PolicyNetwork = PolicyNetwork()
PolicyNetwork.model.load_weights('policy_weights.h5')
ValueNetwork = ValueNetwork()
ValueNetwork.model.load_weights('value_weights.h5')
Environment = GomokuEnv()

NumEpisodes = 100
horizon = 30
n = 10
grad_norms = []

for i in range(0, NumEpisodes):
    print(i)
    if i % 20 == 0:
        OpponentPool.add_opponent(PolicyNetwork)
    if i % 5 == 0:
        PolicyNetwork.model.save_weights('policy_weights.h5')
        ValueNetwork.model.save_weights('value_weights.h5')


    gradients = []
    value_gradients = []
    for j in range(n):
        print('j quals', j )
        States = []
        Actions = []
        trajectory_gradient = [0]
        trajectory_value_gradient = [0]
        reward = 0
        t = 0
        Environment.reset()
        state = Environment.board
        while t<horizon and Environment.winner == 0:
            action = PolicyNetwork.sample(state)  # Sample from PolicyNetwork
            States.append(state)
            Actions.append(action)
            state ,reward ,terminate, _ = Environment.step(action)

            # Opponent is regarded as player -1
            if not terminate:
                opponent = OpponentPool.sample_opponent()
                opponent_action = opponent.sample(-1 * Environment.board)
                state, reward ,terminate, _ = Environment.step(opponent_action)
            t += 1

        # Calculate the gradient of the current trajectory
        if reward == 1:
            for k in range(len(States)):
                u = PolicyNetwork.compute_gradient(States[k], Actions[k])
                v = ValueNetwork.compute_gradient(States[k], reward)
                trajectory_gradient = [grad1 + grad2 for grad1, grad2 in zip(trajectory_gradient, u)]
                trajectory_value_gradient = [grad1 + grad2 for grad1, grad2 in zip(trajectory_value_gradient, v)]
        elif reward == -1:
            for k in range(len(States)):
                u = PolicyNetwork.compute_gradient(States[k], Actions[k])
                u = [-1.0 * grad for grad in u]
                v = ValueNetwork.compute_gradient(States[k], reward)
                trajectory_gradient = [grad1 + grad2 for grad1, grad2 in zip(trajectory_gradient, u)]
                trajectory_value_gradient = [grad1 + grad2 for grad1, grad2 in zip(trajectory_value_gradient, v)]
        if trajectory_gradient:
            grad_norms.append(tf.norm(trajectory_gradient[-1]))
            gradients.append(trajectory_gradient)
        if trajectory_value_gradient:
            value_gradients.append(trajectory_value_gradient)

    # Update PolicyNetwork using Gradient Descent
    PolicyNetwork.update(gradients)
    ValueNetwork.update(value_gradients)
    gradients = []
    value_gradients = []

