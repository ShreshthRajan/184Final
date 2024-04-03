import numpy as np
import copy
import math
from policynetwork import *
from valuenetwork import *
from node import *
from gomoku import *
import random

policy_network = PolicyNetwork()
policy_network.model.load_weights('policy_weights.h5')
value_network = ValueNetwork()
value_network.model.load_weights('value_weights.h5')


# Next state function (assuming it returns the next state given a state and action)
def next_state(node, action, tree_dict = {}):
    '''
    :param node: It is the current node which contains states etc
    :param action: The current action. (board ** 2,) shape numpy array
    :param tree_dict: it is the dictionary with list of nodes, key is nodes level to the root.
    :return: The node of the next state
    '''
    board, curr_player, winner, terminate = node.next_state(action)
    nodes_same_level = tree_dict.get(node.level_from_root + 1, False)
    if nodes_same_level:
        for n in nodes_same_level:
            # If the node is already exist
            if np.array_equal(n.board, board):
                node.children[action] = n
                return n
    next_state_node = Node(board,curr_player,winner,node.board_size,node.level_from_root + 1)
    tmp = tree_dict.get(next_state_node.level_from_root, [])
    if tmp:
        tmp.append(next_state_node)
    else:
        tree_dict[next_state_node.level_from_root] = [next_state_node]
    return next_state_node

def ucb_score(node, action, c_value = 1):
    '''
    Assume this node has children
    :param node: the current node. Node object
    :param action: action going to take, integer
    :param c_value:
    :return: the UCB upper bound calculated
    '''
    child_node = node.children[action]
    if child_node.visits == 0:
        return float('inf')
    p = policy_network.model.predict(np.expand_dims(child_node.board, axis=0)).flatten()
    u = child_node.av_value  + c_value * p[action] * math.sqrt(math.log(node.visits+1e-3) / child_node.visits+1e-3)
    return u

# Monte Carlo Tree Search algorithm with UCB action selection
def mcts(env, playouts=5, c_value = 1, search_level = 5):
    root = Node(env.board.copy(),env.current_player, env.winner, env.board_size)
    tree_dict = {}
    for _ in range(playouts):
        node = root
        path = [node]
        while(node.children):
            action = max(node.children, key=lambda a: ucb_score(node, a, c_value))
            node = node.children[action]
            path.append(node)

        start_level = node.level_from_root
        actions = [a for a in range(node.board_size**2)]
        for a in actions:
            node.children[a] = next_state(node,a)
        # rollout
        while(node.winner == 0 and node.level_from_root - start_level <= search_level):
            action = random.randint(0, node.board_size ** 2-1)
            node = next_state(node,action, tree_dict)
            path.append(node)

        # Backpropagation
        v = value_network.model.predict(np.expand_dims(node.board, axis=0)).max()
        for node in reversed(path):
            node.visits += 1
            if node.current_player == 1:
                node.av_value = node.visits/(node.visits+1)*node.av_value + 1/(node.visits+1)*v
            else:
                node.av_value = node.visits / (node.visits + 1) * node.av_value - 1 / (node.visits + 1) * v

    best_action = max(root.children, key=lambda a: ucb_score(node, a, c_value))
    return best_action

if __name__ == "__main__":
    Environment = GomokuEnv()
    Environment.step(12)
    print(mcts(Environment))
