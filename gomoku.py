import gym
import numpy as np

class GomokuEnv(gym.Env):
    def __init__(self, board_size=7):
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size), dtype=int) # Initialize the game board
        # Current player = -1 or 1. The neuralnetwork is the policy for player = 1
        self.current_player = 1
        self.winner = 0
        # self.action_space = gym.spaces.Discrete(board_size ** 2)
        # self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.board_size, self.board_size), dtype=int)  # Board state

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.winner = 0
        return self.board.copy()

    def step(self, action):
        '''
        :param action: shape (board_size ** 2, )  np.array
        :return: state, reward, terminate, info
        '''
        row = action // self.board_size
        col = action % self.board_size

        if self.board[row][col] == 0:
            self.board[row][col] = self.current_player


            if self.check_win(row, col):
                reward = 1 if self.current_player == 1 else -1
                self.winner = self.current_player
                return self.board.copy(), reward, True, {}

            if self.is_board_full():
                self.winner = 0
                return self.board.copy(), 0, True, {}

            self.current_player = -1 if self.current_player == 1 else 1
            return self.board.copy(), 0, False, {}

        else:
            # Forfeit if choose position not vacant
            if self.current_player == 1:
                reward = -1
                self.winner = -1
            else:
                reward = 1
                self.winner = 1
            return self.board.copy(), reward, True, {}

    def render(self, mode='human'):
        for row in self.board:
            print(' '.join(map(str, row)))
        print()

    def check_win(self, row, col):
        '''
        :param row: indices of current move, int
        :param col: indices of current move,int
        :return: return True if this move end the game
        '''
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 5):
                r = row + dr * i
                c = col + dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == self.current_player:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                r = row - dr * i
                c = col - dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == self.current_player:
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    def is_board_full(self):
        return not np.any(self.board == 0)