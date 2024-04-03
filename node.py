from gomoku import *


class Node:
    def __init__(self, board, current_player=1, winner=0, board_size=7, level_from_root=0):
        '''
        :param env: It is the gomoku environment status of the current game
        '''
        self.board = board
        self.current_player = current_player
        self.winner = winner
        self.board_size = board_size
        self.level_from_root = level_from_root
        self.children = {}
        self.visits = 0
        self.av_value = 0

    def next_state(self, action):
        '''
        :param action: the action which is an array (board_size ** 2, )
        :return: tuple (board_nextstate, current_player, winner, terminate)
        '''
        row = action // self.board_size
        col = action % self.board_size
        board = self.board.copy()

        if board[row][col] == 0:
            board[row][col] = self.current_player

            if self.check_win(row, col):
                return board, self.current_player, self.current_player, True

            if self.is_board_full():
                return board, self.current_player, 0, True

            self.current_player = -1 if self.current_player == 1 else 1
            return board, -1 * self.current_player, 0, False

        else:
            return board, self.current_player, -1 * self.current_player, True

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