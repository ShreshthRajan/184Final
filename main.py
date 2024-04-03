# This is a sample Python script.
from gomoku import *

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def coordinate_to_action(row, col, board_size):
    return row * board_size + col

if __name__ == '__main__':

    # Create Gomoku environment
    env = GomokuEnv(board_size=7)

    state = env.reset()

    done = False
    while not done:
        input_coord = input("Enter your move (row, column): ")
        row, col = map(int, input_coord.split(','))  # Extract row and column from the input

        input_action = coordinate_to_action(row, col, env.board_size)

        state, reward, done, info = env.step(input_action)

        env.render()

        if done:
            if reward == 1:
                print("You win!")
            elif reward == 0:
                print("It's a draw!")
            else:
                print("You lose!")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
