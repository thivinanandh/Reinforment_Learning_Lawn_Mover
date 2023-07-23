## Lawn mover maze solver on openai gym Structure


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

MAZE_HEIGHT = 4
MAZE_WIDTH = 4
UNIT = 50

class lawnMover():
    def __init__(self):
    # Initializes the class
    # Define action and observation space
        self.actions = ["up", "down", "left", "right"]
        self.initiate_maze()
        pass

    def initiate_maze(self):
        ## Adjust Initial Position
        self.player_x = 0
        self.player_y = 0
        
        ## Set goal
        self.goal_x = MAZE_WIDTH-1
        self.goal_y = MAZE_HEIGHT-1

        ## Display MAZE_HEIGHT*MAZE_HEIGHT grid using ( MAZE_HEIGHT* UNIT )  * (MAZE_HEIGHT * UNIT) matplotlib
        ## Create a matrix of size MAZE_HEIGHT * MAZE_HEIGHT
        self.maze = np.zeros((MAZE_HEIGHT, MAZE_WIDTH))

        # display the maze as plot
        plt.imshow(self.maze, cmap='gray', interpolation='nearest')



    def step(self):
    # Executes one timestep within the environment
    # Input to the function is an action
    # Output is observation, reward, done, info
        pass
    def reset(self):
    # Resets the state of the environment to an initial state
        pass

    def render(self):
        pass


if __name__ == "__main__":
    env = lawnMover()

    plt.show()