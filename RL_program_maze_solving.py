import tkinter as tk
import time
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

MAZE_HEIGHT = 4
MAZE_WIDTH = 4
UNIT = 70


class Maze(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('maze')
        self.actions = ["up", "down", "left", "right"]
        self._built_maze()
    
    def position_to_block(self, x, y):

        startPixelx = x*UNIT
        endPixelx = (x+1)*UNIT
        startPixely = (MAZE_HEIGHT - y-1) *UNIT
        endPixely = (MAZE_HEIGHT - y + 1-1) *(UNIT)

        return startPixelx, endPixelx, startPixely, endPixely

    def _built_maze(self):

        #create canvas
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_HEIGHT*UNIT, width=MAZE_WIDTH*UNIT)

        #create grids
        for c in range(0, MAZE_WIDTH+1):
            self.canvas.create_line(UNIT*c, 0, UNIT*c, MAZE_HEIGHT*UNIT)

        for c in range(0, MAZE_HEIGHT+1):
            self.canvas.create_line(0,UNIT*c, MAZE_WIDTH*UNIT, UNIT*c)
        
        ## Block Coordinates
        #block1 2,0
        startPixelx, endPixelx, startPixely, endPixely = self.position_to_block(2,0)
        print("block1: ", startPixelx, endPixelx, startPixely, endPixely)
        self.block1 = self.canvas.create_rectangle( startPixelx, startPixely, endPixelx,  endPixely, fill='black')
        #block2 3,1
        startPixelx, endPixelx, startPixely, endPixely = self.position_to_block(3,1)
        self.block2 = self.canvas.create_rectangle( startPixelx,startPixely, endPixelx,  endPixely, fill='black')

        #block3 0,3
        startPixelx, endPixelx, startPixely, endPixely = self.position_to_block(0,3)
        self.block3 = self.canvas.create_rectangle( startPixelx, startPixely, endPixelx,  endPixely, fill='green')

        #block4 2,3
        startPixelx, endPixelx, startPixely, endPixely = self.position_to_block(2,3)
        self.block4 = self.canvas.create_rectangle( startPixelx, startPixely, endPixelx,  endPixely, fill='green')

        self.player_x = 0
        self.player_y = 0

        self.goal_x = MAZE_WIDTH-1
        self.goal_y = MAZE_HEIGHT-1

        # block 3,3
        startPixelx, endPixelx, startPixely, endPixely = self.position_to_block(self.goal_x,self.goal_y)
        self.goal = self.canvas.create_rectangle(startPixelx, startPixely, endPixelx,  endPixely, fill='blue')
        self.goaltxt = self.canvas.create_text(startPixelx+UNIT/2, endPixely-UNIT/2, text='Goal', fill='white')
        # block 0,0
        startPixelx, endPixelx, startPixely, endPixely = self.position_to_block(self.player_x,self.player_y)
        print("Player start position: ", startPixelx, endPixelx, startPixely, endPixelx)
        self.player = self.canvas.create_oval(startPixelx, startPixely, endPixelx, endPixely, fill='red')

        

        self.canvas.pack(padx = 150, pady= 150)

        # print("Player start position: ", self.player_x, self.player_y)
        # print("Goal position: ", self.goal_x, self.goal_y)
      
        return
    
    def step(self, action):

        old_x, old_y = self.player_x, self.player_y
        if action == "down":
            if self.player_y == 0:
                # print("Can't go dwn")
                return 0,0,0,0,False
                pass
            else:
                self.player_y -= 1

        elif action == "up":
            if self.player_y == (MAZE_HEIGHT-1):
                # print("Can't go up")
                return 0,0,0,0,False
                pass
            else:
                self.player_y += 1

        elif action == "left":
            if self.player_x == 0:
                # print("Can't go left")
                return 0,0,0,0,False
                pass
            else:
                self.player_x -= 1

        elif action == "right":
            if self.player_x == (MAZE_WIDTH-1):
                return 0,0,0,0,False
                print("Can't go right")
                pass
            else:
                self.player_x += 1

        
        
        oldStartx,oldEndx, oldStarty,  oldEndy = self.position_to_block(old_x, old_y)
        currentStartx, currentEndx, currentStarty, currentEndy = self.position_to_block(self.player_x, self.player_y)
        # print("Old player position: ", old_x, old_y, oldStartx, oldStarty)
        # print("New player position: ", self.player_x, self.player_y , currentStartx, currentStarty)

        # print("Shift in xx : ", (currentStartx -oldStartx), " Shift in yy : ", (oldStarty-currentStarty))
        
        self.canvas.move(self.player, (currentStartx -oldStartx), (currentStarty - oldStarty))
        # input("Press Enter to continue...")
        rewardGiven = False
        if self.player_x == self.goal_x and self.player_y == self.goal_y:

            reward = 100
            done = True
            rewardGiven = True
        
        elif self.player_x == 2 and self.player_y == 0:

            reward = -5
            done = True
            rewardGiven = True

        elif self.player_x == 3 and self.player_y == 1:

            reward = -5
            done = True
            rewardGiven = True

        elif self.player_x == 0 and self.player_y == 3:

            reward = 6
            done = False
            rewardGiven = True
        
        elif self.player_x == 2 and self.player_y == 3:
            reward = 6
            done = False
            rewardGiven = True

        else:
            #print("AT THE EDGE")
            reward = 0
            done = False
            rewardGiven = False
        isValidMove = True

        ### check for final actions and provide reward
        # if not rewardGiven:
        #     if action == "down" :
        #         reward = -6
        #     elif action == "up":
        #         reward = -5
        #     elif action == "left":
        #         reward = 5
        #     elif action == "right":
        #         reward = 6
        return (self.player_x, self.player_y, reward, done,isValidMove)

    def render(self):
        #time.sleep(0.5)
        self.update()

    def reset(self):
        #time.sleep(0.5)
        self.canvas.delete(self.player)
        ## Block Coordinates
        #block1 0,0
        startPixelx, endPixelx, startPixely, endPixely = self.position_to_block(0,0)
        self.player = self.canvas.create_oval(startPixelx, startPixely, endPixelx, endPixely, fill='red')
        self.player_x = 0
        self.player_y = 0
        self.update()
        #time.sleep(0.5)
        done = False
        return self.player_x, self.player_y, done








class learning():

    def __init__(self, alpha, gamma, epsilon):

        self.actions = ["up", "down", "left", "right"]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        #initialise Q-table
        self.q_table = pd.DataFrame(np.zeros((MAZE_HEIGHT*MAZE_WIDTH, len(self.actions))), columns=self.actions)

    def choose_action_e_greedy(self, state):

        player_x = state[0]
        player_y = state[1]

        state_id = player_x + player_y*MAZE_WIDTH

        if np.random.rand() < self.epsilon:
            #choose best action
            maxq_action_id = self.q_table.loc[state_id, :].argmax()
            greedy_action = self.actions[maxq_action_id]
            return greedy_action

        else:
            #choose random action
            random_action = np.random.choice(self.actions)
            return random_action
        
    def q_learn(self, state, action, reward, next_state):

        player_x = state[0]
        player_y = state[1]
        state_id = player_x + player_y*MAZE_WIDTH

        player_x = next_state[0]
        player_y = next_state[1]
        next_state_id = player_x + player_y*MAZE_WIDTH

        self.q_table.loc[state_id, action] += self.alpha*(reward + self.gamma * (self.q_table.loc[next_state_id].max()) - self.q_table.loc[state_id, action])

        return
    
    def SARSA(self, state, action, reward, next_state, next_action):

        player_x = state[0]
        player_y = state[1]
        state_id = player_x + player_y*MAZE_WIDTH

        player_x = next_state[0]
        player_y = next_state[1]
        next_state_id = player_x + player_y*MAZE_WIDTH

        self.q_table.loc[state_id, action] += self.alpha*(reward + self.gamma * (self.q_table.loc[next_state_id, next_action]) - self.q_table.loc[state_id, action])

        return

    def two_step_bootstrap_SARSA(self, state, action, reward, next_state, next_action, next_next_state, next_next_action):

        player_x = state[0]
        player_y = state[1]
        state_id = player_x + player_y*MAZE_WIDTH

        player_x = next_state[0]
        player_y = next_state[1]
        next_state_id = player_x + player_y*MAZE_WIDTH

        player_x = next_next_state[0]
        player_y = next_next_state[1]
        next_next_state_id = player_x + player_y*MAZE_WIDTH

        max_q_next = self.q_table.loc[next_state_id].max()
        next_q = self.q_table.loc[next_next_state_id, next_next_action]
        two_step_bootstrapped_q = reward + self.gamma*next_q
        self.q_table.loc[state_id, action] += self.alpha*(two_step_bootstrapped_q - self.q_table.loc[state_id, action])

        return







def update_SARSA():

    MAX_EPISODES = 300

    instance = learning(alpha=0.2, gamma=0.3, epsilon=0.5)
    totalrewardArray = []
    actionArray = []

    for episode in tqdm(range(1, MAX_EPISODES+1)):
        # print("EPISODE: ", episode)
        x, y, done = env.reset()
        env.render()

        instance.state = (x,y)
        rewardArray = []
        
        actionTaken = 0
        
        while not done:
            
            validAction = False
            while not validAction:
                action = instance.choose_action_e_greedy((x,y))               
                x_new, y_new, reward, done,isValidMove = env.step(action)
                # print(comp)
                if(isValidMove):
                    validAction = True
                
            rewardArray.append(reward)
            if(episode % 100 == 0):
                env.render()
            next_action = instance.choose_action_e_greedy((x_new,y_new))
            instance.SARSA((x,y), action, reward, (x_new, y_new), next_action)
            x = x_new
            y = y_new
            actionTaken += 1
        actionArray.append(actionTaken)
        if(100 in rewardArray):
            totalrewardArray.append(sum(rewardArray)-100)
        else:
            totalrewardArray.append(sum(rewardArray))

        # print( " Episode: ", episode, "Reward: ", sum(rewardArray))
    ## plot the reward curve vs episodes
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(np.arange(1, MAX_EPISODES+1), totalrewardArray , 'r-o' , linewidth=2, markersize=4,markevery=10)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    ax.set_title("Reward vs Episodes - sarsa")
    plt.savefig("sarsa_reward.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(np.arange(1, MAX_EPISODES+1), actionArray , 'b-o' , linewidth=2, markersize=4,markevery=10)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Number Of Actions Taken")
    ax.set_title("Number Of Actions Taken vs Episodes - sarsa")
    plt.savefig("sarsa_actions.png")
    plt.close()

    rewPerAction = [a/b for (a,b) in zip(totalrewardArray,actionArray)]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(np.arange(1, MAX_EPISODES+1), rewPerAction , 'g-o' , linewidth=2, markersize=4,markevery=10)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward Per Action")
    ax.set_title("Reward per action vs Episodes - sarsa")
    plt.savefig("sarsa_reward_per_action.png")
    plt.close()

    q_table = instance.q_table
    print(q_table)
    ## move the player as per the q_table
    x, y, done = env.reset()
    env.render()
    rewardArray = []
    
    while not done:
        
        # sleep for 0.5 seconds
        time.sleep(0.1)

        action = q_table.loc[x+y*MAZE_WIDTH, :].argmax()
        # print("Current State: ", x, y)
        # print("Action: ", instance.actions[action])
        x_new, y_new, reward, done,isValidMove = env.step(instance.actions[action])
        rewardArray.append(reward)
        # print("New State: ", x_new, y_new, "Reward: ", reward, "Done: ", done)
        env.render()
        x = x_new
        y = y_new
    print( "Reward: ", sum(rewardArray)-100)
    
    env.destroy()

    return 

def update_TWO_STEP_SARSA():

    MAX_EPISODES = 300

    instance = learning(alpha=0.35, gamma=0.25, epsilon=0.45)
    totalrewardArray = []
    actionArray = []

    for episode in tqdm(range(1, MAX_EPISODES+1)):
        # print("EPISODE: ", episode)
        x, y, done = env.reset()
        env.render()

        instance.state = (x,y)
        rewardArray = []
        
        actionTaken = 0
        
        while not done:
            
            validAction = False
            while not validAction:
                action = instance.choose_action_e_greedy((x,y))               
                x_new, y_new, reward, done,isValidMove = env.step(action)
                # print(comp)
                if(isValidMove):
                    validAction = True
                
            rewardArray.append(reward)
            if(episode % 100 == 0):
                env.render()
            next_action = instance.choose_action_e_greedy((x_new,y_new))
            
            if not done:
                x_new_new, y_new_new, reward_new, done_new,isValidMove_new = env.step(next_action)
                next_next_action = instance.choose_action_e_greedy((x_new_new,y_new_new))
            else:
                x_new_new = x_new
                y_new_new = y_new
                reward_new = reward
                done_new = done
                next_next_action = next_action

            

            instance.two_step_bootstrap_SARSA((x,y), action, reward, (x_new, y_new), next_action, (x_new_new, y_new_new), next_next_action)
            x = x_new
            y = y_new
            actionTaken += 1
        actionArray.append(actionTaken)
        if(100 in rewardArray):
            totalrewardArray.append(sum(rewardArray)-100)
        else:
            totalrewardArray.append(sum(rewardArray))

        # print( " Episode: ", episode, "Reward: ", sum(rewardArray))
    ## plot the reward curve vs episodes
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(np.arange(1, MAX_EPISODES+1), totalrewardArray , 'r-o' , linewidth=2, markersize=4,markevery=10)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    ax.set_title("Reward vs Episodes - two_sarsa")
    plt.savefig("two_sarsa_reward.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(np.arange(1, MAX_EPISODES+1), actionArray , 'b-o' , linewidth=2, markersize=4,markevery=10)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Number Of Actions Taken")
    ax.set_title("Number Of Actions Taken vs Episodes - two_sarsa")
    plt.savefig("two_sarsa_actions.png")
    plt.close()

    rewPerAction = [a/b for (a,b) in zip(totalrewardArray,actionArray)]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(np.arange(1, MAX_EPISODES+1), rewPerAction , 'g-o' , linewidth=2, markersize=4,markevery=10)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward Per Action")
    ax.set_title("Reward per action vs Episodes - two_sarsa")
    plt.savefig("two_sarsa_reward_per_action.png")
    plt.close()

    q_table = instance.q_table
    print(q_table)
    ## move the player as per the q_table
    x, y, done = env.reset()
    env.render()
    rewardArray = []
    
    while not done:
        
        # sleep for 0.5 seconds
        time.sleep(0.1)

        action = q_table.loc[x+y*MAZE_WIDTH, :].argmax()
        # print("Current State: ", x, y)
        # print("Action: ", instance.actions[action])
        x_new, y_new, reward, done,isValidMove = env.step(instance.actions[action])
        rewardArray.append(reward)
        # print("New State: ", x_new, y_new, "Reward: ", reward, "Done: ", done)
        env.render()
        x = x_new
        y = y_new
    print( "Reward: ", sum(rewardArray)-100)
    
    env.destroy()

    return 


def update_qlearn():
    """
    This function is used to implement the Q-learning algorithm
    Input:
        None
    Output:
        None
    
    This function does the following:
    1. Creates the maze environment
    2. Creates the learning instance
    3. Loops over the episodes
        4. Loops over the steps in each episode
            5. Chooses an action based on the epsilon greedy policy
            6. Updates the Q-table based on the Q-learning algorithm
    7. Returns the Q-table
    """

    MAX_EPISODES = 300

    instance = learning(alpha=0.3, gamma=0.7, epsilon=0.3)
    totalrewardArray = []
    actionArray = []
    for episode in tqdm(range(1, MAX_EPISODES+1)):
        # print("EPISODE: ", episode)
        x, y, done = env.reset()
        env.render()

        instance.state = (x,y)
        rewardArray = []
        actionTaken = 0
        while not done:
            action = instance.choose_action_e_greedy((x,y))
            #print(action)
            x_new, y_new, reward, done,comp = env.step(action)
            rewardArray.append(reward)
            if(episode % 100 == 0):
                env.render()
            instance.q_learn((x,y), action, reward, (x_new, y_new))
            x = x_new
            y = y_new
            actionTaken += 1
        actionArray.append(actionTaken)
        if(100 in rewardArray):
            totalrewardArray.append(sum(rewardArray)-100)
        else:
            totalrewardArray.append(sum(rewardArray))
    # print("Reward Array: ", rewardArray, " Length: ", len(rewardArray))

    ## plot the reward curve vs episodes
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(np.arange(1, MAX_EPISODES+1), totalrewardArray , 'r-o' , linewidth=2, markersize=4,markevery=10)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward")
    ax.set_title("Reward vs Episodes - q-learning")
    plt.savefig("qlearn_reward.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(np.arange(1, MAX_EPISODES+1), actionArray , 'b-o' , linewidth=2, markersize=4,markevery=10)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Number Of Actions Taken")
    ax.set_title("Number Of Actions Taken vs Episodes - q-learning")
    plt.savefig("qlearn_actions.png")
    plt.close()

    rewPerAction = [a/b for (a,b) in zip(totalrewardArray,actionArray)]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(np.arange(1, MAX_EPISODES+1), rewPerAction , 'g-o' , linewidth=2, markersize=4,markevery=10)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Reward Per Action")
    ax.set_title("Reward per action vs Episodes - q-learning")
    plt.savefig("qlearn_reward_per_action.png")
    plt.close()

    q_table = instance.q_table
    print(q_table)

    ## move the player as per the q_table
    x, y, done = env.reset()
    env.render()
    rewardArray = []
    while not done:
        
        # sleep for 0.5 seconds
        time.sleep(0.1)

        action = q_table.loc[x+y*MAZE_WIDTH, :].argmax()
        # print("Current State: ", x, y)
        # print("Action: ", instance.actions[action])
        x_new, y_new, reward, done,isValidMove = env.step(instance.actions[action])
        rewardArray.append(reward)
        # print("New State: ", x_new, y_new, "Reward: ", reward, "Done: ", done)
        env.render()
        x = x_new
        y = y_new
    print( "Reward: ", sum(rewardArray)-100)
    

    
    env.destroy()



    return 

if __name__ == '__main__':
    env = Maze()
    q_table = env.after(100, update_TWO_STEP_SARSA)
    env.mainloop()
    del env




