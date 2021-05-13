import os, sys
import numpy as np
from collections import deque 
sys.path.append(os.getcwd())


# -1 means block
# 5 means agent
# 0 means empty space


class Agent():
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def move(self, direction, maze):
        assert direction in [0, 1, 2, 3], "nevernoe napravlenie"
        print(direction)
        if direction == 2:

            if self.y >=maze.shape[1]-1:
                pass
            elif maze[self.x, self.y+1] == -1:
                pass
            else:
                self.y += 1

        if direction == 3:
            if self.x >=maze.shape[0]-1:
                pass
            elif maze[self.x+1, self.y] == -1:
                pass
            else:
                self.x += 1

        if direction == 1:
            if self.x <=0:
                pass
            elif maze[self.x-1, self.y] == -1:
                pass
            else:
                self.x -= 1

        if direction == 0:
            if self.y <=0:
                pass
            elif maze[self.x, self.y-1] == -1:
                pass
            else:
                self.y -= 1


class Maze():

    def __init__(self, height=4, width=4, agent_random=True):
        self.height = height
        self.width = width
        self.agent_h = np.random.randint(height-1, size=1)[0] if agent_random else height-1
        self.agent_w = np.random.randint(width-1, size=1)[0] if agent_random else width-1
        self.agent = Agent(self.agent_h, self.agent_w)
        self.view = self.set_blocks()

    def set_blocks(self, p=10):
        kolich = self.height*self.width
        lab = np.random.choice([0,-1], size=(kolich,), p=[(100-p)/100, p/100]).reshape(self.height, self.width)
        lab[self.agent_h,self.agent_w] = 0
        return lab

    def visualize(self):
        temp = self.view.copy()
        temp[self.agent.x, self.agent.y] = 5
        print(temp,'\n')

    def move_random(self):
        direction = np.random.randint(0, 4, 1)[0]
        self.agent.move(direction, self.view)

    def got_it(self):
        return True if self.agent.x > self.height-3 and self.agent.y > self.width-3 else False

    def show_a(self):
        print(self.agent.x, self.agent.y)

    def record_ep(self):
        a_pos_nach = [self.agent.x, self.agent.y]
        direction = np.random.randint(0, 4, 1)[0]
        self.agent.move(direction, self.view)
        a_pos_kon = [self.agent.x, self.agent.y]
        reward = 10 if self.got_it() else -0.1
        return self.view, a_pos_nach, a_pos_kon, direction, reward

    def get_state(self):
        temp = self.view.copy()
        temp[self.agent.x, self.agent.y] = 5
        return temp 

    def play_ep(self):
        nach_state  = self.get_state()
        direction = np.random.randint(0, 4, 1)[0]
        self.agent.move(direction, self.view)
        kon_state  = self.get_state()
        reward = 10 if self.got_it() else -0.1
        return nach_state, direction, reward, kon_state 

class BufferCreator():
    def __init__(self, buf, size = 10000):
        self.buf = deque(maxlen=size)

    def dobav(self, state, action, reward, next_state):
        self.buf.append([state, action, reward, next_state])


def main():
    m = Maze(5, 5)
    m.view = m.set_blocks(10)
    razi = 99
    itog = []
    for i in range(razi):
        if m.got_it():
            print(f'Got it in {i} moves')
            break
        itog.append([m.record_ep()])
        m.visualize()
    if not m.got_it():
        print('didnt, get it ')

if __name__ == '__main__':
    main()


    
