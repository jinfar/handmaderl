import os
import sys
import numpy as np

sys.path.append(os.getcwd())


# -1 means block
# 5 means agent
# 0 means empty space


class Agent():
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def move(self, direction, maze):
        if direction == 'right':
            if self.y >= maze.shape[1]-1:
                pass
            elif maze[self.x, self.y+1] == -1:
                pass
            else:
                self.y += 1

        if direction == 'down':
            if self.x >= maze.shape[0]-1:
                pass
            elif maze[self.x+1, self.y] == -1:
                pass
            else:
                self.x += 1

        if direction == 'up':
            if self.x <= 0:
                pass
            elif maze[self.x-1, self.y] == -1:
                pass
            else:
                self.x -= 1

        if direction == 'left':
            if self.y <= 0:
                pass
            elif maze[self.x, self.y-1] == -1:
                pass
            else:
                self.y -= 1


class Maze():

    def __init__(self, height=4, width=4, agent_random=True):
        self.height = height
        self.width = width
        self.agent_h = np.random.randint(
            height-1, size=1)[0] if agent_random else height-1
        self.agent_w = np.random.randint(
            width-1, size=1)[0] if agent_random else width-1
        self.agent = Agent(self.agent_h, self.agent_w)
        self.view = self.set_blocks()

    def set_blocks(self, p=10):
        kolich = self.height*self.width
        lab = np.random.choice(
            [0, -1], size=(kolich,), p=[(100-p)/100, p/100]).reshape(self.height, self.width)
        lab[self.agent_h, self.agent_w] = 0
        return lab

    def visualize(self):
        temp = self.view.copy()
        temp[self.agent.x, self.agent.y] = 5
        print(temp, '\n')

    def move_random(self):
        direction = np.random.choice(['up', 'down', 'right', 'left'], 1)[0]
        print(direction)
        self.agent.move(direction, self.view)


if __name__ == '__main__':
    m = Maze(7, 7)
    m.view = m.set_blocks(30)
    razi = 9
    for i in range(razi):
        m.move_random()
        m.visualize()
    # print(dir(m.view))
    # print(m.__sizeof__())
