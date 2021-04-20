import os, sys
import numpy as np

sys.path.append(os.getcwd())


class Agent():
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        
        self.y = y

    def move(self, direction, maze):
        if direction == 'right':
            if self.y >=maze.shape[1]-1:
                pass
            elif maze[self.x, self.y+1] == -1:
                pass
            else:
                self.y += 1

        if direction == 'down':
            if self.x >=maze.shape[0]-1:
                pass
            elif maze[self.x+1, self.y] == -1:
                pass
            else:
                self.x += 1

        if direction == 'up':
            if self.x <=0:
                pass
            elif maze[self.x+1, self.y] == -1:
                pass
            else:
                self.x -= 1
                
        if direction == 'left':
            if self.y <=0:
                pass
            elif maze[self.x, self.y-1] == -1:
                pass
            else:
                self.y -= 1


class Maze():
    def __init__(self, height=4, width=4):
        super().__init__()
        self.height = height
        self.width = width
        #self.view = np.zeros((height, width))
        self.view = self.set_blocks()
        self.agent = Agent(0,0)

    def set_blocks(self, p=10):
        #for i in range(self.height*self.width/2):
        kolich = self.height*self.width
        lab = np.random.choice([0,-1], size=(kolich,), p=[(100-p)/100, p/100]).reshape(self.height, self.width)
        lab[0,0] = 0
        return lab

    def visualize(self):
        temp = self.view.copy()
        temp[self.agent.x, self.agent.y] = 5
        print(temp,'\n')

    def move_random(self):
        direction = np.random.choice(['up', 'down', 'right', 'left'], 1)[0]
        #direction = 'down'
        print(direction)
        #m = self.view.copy()
        self.agent.move(direction, self.view)
        self.visualize()



if __name__ == '__main__':
    m = Maze(7,7)
    m.view = m.set_blocks(30)
    razi = 10 
    for i in range(razi):
        m.move_random()
