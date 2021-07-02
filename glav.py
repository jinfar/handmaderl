import os 
import sys 
import numpy as np
from collections import deque
import random
import time
import torch
sys.path.append(os.getcwd())
# -1 means block
# 5 means agent
# 0 means empty space


class Bolvanka(torch.nn.Module):

    def __init__(self, size):
        super(Bolvanka, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(size, 256),
            torch.nn.BatchNorm1d(),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 256),
            torch.nn.BatchNorm1d(),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 4),
        )

    def forwad(self, x):
        x = x.to('cuda')
        return self.model(x)


class BufferCreator():

    def __init__(self, buf, size=10000):
        self.buf = deque(maxlen=size)

    def dobav(self, state, action, reward, next_state):
        self.buf.append([state, action, reward, next_state])

    def sample(self, num_samples):
        assert num_samples <= len(self.buf), "malenkiy razmer viborki"
        return random.sample(self.buf, num_samples)

    def create_model(self):
        temp = torch.nn.Module()


class Agent():

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def move(self, direction, maze):
        assert direction in [0, 1, 2, 3], f"nevernoe napravlenie{direction}"
        # print(direction)
        if direction == 2:

            if self.y >= maze.shape[1] - 1:
                pass
            elif maze[self.x, self.y + 1] == -1:
                pass
            else:
                self.y += 1

        if direction == 3:
            if self.x >= maze.shape[0] - 1:
                pass
            elif maze[self.x + 1, self.y] == -1:
                pass
            else:
                self.x += 1

        if direction == 1:
            if self.x <= 0:
                pass
            elif maze[self.x - 1, self.y] == -1:
                pass
            else:
                self.x -= 1

        if direction == 0:
            if self.y <= 0:
                pass
            elif maze[self.x, self.y - 1] == -1:
                pass
            else:
                self.y -= 1


class Maze():

    def __init__(self, height=4, width=4, agent_random=False):
        self.height = height
        self.width = width
        self.agent_h = np.random.randint(
            height - 1, size=1)[0] if agent_random else 0
        self.agent_w = np.random.randint(
            width - 1, size=1)[0] if agent_random else 0
        self.agent = Agent(self.agent_h, self.agent_w)
        self.view = self.set_blocks(30, True)
        self.indexing = dict()
        self.multik = []

    def set_blocks(self, p=10, setit=False):
        # if setit:
            # np.random.seed(seed=122)
        kolich = self.height * self.width
        lab = np.random.choice([0, -1], size=kolich,
                               p=[(100 - p) / 100, p / 100])
        lab = lab.reshape(self.height, self.width)
        lab[self.agent.x, self.agent.y] = 0
        return lab

    def st(self, p, rnd=False):
        self.view = self.set_blocks(p, rnd)

    def visualize(self):
        temp = self.view.copy()
        temp[self.agent.x, self.agent.y] = 7
        return temp

    def move_random(self):
        direction = np.random.randint(0, 4, 1)[0]
        self.agent.move(direction, self.view)

    def move(self, direction):
        self.agent.move(direction, self.view)

    def got_it(self):
        return True if self.agent.x > self.height - \
            3 and self.agent.y > self.width - 3 else False

    def get_state_index(self):
        temp = self.get_state()
        temp = repr(temp)
        if len(self.indexing) < 1:
            self.indexing[temp] = len(self.indexing)
            return self.indexing[temp]
        if temp not in self.indexing.keys():
            self.indexing[temp] = len(self.indexing)
        return self.indexing[temp]

    def get_state(self):
        temp = self.view.copy()
        temp[self.agent.x, self.agent.y] = 7
        return temp

    def play_ep(self, direction):
        nach_state = self.get_state_index()
        self.agent.move(direction, self.view)
        kon_state = self.get_state_index()
        reward = 1000 if self.got_it() else -0.1
        if nach_state == kon_state:
            reward = -1
        return nach_state, direction, reward, kon_state

class DQNagent():

    def __init__(self, state, indexi):
        self.osn_model = Bolvanka(state.reshape(-1)).to('cuda')
        self.tar_model = Bolvanka(state.reshape(-1)).to('cuda')
        self.optim = torch.optim.Adam(self.osn_model.parameters(), lr=0.0003)
        self.qt = np.zeros((state.reshape(-1)), 4)
        self.indexi = indexi 

    def train(self, x):
        y_pred = self.osn_model(x)
            
        
def sozdanie(size = 6):
    maze = Maze(size, size)
    # print(maze.visualize(), '\n')

    while True:
        maze.view = maze.set_blocks(p=30)
        print(maze.visualize(), '\n')
        for move in range(3000):
            nap = random.randint(0, 3)
            maze.move(nap)
            if maze.got_it():
                break
        if maze.got_it():
            break
        # print("\1B[2J\x1B[1;1H")
        os.system('cls')
    return maze





def main():
    m = sozdanie(6)

    q_t = np.zeros((m.height * m.width, 4))
    lr = 0.01
    dr = 0.99
    state_next = random.random()
    np.set_printoptions(precision=3)
    chislo_hodov = []
    for episode in range(1000):
        i = 0
        m.multik = []
        m.agent.x, m.agent.y = 0, 0
        nap = random.randint(0, 3)
        while not m.got_it():
            m.multik.append(m.visualize())
            chislo = random.random()
            state, action, reward, state_next = m.play_ep(nap)
            #import ipdb; ipdb.set_trace()
            q_t[state, action] = q_t[state, action] * \
                (1 - lr) + lr * (reward + dr * np.max(q_t[state_next, :]))
            i += 1
            if chislo < episode * 0.04:
                nap = np.argmax(q_t[state_next])
            else:
                nap = random.randint(0, 3)
            #if i>200: import ipdb; ipdb.set_trace()
            if m.got_it():
                # chislo_hodov.append(i)
                print(f'Got it in {i} moves')
                if episode % 50 == 0:
                    for item in m.multik:
                        print("\x1B[2J\x1B[1;1H")
                        print(item, '\n')
                        time.sleep(0.1)
        # np.save(f'logs/QT-on_ep{episode}.npy',q_t)
        nap = random.random()


if __name__ == '__main__':
    main()
