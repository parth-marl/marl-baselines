import numpy as np
# --> Environment
class GridWorld():
    def __init__(self, size=5):
        self.size = size
        self.goal = (size-1, size-1)
        self.start = (0,0)
        self.state = self.start
        self.traps = [[1,1], [1,2], [1,3], [2,3], [3,1]]

        self.actions = {
            0: (0,-1),
            1:  (0,1),
            2:  (1,0),
            3:  (-1,0)
        }

    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        move = self.actions[action]

        row = max(0, min(self.size - 1, self.state[0] + move[0]))
        column = max(0, min(self.size - 1, self.state[1] + move[1]))

        self.state = (row, column)

        if self.state == self.goal:
            return self.state , 10, True
        elif self.state in self.traps:
            self.state = self.start
            return self.state, -10, False
        else:
            return self.state, -1, False   
