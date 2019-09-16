import numpy as np


class Agent():
    def __init__(self):
        self.model = None
        self.policy = None
        self.memory = None
    
    def action(self, info, state):
        # return self.policy.select_action(state)
        return np.random.randint(-1, 2)

    def update(self):
        pass