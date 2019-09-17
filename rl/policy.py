import numpy as np


class EpsGreedy():
    def __init__(self, eps=0.3, trainable=True):
        self.eps = eps
        self.trainable = trainable

    def select_action(self, q_value):
        if self.eps <= np.random.uniform(0, 1) or not self.trainable:
            # 最大の報酬を返す行動を選択する
            action = np.argmax(q_value)
        else:
            # ランダムに行動する
            action = np.random.choice(list(range(4)))

        return action

    def inv_action(self, model_output_action):
        a = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        return a[model_output_action]
