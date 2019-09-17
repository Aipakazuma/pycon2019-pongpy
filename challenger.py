import os

from pongpy.interfaces.team import Team
from pongpy.models.game_info import GameInfo
from pongpy.models.state import State

from rl.preprocessing import preprocessing_state

import numpy as np


PLAYER_NAME = os.environ['PLAYER_NAME']


class ChallengerTeam(Team):
    def __init__(self, agent=None):
        self.agent = agent
        self.game_state = None
        self.state = None
        self.model_output_action = None
        self._atk_action = 0
        self._def_action = 0
        self.before_action = 0
        super().__init__()

    @property
    def name(self) -> str:
        return PLAYER_NAME

    def atk_action(self, info: GameInfo, state: State) -> int:
        '''
        前衛の青色のバーをコントロールします。
        '''
        self.gaem_state = info
        self.state = state
        _state = preprocessing_state(state)

        if self.agent.steps % 10 == 0:
            q_values = self.agent.model.predict(np.array([_state]))
            action = self.agent.action(q_values[0])
            self.before_action = action
        else:
            action = self.before_action

        self.model_output_action = action
        _atk, _def = self.agent.policy.inv_action(action)
        self._atk_action = _atk
        self._def_action = _def
        print(action)
        return self._atk_action

    def def_action(self, info: GameInfo, state: State) -> int:
        '''
        後衛のオレンジ色のバーをコントロールします。
        '''
        return self._def_action
