import os

from pongpy.interfaces.team import Team
from pongpy.models.game_info import GameInfo
from pongpy.models.state import State

from rl.preprocessing import preprocessing_state


PLAYER_NAME = os.environ['PLAYER_NAME']


class ChallengerTeam(Team):
    def __init__(self, agent=None):
        self.agent = agent
        self.game_state = None
        self.state = None
        self.model_output_action = None
        self._atk_action = 0
        self._def_action = 0
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

        action = self.agent.action(info, _state)
        self.model_output_action = action
        self._atk_action = action
        self._def_action = action
        return action

    def def_action(self, info: GameInfo, state: State) -> int:
        '''
        後衛のオレンジ色のバーをコントロールします。
        '''
        return self._def_action
