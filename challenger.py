import os
import time

from pongpy.interfaces.team import Team
from pongpy.models.game_info import GameInfo
from pongpy.models.state import State


PLAYER_NAME = os.environ['PLAYER_NAME']
class ChallengerTeam(Team):
    def __init__(self, agent=None):
        self.agent = agent
        super().__init__()

    @property
    def name(self) -> str:
        return PLAYER_NAME

    def atk_action(self, info: GameInfo, state: State) -> int:
        '''
        前衛の青色のバーをコントロールします。
        '''
        action = self.agent.action(info, state)
        print('atk', info, state)
        return action

    def def_action(self, info: GameInfo, state: State) -> int:
        '''
        後衛のオレンジ色のバーをコントロールします。
        '''
        action = self.agent.action(info, state)
        print('def', info, state)
        return action