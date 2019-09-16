from pongpy.controllers.board import Board
from pongpy.definitions import PADDING, HEIGHT, WIDTH, BOARD_WIDTH, BOARD_HEIGHT, SET_POINT, FRAME_RATE
from pongpy.interfaces.team import Team
from pongpy.models.color import Color
from pongpy.sounds import play_bgm, init_sounds
from pongpy.pong import Game, Pong
from pongpy.cmd import dynamic_import
from rl.agent import Agent

import pyxel

import time



class Trainer():
    def __init__(self):
        self.agent = Agent()

        team1_path = 'challenger:ChallengerTeam'
        team2_path = 'enemy:EnemyTeam'
        self.team1 = dynamic_import(team1_path)
        self.team2 = dynamic_import(team2_path)
        self.game = GymPong(self.team1(self.agent), self.team2())
        self.n_episode = 10
    
    def train(self):
        for episode in range(self.n_episode):
            done = self.game.update()
            
            # 無理やりrestart
            self.game.init()

class GymPong(Pong):
    def __init__(self, left_team: Team, right_team: Team):
        super().__init__(left_team, right_team)

    def init(self):
        try:
            # init_sounds()
            # self.left_team = left_team
            # self.right_team = right_team
            self.board = Board(BOARD_WIDTH, BOARD_HEIGHT,
                PADDING, PADDING, left_team=self.left_team,
                right_team=self.right_team)
            self.board.p1.score = 0
            self.board.p2.score = 0
            # play_bgm()
        except Exception as e:
            print(e)
        
        print('### init!! ###')

    def update(self):

        self.board.update()

        # ゲーム終了条件
        if self.board.p1.score >= SET_POINT or self.board.p2.score >= SET_POINT:
            # デュース判定
            if abs(self.board.p1.score - self.board.p2.score) >= 2:
                print(f'{self.board.p1.score_label} {self.board.p2.score_label}')
                # 最新のPyxel だと quit したときに Python 自体 exit してしまう
                # pyxel.quit()
                self.init()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()