from pongpy.controllers.board import Board
from pongpy.definitions import PADDING, HEIGHT, WIDTH, BOARD_WIDTH, BOARD_HEIGHT, SET_POINT, FRAME_RATE
from pongpy.interfaces.team import Team
from pongpy.models.color import Color
from pongpy.sounds import play_bgm, init_sounds
from pongpy.pong import Game, Pong
from pongpy.cmd import dynamic_import
from rl.agent import Agent
from rl.preprocessing import preprocessing_state
from rl.memory import Memory
from rl.get_model import v1

import pyxel

import time


class Trainer():
    def __init__(self):
        self.agent = Agent()
        # フィールド内の相対座標
        # 左上が (0, 0) 。
        # 下に行くほど y が、右に行くほど x が増加する。
        #
        # is_right_side: bool  # 右側かどうか
        # mine_team: TeamState  # 自チーム
        # enemy_team: TeamState  # 相手チーム
        # ball_pos: Pos  # ボール
        # time: int  # 経過フレーム数
        # ↑から考えた入力
        # [自チーム atk.x, 自チーム atk.y,
        #  自チーム def.x, 自チーム def.y,
        #  敵チーム atk.x, 敵チーム atk.y,
        #  敵チーム def.x, 敵チーム def.y,
        #  ボール x, ボール y]
        # output
        # とりあえず全部どっちかに動かす.
        # [-1, -1], [-1, 1], [1, -1], [1, 1]
        self.model = v1(input_shape=(10,),
                        n_output=4)
        self.memory = Memory(max_size=100000)
        team1_path = 'challenger:ChallengerTeam'
        team2_path = 'enemy:EnemyTeam'
        self.team1 = dynamic_import(team1_path)
        self.team2 = dynamic_import(team2_path)
        self.game = GymPong(self, self.team1(self.agent), self.team2())
        self.n_episode = 10

    def train(self):
        # 本当はここで実行したかったけど
        # 返り値が帰ってこないですべてgame objでやる.
        self.game.update()


class GymPong(Pong):
    def __init__(self, trainer: Trainer, left_team: Team, right_team: Team):
        self.trainer = trainer
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

        state = None
        if self.board.p1.team.state is not None:
            state = self.board.p1.team.state

        before_reward = 0
        reward = 0
        done = False

        # actionを実行する.
        self.board.update()

        # ゲーム終了条件
        # action = self.board.p1.
        # reward = self.board.p1.score
        # reward -= before_reward
        action = self.board.p1.team.model_output_action

        # update後は状態が変更になっているので取得する.
        # FIXME: クソコード.
        next_state = None
        if state is None:
            state = self.board.p1.team.state
        else:
            next_state = self.board.p1.team.state

        if state.time != 0:
            if self.board.p1.score >= SET_POINT or self.board.p2.score >= SET_POINT:
                # デュース判定
                if abs(self.board.p1.score - self.board.p2.score) >= 2:
                    print(
                        f'{self.board.p1.score_label} {self.board.p2.score_label}')
                    # 最新のPyxel だと quit したときに Python 自体 exit してしまう
                    # pyxel.quit()
                    done = True
                    self.init()

            # memoryを追加
            _state = preprocessing_state(state)
            _next_state = preprocessing_state(next_state)
            self.trainer.memory.add(
                [_state, _next_state, action, reward, done])

            if done:
                # model update
                done = False
                print(len(self.trainer.memory.buffer))


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
