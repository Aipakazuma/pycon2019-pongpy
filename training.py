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
from rl.policy import EpsGreedy
from rl.get_model import v1

import pyxel
import tensorflow as tf
import tensorboard.summary as tb_summary

import time
from datetime import datetime
import os


class Trainer():
    def __init__(self):
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
        model = v1(input_shape=(10,),
                   n_output=4)
        self.agent = Agent(model,
                           memory=Memory(max_size=100000),
                           policy=EpsGreedy(eps=0.3),
                           batch_size=512)

        _path = os.path.join('model', '20190917_140858', 'weights_200.h5')
        self.agent.model.load_weights(_path)
        self.agent.target_update()

        team1_path = 'challenger:ChallengerTeam'
        team2_path = 'enemy:EnemyTeam'
        self.team1 = dynamic_import(team1_path)
        self.team2 = dynamic_import(team2_path)
        self.game = GymPong(self, self.team1(self.agent), self.team2())
        self.n_episode = 10

    def train(self):
        # 本当はここで実行したかったけど
        # 返り値が帰ってこないのですべてgame objでやる.
        self.game.update()


class GymPong(Pong):
    def __init__(self, trainer: Trainer, left_team: Team, right_team: Team):
        self.trainer = trainer
        self.steps = 0
        self.episode = 0
        self.before_reward = 0
        self.total_rewards = 0

        self._date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join('logs', self._date_str)
        os.mkdir(os.path.join('model', self._date_str))
        self.writer = tf.summary.FileWriter(log_path)
        super().__init__(left_team, right_team)

    def init(self):
        try:
            self.before_reward = 0
            self.steps = 0
            self.total_rewards = 0
            self.board = Board(BOARD_WIDTH, BOARD_HEIGHT,
                               PADDING, PADDING, left_team=self.left_team,
                               right_team=self.right_team)
            self.board.p1.score = 0
            self.board.p2.score = 0
        except Exception as e:
            print(e)

    def update(self):

        state = None
        if self.board.p1.team.state is not None:
            state = self.board.p1.team.state

        done = False
        reward = 0

        # actionを実行する.
        self.board.update()

        # scoreをとったときにしかrewardを与えない
        # scoreはカウントされていく
        # state.mine_team.score - before state.mine_team.score
        if state is not None:
            reward = state.mine_team.score - self.before_reward

        # ゲーム終了条件
        action = self.board.p1.team.model_output_action

        # update後は状態が変更になっているので取得する.
        # FIXME: クソコード.
        next_state = None
        if state is None:
            state = self.board.p1.team.state
        else:
            next_state = self.board.p1.team.state

        self.steps += 1
        self.trainer.agent.steps += 1
        if state.time != 0:
            SET_POINT = 3
            if self.board.p1.score >= SET_POINT or self.board.p2.score >= SET_POINT:
                # デュース判定
                if abs(self.board.p1.score - self.board.p2.score) >= 2:
                    print(
                        f'{self.board.p1.score_label} {self.board.p2.score_label}')
                    # 最新のPyxel だと quit したときに Python 自体 exit してしまう
                    # pyxel.quit()
                    done = True
                    self.episode += 1
                    if self.board.p1.score > self.board.p2.score:
                        reward += 10

            # memoryを追加
            _state = preprocessing_state(state)
            _next_state = preprocessing_state(next_state)
            self.trainer.agent.memory.add([_state, _next_state,
                                           action, reward, done])

            self.total_rewards += reward
            self.before_reward = state.mine_team.score
            if done and self.trainer.agent.trainable:
                # model update
                done = False
                _len = len(self.trainer.agent.memory.buffer)
                print('episode: {}, memory_size: {}, total_reward: {}'.format(self.episode,
                                                                              _len,
                                                                              self.total_rewards))
                loss, getted_reward = self.trainer.agent.update()
                self.writer.add_summary(tb_summary.scalar_pb('loss', loss),
                    global_step=self.episode)
                self.writer.add_summary(tb_summary.scalar_pb('reward', self.total_rewards),
                    global_step=self.episode)
                self.writer.add_summary(tb_summary.scalar_pb('step_per_episode', self.steps),
                    global_step=self.episode)
                self.writer.add_summary(tb_summary.scalar_pb('learning_reward', getted_reward),
                    global_step=self.episode)

                if self.episode % 100 == 0:
                    self.trainer.agent.target_update()
                
                if self.episode % 100 == 0:
                    save_path = os.path.join('model', self._date_str,
                        'weights_{}.h5'.format(self.episode))
                    self.trainer.agent.model.save_weights(save_path)

                self.init()


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
