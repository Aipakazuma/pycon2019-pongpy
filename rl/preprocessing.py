from pongpy.definitions import BOARD_WIDTH, BOARD_HEIGHT


h_max = BOARD_HEIGHT
h_min = 0
h_min_max = h_max - h_min
w_max = BOARD_WIDTH
w_min = 0
w_min_max = w_max - w_min
def preprocessing_state(state):
    mine_atk_pos_x = round((state.mine_team.atk_pos.x - w_min) / w_min_max, 2)
    mine_atk_pos_y = round((state.mine_team.atk_pos.y - h_min) / h_min_max, 2)

    mine_def_pos_x = round((state.mine_team.def_pos.x - w_min) / w_min_max, 2)
    mine_def_pos_y = round((state.mine_team.def_pos.y - h_min) / h_min_max, 2)

    enemy_atk_pos_x = round((state.enemy_team.atk_pos.x - w_min) / w_min_max, 2)
    enemy_atk_pos_y = round((state.enemy_team.atk_pos.y - h_min) / h_min_max, 2)

    enemy_def_pos_x = round((state.enemy_team.def_pos.x - w_min) / w_min_max, 2)
    enemy_def_pos_y = round((state.enemy_team.def_pos.y - h_min) / h_min_max, 2)

    ball_x = round((state.ball_pos.x - w_min) / w_min_max, 2)
    ball_y = round((state.ball_pos.y - h_min) / h_min_max, 2)
    return [mine_atk_pos_x, mine_atk_pos_y,
        mine_def_pos_x, mine_def_pos_y,
        enemy_atk_pos_x, enemy_atk_pos_y,
        enemy_def_pos_x, enemy_def_pos_y,
        ball_x, ball_y]
            
            
            
            
