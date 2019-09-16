def preprocessing_state(state):
    return [state.mine_team.atk_pos.x, state.mine_team.atk_pos.y,
            state.mine_team.def_pos.x, state.mine_team.def_pos.y,
            state.enemy_team.atk_pos.x, state.enemy_team.atk_pos.y,
            state.enemy_team.def_pos.x, state.enemy_team.def_pos.y,
            state.ball_pos.x, state.ball_pos.y]
