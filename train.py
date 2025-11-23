# train.py

import random

from connect4.connect_state import ConnectState
from agent.agent import Connect4MCTSAgent
from agent.config import get_default_config, MODEL_PATH
from connect4.utils import (
    find_immediate_win_action,
    find_block_action_against_immediate_win,
)
#from evaluation.tournament.utils import append_game_dump


OLD_MODEL_PATH = "models/baselines/old_policy_model.json"

# Modo de entrenamiento:
# "random"   -> vs agente aleatorio
# "self_play"-> self-play simétrico
# "old_play" -> vs política congelada (old_policy_model.json)
TRAIN_MODE = "old_play"

# ¿El agente que entrena debe bloquear mates en 1?
AGENT_ENABLE_BLOCK = False
# ¿El oponente debe bloquear mates en 1?
OPPONENT_ENABLE_BLOCK = True

# ¿El agente que entrena debe FORZAR ganar en 1 cuando puede?
AGENT_ENABLE_WIN_IN_1 = False
# ¿El oponente debe FORZAR ganar en 1 cuando puede?
OPPONENT_ENABLE_WIN_IN_1 = True

# Debug: traza de movimientos por episodio
DEBUG_TRAIN_MOVES = False       # pon True cuando quieras ver jugadas
DEBUG_TRAIN_MAX_EPISODES_LOG = 20  # cuántos episodios loguear como máximo


def random_policy(state: ConnectState) -> int:
    """Jugador aleatorio simple: elige cualquier columna legal al azar."""
    legal = state.get_free_cols()
    return random.choice(legal) if legal else 0


def play_game(
    agent_new: Connect4MCTSAgent,
    mode: str,
    agent_old: Connect4MCTSAgent | None = None,
    train_as_player: int = -1,
    episode_idx: int | None = None,
) -> tuple[int, ConnectState]:
    """
    Juega una partida completa según el modo:

    - mode == "random":
        * El jugador `train_as_player` es el agente nuevo (entrena Q con improve_policy_with_mcts).
        * El otro jugador es aleatorio.

    - mode == "self_play":
        * Ambos jugadores usan agent_new.improve_policy_with_mcts (self-play simétrico).
        * Ambos lados actualizan la misma Q-table.

    - mode == "old_play":
        * El jugador `train_as_player` es el agente nuevo (entrena Q).
        * El otro jugador es agent_old.select_action (política congelada).
          (agent_old nunca llama a improve_policy_with_mcts, así que no aprende).

    Devuelve:
        winner, estado_final
    """
    state = ConnectState()

    # Traza completa de jugadas (para debug): (player, action)
    moves: list[tuple[int, int]] = []

    while not state.is_final():
        if state.player == train_as_player:
            # --- Turno del AGENTE que entrena ---
            win_action = None
            if AGENT_ENABLE_WIN_IN_1:
                win_action = find_immediate_win_action(state)

            if win_action is not None:
                # Forzamos mate en 1 si está activado
                action = win_action
            else:
                # 2) Opcional: bloquear victoria inmediata del rival
                block_action = None
                if AGENT_ENABLE_BLOCK:
                    block_action = find_block_action_against_immediate_win(state)

                if block_action is not None:
                    action = block_action
                else:
                    # 3) Si no hay nada urgente, mejorar política vía MCTS + Q
                    action = agent_new.improve_policy_with_mcts(state)
        else:
            # --- Turno del OPONENTE ---
            win_action = None
            if OPPONENT_ENABLE_WIN_IN_1:
                win_action = find_immediate_win_action(state)

            if win_action is not None:
                # Si el oponente puede ganar y está activado, SIEMPRE lo hará
                action = win_action
            else:
                block_action = None
                if OPPONENT_ENABLE_BLOCK:
                    block_action = find_block_action_against_immediate_win(state)

                if block_action is not None:
                    # El oponente también bloquea mate en 1 si se lo permitimos
                    action = block_action
                else:
                    if mode == "random":
                        legal = state.get_free_cols()
                        action = random.choice(legal)
                    elif mode == "old_play":
                        if agent_old is None:
                            raise ValueError("agent_old no puede ser None en modo 'old_play'.")
                        action = agent_old.select_action(state)
                    elif mode == "self_play":
                        action = agent_new.select_action(state)
                    else:
                        raise ValueError(f"Modo desconocido: {mode}")

        # Para debug: registrar TODAS las jugadas
        moves.append((state.player, action))

        state = state.transition(action)

    winner = state.get_winner()

    if DEBUG_TRAIN_MOVES and (episode_idx is None or episode_idx <= DEBUG_TRAIN_MAX_EPISODES_LOG):
        moves_str = " ".join(
            f"{'X' if p == -1 else 'O'}{a}"
            for (p, a) in moves
        )
        print(
            f"[DEBUG TRAIN] ep={episode_idx} "
            f"train_as_player={train_as_player} winner={winner} "
            f"moves={moves_str}"
        )

    return winner, state


def main() -> None:
    print(f"=== Entrenamiento de Q(s,a) + MCTS para Connect4 (modo={TRAIN_MODE}) ===")

    config = get_default_config()
    agent_new = Connect4MCTSAgent(config=config)

    # Cargar Q-table actual (si existe)
    agent_new.load_policy(MODEL_PATH)

    agent_old: Connect4MCTSAgent | None = None
    if TRAIN_MODE == "old_play":
        # Creamos un agente congelado con la política vieja
        agent_old = Connect4MCTSAgent(config=config)
        agent_old.load_policy(OLD_MODEL_PATH)
        # No llamamos jamás a improve_policy_with_mcts sobre agent_old

    NUM_GAMES = 20  # ajusta según el tiempo que quieras entrenar

    # Debug: volcar estados finales de las partidas de entrenamiento
    DUMP_TRAIN_FINAL_STATES = True  # pon True cuando quieras inspeccionar
    TRAIN_DUMP_PATH = "debug/train_final_states.txt"

    wins_train = 0
    wins_opp = 0
    draws = 0

    for episode in range(1, NUM_GAMES + 1):
        train_as_player = -1 if (episode % 2 == 1) else 1

        winner, final_state = play_game(
            agent_new=agent_new,
            mode=TRAIN_MODE,
            agent_old=agent_old,
            train_as_player=train_as_player,
            episode_idx=episode,
        )

        # Contadores de resultado desde la perspectiva del jugador que entrena
        if winner == train_as_player:
            wins_train += 1
        elif winner == -train_as_player:
            wins_opp += 1
        else:
            draws += 1

        # Dump opcional del estado final
        if episode % 5 == 0 and DUMP_TRAIN_FINAL_STATES:
            header = (
                f"TRAIN game {episode}: "
                f"winner={winner}, mode={TRAIN_MODE}, train_as_player={train_as_player}"
            )
            #append_game_dump(TRAIN_DUMP_PATH, header, final_state)

        if episode % 5 == 0:
            total = wins_train + wins_opp + draws
            win_rate = wins_train / total if total > 0 else 0.0
            print(
                f"[{episode}/{NUM_GAMES}] "
                f"W_train={wins_train}  W_opp={wins_opp}  D={draws}  win_rate_train={win_rate:.3f}"
            )

    # Guardar la Q-table nueva como policy_model.json
    agent_new.save_policy(MODEL_PATH)
    print(f"Modelo actualizado guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
