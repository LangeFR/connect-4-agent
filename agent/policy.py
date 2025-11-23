import os
import json
import math
from typing import Dict
import numpy as np
from connect4.policy import Policy

# Dimensiones fijas del tablero de Connect4
ROWS = 6
COLS = 7
# Ruta del modelo entrenado, consistente con agent/config.py
MODEL_PATH = "models/current/policy_model.json"


def _legal_actions(board: np.ndarray) -> list[int]:
    """
    Columnas donde la casilla de la fila superior está vacía.
    (mismas acciones legales que en ConnectState.get_free_cols)
    """
    legal: list[int] = []
    for c in range(COLS):
        if board[0, c] == 0:
            legal.append(c)
    return legal


def _drop(board: np.ndarray, col: int, player: int) -> np.ndarray | None:
    """
    Simula dejar caer una ficha de 'player' en la columna 'col'.
    Devuelve un nuevo tablero o None si la columna está llena.
    """
    r = max((r for r in range(ROWS) if board[r, col] == 0), default=None)
    if r is None:
        return None
    newb = board.copy()
    newb[r, col] = player
    return newb


def _win(board: np.ndarray, player: int) -> bool:
    """
    True si 'player' tiene cuatro en línea en 'board'.
    """
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if np.all(board[r, c : c + 4] == player):
                return True

    # Vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            if np.all(board[r : r + 4, c] == player):
                return True

    # Diagonales
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            # Diagonal principal (\)
            if all(board[r + i, c + i] == player for i in range(4)):
                return True
            # Diagonal secundaria (/)
            if all(board[r + 3 - i, c + i] == player for i in range(4)):
                return True

    return False


def _guess_current_player(board: np.ndarray) -> int:
    """
    Adivina quién juega:
    - En tu ConnectState el juego empieza con player = -1.
    - Tras cada jugada se alterna el jugador.
    - Por tanto: si #fichas es par -> turno de -1, si es impar -> turno de 1.
    """
    tokens = int(np.count_nonzero(board))
    return -1 if tokens % 2 == 0 else 1


def _encode_state(board: np.ndarray) -> str:
    """
    Codifica el tablero como string para usar como llave en policy_model.json.

    Formato compatible con connect4.encoder.encode_state:
        "<player>|<board_flattened>"

    donde board_flattened es la concatenación de los 42 valores (-1,0,1)
    en orden fila-major.
    """
    player = _guess_current_player(board)
    flat = board.flatten()
    flat_str = "".join(str(int(x)) for x in flat)
    return f"{player}|{flat_str}"


class GroupAPolicy(Policy):
    """
    Política final:
    - Juega como jugador -1.
    - mount(time_out): carga policy_model.json si existe.
    - act(s): recibe un np.ndarray (6x7) y devuelve una columna (0..6).

    Prioridad de decisión:
      1) Ganar en una jugada si es posible (para el jugador al turno).
      2) Bloquear victoria inmediata del rival.
      3) Usar policy_model.json con UCB sobre (N,Q) si hay entrada.
      4) Heurística fija: preferencia por el centro (3,2,4,1,5,0,6).
      5) Fallback: primera columna legal.
    """

    def __init__(self) -> None:
        self.me = -1
        self.opp = 1
        # stats_table: state_key -> { action_int -> {"N": int, "Q": float} }
        self.stats_table: dict[str, dict[int, dict[str, float]]] = {}
        # Constante de exploración UCB
        self.c_explore: float = 1.4
        # Timeout por jugada (lo setea Gradescope en mount)
        self.time_out: int = 10

    def mount(self, time_out: float | None = None) -> None:
        """
        Inicialización "pesada": cargar el modelo de política si existe.
        El autograder llamará a mount(time_out), donde time_out es el
        máximo de segundos permitidos por jugada.
        """
        self.time_out = time_out

        if not os.path.exists(MODEL_PATH):
            self.stats_table = {}
            return

        try:
            with open(MODEL_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            table: dict[str, dict[int, dict[str, float]]] = {}

            if isinstance(data, dict):
                for state_key, actions_dict in data.items():
                    if not isinstance(actions_dict, dict):
                        continue

                    inner: Dict[int, Dict[str, float]] = {}
                    for action_str, stats in actions_dict.items():
                        try:
                            action_int = int(action_str)
                        except (TypeError, ValueError):
                            continue

                        if not isinstance(stats, dict):
                            continue

                        # Leemos N y Q con defaults seguros
                        try:
                            n_val = int(stats.get("N", 0))
                        except (TypeError, ValueError):
                            n_val = 0

                        try:
                            q_val = float(stats.get("Q", 0.0))
                        except (TypeError, ValueError):
                            q_val = 0.0

                        inner[action_int] = {"N": n_val, "Q": q_val}

                    if inner:
                        table[str(state_key)] = inner

            self.stats_table = table
        except Exception:
            # Si hay cualquier problema leyendo el archivo, seguimos sin tabla
            self.stats_table = {}

    def act(self, s: np.ndarray) -> int:
        """
        Recibe el tablero como np.ndarray(6x7) con valores -1, 0, 1
        y devuelve una columna (0..6) donde jugar.
        """
        board = np.array(s, dtype=int, copy=True)
        legal = _legal_actions(board)

        if not legal:
            # Situación anómala, pero devolvemos algo por seguridad
            return 0

        # Jugador al turno deducido por paridad de fichas
        current_player = _guess_current_player(board)
        opponent = -current_player

        # 1) Intentar ganar ya mismo (para el jugador al turno)
        for c in legal:
            newb = _drop(board, c, current_player)
            if newb is not None and _win(newb, current_player):
                return int(c)

        # 2) Bloquear victoria inmediata del rival
        for c in legal:
            newb = _drop(board, c, opponent)
            if newb is not None and _win(newb, opponent):
                return int(c)

        # 3) Intentar usar policy_model.json (acción greedy sobre Q)
        if self.stats_table:
            key = _encode_state(board)
            state_stats = self.stats_table.get(key)
            if state_stats:
                best_action = None
                best_q = -float("inf")
                for a in legal:
                    stats_a = state_stats.get(a)
                    if not stats_a:
                        continue
                    q_val = float(stats_a.get("Q", 0.0))
                    if q_val > best_q:
                        best_q = q_val
                        best_action = a
                if best_action is not None:
                    return int(best_action)

        # 4) Heurística de preferencia por el centro
        for c in [3, 2, 4, 1, 5, 0, 6]:
            if c in legal:
                return int(c)

        # 5) Fallback: cualquier columna legal
        return int(legal[0])
