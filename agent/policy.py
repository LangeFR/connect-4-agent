import os
import json
import numpy as np
from connect4.policy import Policy
#from connect4.agente.encoder import encode_state

ROWS = 6
COLS = 7
MODEL_PATH = "policy_model.json"


def _legal_actions(board: np.ndarray):
    """Columnas donde la fila superior está vacía."""
    legal = []
    for c in range(COLS):
        if board[0, c] == 0:
            legal.append(c)
    return legal


def _drop(board: np.ndarray, col: int, player: int):
    """
    Simula dejar caer una ficha de 'player' en la columna 'col'.
    Devuelve un nuevo tablero o None si la columna está llena.
    """
    r = max([r for r in range(ROWS) if board[r, col] == 0], default=None)
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
            if np.all(board[r, c:c + 4] == player):
                return True

    # Vertical
    for r in range(ROWS - 3):
        for c in range(COLS):
            if np.all(board[r:r + 4, c] == player):
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
    - Suponemos que la partida empieza con -1.
    - Si hay un número par de fichas, mueve -1; si impar, mueve 1.
    Esto es solo para codificar el estado en policy_model.json.
    """
    tokens = int(np.count_nonzero(board))
    return -1 if tokens % 2 == 0 else 1


def _encode_state(board: np.ndarray) -> str:
    """
    Codifica el tablero como string para usar como llave en policy_model.json.
    Formato: "<player>|<board_flattened>"
    donde board_flattened es la concatenación de los 42 valores en orden fila-major.
    """
    player = _guess_current_player(board)
    flat = board.flatten()
    flat_str = "".join(str(int(x)) for x in flat)
    return f"{player}|{flat_str}"


class GroupAPolicy(Policy):
    """
    Política final:
    - Juega como jugador -1.
    - mount(): carga policy_model.json si existe.
    - act(s): recibe un np.ndarray (6x7) y devuelve una columna (0..6).
    Prioridad de decisión:
      1) Ganar en una jugada si es posible.
      2) Bloquear victoria inmediata del rival.
      3) Usar policy_model.json si hay acción aprendida y es legal.
      4) Heurística fija: preferencia por el centro (3,2,4,1,5,0,6).
    """

    def __init__(self) -> None:
        self.me = -1
        self.opp = 1
        # Q-table: state_key -> {action -> Q_value}
        self.q_table: dict[str, dict[int, float]] = {}


    def mount(self) -> None:
        """
        Inicialización "pesada": cargar el modelo de política si existe.
        Si algo falla, simplemente no se usa policy_table.
        """
        if not os.path.exists(MODEL_PATH):
            return

        try:
            with open(MODEL_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Ahora esperamos: state_key -> { action_str -> {"N": ..., "Q": ...} }
            table: dict[str, dict[int, float]] = {}

            if isinstance(data, dict):
                for state_key, actions_dict in data.items():
                    if not isinstance(actions_dict, dict):
                        continue

                    inner: dict[int, float] = {}
                    for action_str, stats in actions_dict.items():
                        try:
                            action_int = int(action_str)
                        except (TypeError, ValueError):
                            continue

                        if not isinstance(stats, dict):
                            continue

                        try:
                            q_val = float(stats.get("Q", 0.0))
                        except (TypeError, ValueError):
                            continue

                        inner[action_int] = q_val

                    if inner:
                        table[str(state_key)] = inner

            self.q_table = table
        except Exception:
            # Si hay cualquier problema leyendo el archivo, seguimos sin tabla
            self.q_table = {}



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

        # 1) Intentar ganar ya mismo
        for c in legal:
            newb = _drop(board, c, self.me)
            if newb is not None and _win(newb, self.me):
                return int(c)

        # 2) Bloquear victoria inmediata del rival
        for c in legal:
            newb = _drop(board, c, self.opp)
            if newb is not None and _win(newb, self.opp):
                return int(c)

        # 3) Intentar usar policy_model.json (Q-table) si hay entrada para este estado
        if self.q_table:
            key = _encode_state(board)
            if key in self.q_table:
                print("✓ KEY ENCONTRADA:", key)
            else:
                print("✗ KEY NO EXISTE:", key)
            actions_dict = self.q_table.get(key)
            if actions_dict:
                best_action = None
                best_q = -float("inf")
                for a, q_val in actions_dict.items():
                    if a in legal and q_val > best_q:
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
