# connect4/utils.py

from typing import Optional

from connect4.connect_state import ConnectState


def is_action_legal(state: ConnectState, action: int) -> bool:
    """
    Indica si una acción (columna) es legal en el estado dado.
    """
    return state.is_applicable(action)


def get_legal_actions(state: ConnectState) -> list[int]:
    """
    Devuelve todas las columnas legales en el estado actual.
    """
    return state.get_free_cols()


def is_winning_move(state: ConnectState, col: int, player: int) -> bool:
    """
    Devuelve True si poner una ficha de `player` en la columna `col`
    (en el estado `state`) produciría un 4 en línea.

    No modifica el estado. Usa board + heights en O(1).
    """
    rows = state.ROWS
    cols = state.COLS
    heights = state.heights

    # Si la columna está llena, no puede ser jugada ganadora.
    if heights[col] >= rows:
        return False

    # Fila donde caería la ficha
    row = rows - 1 - heights[col]
    board = state.board

    def count_direction(dr: int, dc: int) -> int:
        r = row + dr
        c = col + dc
        count = 0
        while 0 <= r < rows and 0 <= c < cols and board[r, c] == player:
            count += 1
            r += dr
            c += dc
        return count

    # Horizontal
    if 1 + count_direction(0, -1) + count_direction(0, 1) >= 4:
        return True
    # Vertical
    if 1 + count_direction(-1, 0) + count_direction(1, 0) >= 4:
        return True
    # Diagonal principal
    if 1 + count_direction(-1, -1) + count_direction(1, 1) >= 4:
        return True
    # Diagonal secundaria
    if 1 + count_direction(-1, 1) + count_direction(1, -1) >= 4:
        return True

    return False


def find_immediate_win_action(state: ConnectState) -> Optional[int]:
    """
    Devuelve una acción (columna) que da victoria inmediata al jugador actual,
    o None si no existe.
    """
    player = state.player
    legal_actions = get_legal_actions(state)

    for a in legal_actions:
        next_state = state.transition(a)
        if next_state.get_winner() == player:
            return a

    return None


def find_block_action_against_immediate_win(state: ConnectState) -> Optional[int]:
    """
    Si el rival (-state.player) tiene alguna jugada que gana inmediatamente
    en el tablero actual, devuelve una columna donde el jugador actual puede
    bloquear (jugando en la misma columna). Si no hay amenaza, devuelve None.
    """
    player = state.player
    opponent = -player
    legal_actions = get_legal_actions(state)

    for a in legal_actions:
        # ¿Si el oponente jugara en esta columna ahora mismo, ganaría?
        if is_winning_move(state, a, opponent):
            return a

    return None
