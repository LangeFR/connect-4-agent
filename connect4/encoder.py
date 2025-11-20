import numpy as np

from connect4.connect_state import ConnectState


def encode_state(state: ConnectState) -> str:
    """
    Codifica un estado de ConnectState en un string estable y hashable.
    Incluye el jugador al turno y el tablero aplanado.
    """
    board: np.ndarray = state.board
    player: int = state.player
    flat = board.flatten()
    flat_str = "".join(str(int(x)) for x in flat) # "-1|0000000...."
    key = f"{player}|{flat_str}"
    return key
