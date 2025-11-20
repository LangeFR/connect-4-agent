# Abstract
from connect4.environment_state import EnvironmentState

# Types
from typing import Any

# Libraries
import numpy as np
import matplotlib.pyplot as plt


class ConnectState(EnvironmentState):
    ROWS = 6
    COLS = 7

    def __init__(
        self,
        board: np.ndarray | None = None,
        player: int = -1,
        last_move: tuple[int, int] | None = None,
        heights: list[int] | None = None,
    ):
        if board is None:
            self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        else:
            self.board = board.copy()
        self.player = player
        self.last_move = last_move
        self._winner: int | None = None

        if heights is not None:
            self.heights = heights.copy()
        else:
            # reconstruir alturas desde el board (solo cuando se crea el estado raíz)
            self.heights = [0] * self.COLS
            for c in range(self.COLS):
                col_vals = self.board[:, c]
                # número de fichas en la columna = celdas != 0
                self.heights[c] = int(np.count_nonzero(col_vals))



    def is_final(self) -> bool:
        # ganador por last_move
        if self.get_winner() != 0:
            return True
        # empate: todas las columnas llenas
        return all(h == self.ROWS for h in self.heights)


    def is_applicable(self, event: Any) -> bool:
        return (
            isinstance(event, int)
            and 0 <= event < self.COLS
            and self.is_col_free(event)
        )
    
    def get_winner(self) -> int:
        """
        Devuelve el jugador ganador (-1 o 1) si hay cuatro en línea
        pasando por la última jugada; 0 en caso contrario.
        Usa un caché interno para evitar recalcular muchas veces.
        """
        if self._winner is not None:
            return self._winner

        if self.last_move is None:
            self._winner = 0
            return 0

        row, col = self.last_move
        player = self.board[row, col]

        if player == 0:
            self._winner = 0
            return 0

        def count_direction(dr: int, dc: int) -> int:
            r = row + dr
            c = col + dc
            count = 0
            while (
                0 <= r < self.ROWS
                and 0 <= c < self.COLS
                and self.board[r, c] == player
            ):
                count += 1
                r += dr
                c += dc
            return count

        # Horizontal
        if 1 + count_direction(0, -1) + count_direction(0, 1) >= 4:
            self._winner = player
            return player

        # Vertical
        if 1 + count_direction(-1, 0) + count_direction(1, 0) >= 4:
            self._winner = player
            return player

        # Diagonal principal
        if 1 + count_direction(-1, -1) + count_direction(1, 1) >= 4:
            self._winner = player
            return player

        # Diagonal secundaria
        if 1 + count_direction(-1, 1) + count_direction(1, -1) >= 4:
            self._winner = player
            return player

        self._winner = 0
        return 0

    def is_col_free(self, col: int) -> bool:
        return self.heights[col] < self.ROWS

    def get_heights(self) -> list[int]:
        heights = []
        for c in range(self.COLS):
            col = self.board[:, c]
            for r in range(self.ROWS):
                if col[r] != 0:
                    heights.append(self.ROWS - r)
                    break
            else:
                heights.append(0)
        return heights

    def get_free_cols(self) -> list[int]:
        return [c for c in range(self.COLS) if self.heights[c] < self.ROWS]

    def transition(self, col: int) -> "ConnectState":
        if not self.is_applicable(col):
            raise ValueError(f"Move not allowed in column {col}.")

        # copiar tablero y alturas
        new_board = self.board.copy()
        new_heights = self.heights.copy()

        # fila en la que cae la nueva ficha
        current_height = new_heights[col]
        if current_height >= self.ROWS:
            raise ValueError(f"Column {col} is full.")

        row_played = self.ROWS - 1 - current_height

        new_board[row_played, col] = self.player
        new_heights[col] += 1

        return ConnectState(
            new_board,
            -self.player,
            last_move=(row_played, col),
            heights=new_heights,
        )
    
    def show(self, size: int = 1500, ax: plt.Axes | None = None) -> None:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        pos_red = np.where(self.board == -1)
        pos_yellow = np.where(self.board == 1)

        ax.scatter(pos_yellow[1] + 0.5, 5.5 - pos_yellow[0], color="yellow", s=size)
        ax.scatter(pos_red[1] + 0.5, 5.5 - pos_red[0], color="red", s=size)

        ax.set_ylim([0, self.board.shape[0]])
        ax.set_xlim([0, self.board.shape[1]])
        ax.set_xticks(np.arange(self.board.shape[1] + 1))
        ax.set_yticks(np.arange(self.board.shape[0] + 1))
        ax.grid(True)

        ax.set_title("Connect Four")

        if fig is not None:
            plt.show()
