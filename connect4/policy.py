from __future__ import annotations

import random
from typing import Protocol

from connect4.connect_state import ConnectState
from agent.agent import Connect4MCTSAgent
from agent.config import MCTSConfig, get_default_config


class PolicyProtocol(Protocol):
    """
    Protocolo orientativo para políticas.
    (No es estrictamente necesario, pero aclara la intención.)
    """
    def select_action(self, state: ConnectState) -> int: ...


class Policy:
    """
    Política principal basada en MCTS.

    Esta es la clase que dtos.Participant utiliza como tipo Policy.
    """

    def __init__(self, config: MCTSConfig | None = None) -> None:
        self._agent = Connect4MCTSAgent(config or get_default_config()) # get_default_config

    def select_action(self, state: ConnectState) -> int:
        return self._agent.select_action(state)


class RandomPolicy:
    """
    Política aleatoria: elige una columna válida al azar.

    Útil para probar el rendimiento de MCTS enfrentándolo contra un baseline simple.
    """

    def select_action(self, state: ConnectState) -> int:
        free_cols = state.get_free_cols()
        if not free_cols:
            # Si no hay columnas libres, no hay acción válida.
            # Esto solo debería ocurrir en estados finales.
            raise RuntimeError("No hay acciones válidas para RandomPolicy.")
        return random.choice(free_cols)
