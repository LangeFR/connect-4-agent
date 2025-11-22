  # connect4/agente/mcts.py

import math
import random
from typing import Dict, Optional, Tuple

from connect4.connect_state import ConnectState
from .config import MCTSConfig
from connect4.encoder import get_legal_actions, is_winning_move


class MCTSNode:
    """Nodo del árbol MCTS."""

    def __init__(
        self,
        state: ConnectState,
        parent: Optional["MCTSNode"] = None,
        action_from_parent: Optional[int] = None,
    ) -> None:
        self.state: ConnectState = state
        self.parent: Optional[MCTSNode] = parent
        self.action_from_parent: Optional[int] = action_from_parent
        self.children: Dict[int, MCTSNode] = {}
        self.n_visits: int = 0
        self.total_value: float = 0.0
        self.untried_actions: set[int] = set(get_legal_actions(state))

    def is_terminal(self) -> bool:
        """Indica si el estado del nodo es final."""
        return self.state.is_final()

    def is_fully_expanded(self) -> bool:
        """Indica si ya se expandieron todas las acciones legales."""
        return self.is_terminal() or len(self.untried_actions) == 0

    def best_child_ucb(self, c_explore: float) -> "MCTSNode":
        """Devuelve el hijo con mejor valor UCB."""
        best_score = -float("inf")
        best_child: Optional[MCTSNode] = None

        for action, child in self.children.items():
            if child.n_visits == 0:
                q_value = 0.0
            else:
                q_value = child.total_value / child.n_visits

            # Evitar log(0): usamos n_visits del padre + 1
            parent_visits = max(1, self.n_visits)
            ucb = q_value + c_explore * math.sqrt(
                math.log(parent_visits) / max(1, child.n_visits)
            )

            if ucb > best_score:
                best_score = ucb
                best_child = child

        if best_child is None:
            # Si algo sale mal, elegir hijo aleatorio
            best_child = random.choice(list(self.children.values()))

        return best_child

    def expand(self) -> "MCTSNode":
        """Expande una acción no probada y devuelve el nuevo hijo."""
        action = random.choice(list(self.untried_actions))
        self.untried_actions.remove(action)

        next_state = self.state.transition(action)
        child = MCTSNode(state=next_state, parent=self, action_from_parent=action)
        self.children[action] = child
        return child


def _tree_policy(
    node: MCTSNode, config: MCTSConfig
) -> Tuple[MCTSNode, int]:
    """
    Fase de selección/expansión.
    Avanza por el árbol hasta encontrar un nodo hoja para simular.
    Devuelve el nodo hoja y la profundidad alcanzada.
    """
    depth = 0
    current = node

    while not current.is_terminal() and depth < config.max_depth:
        if not current.is_fully_expanded():
            # Expandir una acción nueva
            child = current.expand()
            return child, depth + 1
        else:
            # Elegir el mejor hijo según UCB
            current = current.best_child_ucb(config.c_explore)
            depth += 1

    return current, depth


def _default_policy(
    state: ConnectState,
    root_player: int,
    max_depth_remaining: int,
) -> float:
    """
    Rollout con política heurística optimizada:
      1) Si el jugador actual puede ganar en 1, juega esa jugada.
      2) Si el rival puede ganar en 1 desde el estado actual, se intenta bloquear.
      3) Si no, prefiere columnas centrales.
      4) Fallback: random.

    Importante: usamos _is_winning_move para evitar crear estados
    intermedios (transition) durante la heurística. Solo hacemos
    transition una vez, con la acción elegida.
    """
    current = state
    depth = 0

    while not current.is_final() and depth < max_depth_remaining:
        legal_actions = get_legal_actions(current)
        if not legal_actions:
            break

        player = current.player
        opponent = -player

        chosen_action = None

        # 1) Intentar ganar en 1 jugada
        for a in legal_actions:
            if is_winning_move(current, a, player):
                chosen_action = a
                break

        # 2) Intentar bloquear victoria inmediata del rival
        if chosen_action is None:
            for a in legal_actions:
                if is_winning_move(current, a, opponent):
                    chosen_action = a
                    break


        # 3) Heurística: preferir columnas centrales
        if chosen_action is None:
            center_order = [3, 2, 4, 1, 5, 0, 6]
            for c in center_order:
                if c in legal_actions:
                    chosen_action = c
                    break

        # 4) Fallback: aleatorio (defensivo)
        if chosen_action is None:
            chosen_action = random.choice(legal_actions)

        # Aplicar SOLO la acción elegida
        current = current.transition(chosen_action)
        depth += 1

    winner = current.get_winner()
    if winner == root_player:
        return 1.0
    elif winner == 0:
        return 0.0
    else:
        return -1.0



def _backup(node: MCTSNode, reward: float) -> None:
    """
    Fase de retropropagación: actualiza n_visits y total_value
    a lo largo del camino desde la hoja hasta la raíz.
    """
    current: Optional[MCTSNode] = node
    while current is not None:
        current.n_visits += 1
        current.total_value += reward
        current = current.parent


def run_mcts_for_state(
    state: ConnectState,
    root_player: int,
    config: MCTSConfig,
) -> Tuple[int, Dict[int, Tuple[int, float]]]:
    """
    Ejecuta MCTS desde el estado dado y devuelve:
    - la mejor acción (columna)
    - un diccionario con estadísticas en la raíz: acción -> (N, Q)
    """
    root = MCTSNode(state=state, parent=None, action_from_parent=None)

    for _ in range(config.n_simulations):
        # Selección + expansión
        leaf, depth = _tree_policy(root, config)

        # Simulación
        reward = _default_policy(
            leaf.state,
            root_player=root_player,
            max_depth_remaining=config.max_depth - depth,
        )

        # Backpropagation
        _backup(leaf, reward)

    # Elegir acción final en la raíz según visitas
    if not root.children:
        # No hay acciones (rara vez en partida "normal"), devolvemos random legal
        legal = get_legal_actions(state)
        if not legal:
            return 0, {}
        action = random.choice(legal)
        return action, {}

    best_action = None
    best_visits = -1
    root_stats: Dict[int, Tuple[int, float]] = {}

    for action, child in root.children.items():
        if child.n_visits > 0:
            q_value = child.total_value / child.n_visits
        else:
            q_value = 0.0
        root_stats[action] = (child.n_visits, q_value)

        if child.n_visits > best_visits:
            best_visits = child.n_visits
            best_action = action

    # Fallback defensivo
    if best_action is None:
        best_action = random.choice(list(root.children.keys()))

    return best_action, root_stats
