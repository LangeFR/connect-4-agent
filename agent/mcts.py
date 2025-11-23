  # connect4/agente/mcts.py

import math
import random
from typing import Dict, Optional, Tuple

from connect4.connect_state import ConnectState
from .config import MCTSConfig
from connect4.utils import get_legal_actions, is_winning_move


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

def _default_policy_tbopi(
    state: ConnectState,
    root_player: int,
    max_depth_remaining: int,
    agent,
    epsilon: float = 0.2,
    eps_unknown: float = 0.4,
) -> float:
    """
    Rollout para TBOPI con exploración explícita de acciones sin Q:

    - En cada estado:
        0) (opcional) ganar en 1 / bloquear.
        1) Con prob. eps_unknown, si hay acciones sin Q(s,a), explora una al azar.
        2) Si no, con prob. 1-epsilon usa Q(s,a) entre acciones conocidas (argmax Q).
        3) Si no aplica nada de lo anterior, heurística clásica
           (ganar en 1, bloquear, centro, random).
    - Al final, se actualiza Q(s,a) para toda la trayectoria con la recompensa terminal.
    """
    current = state
    depth = 0
    trajectory = []  # (state_key, action, player_who_acted)

    while not current.is_final() and depth < max_depth_remaining:
        legal_actions = get_legal_actions(current)
        if not legal_actions:
            break

        state_key = agent._get_state_key(current)

        q_actions = agent._get_q_for_state(state_key)  # {a: (N, Q)}
        known_actions = {
            a: stats for a, stats in q_actions.items() if a in legal_actions
        }
        unknown_actions = [a for a in legal_actions if a not in q_actions]

        chosen_action = None

        # ganar en 1 / bloquear 
        player = current.player
        opponent = -player

        # ganar en 1
        for a in legal_actions:
            if is_winning_move(current, a, player):
                chosen_action = a
                break

        # bloquear en 1
        if chosen_action is None:
            for a in legal_actions:
                if is_winning_move(current, a, opponent):
                    chosen_action = a
                    break

        # 1) Explorar acciones sin Q(s,a)
        if chosen_action is None and unknown_actions and random.random() < eps_unknown:
            chosen_action = random.choice(unknown_actions)

        # 2) Usar Q(s,a) entre acciones conocidas (epsilon-greedy)
        use_q = (
            chosen_action is None
            and bool(known_actions)
            and (random.random() > epsilon)
        )

        if use_q and chosen_action is None:
            # stats = (N, Q_val)
            def q_value_of(action: int) -> float:
                N, Q_val = known_actions[action]
                return Q_val

            chosen_action = max(known_actions.keys(), key=q_value_of)

        # 3) Heurística clásica como fallback
        if chosen_action is None:
            # Preferir columnas centrales
            center_order = [3, 2, 4, 1, 5, 0, 6]
            for c in center_order:
                if c in legal_actions:
                    chosen_action = c
                    break

        # 4) Fallback: aleatorio
        if chosen_action is None:
            chosen_action = random.choice(legal_actions)

        trajectory.append((state_key, chosen_action, player))
        current = current.transition(chosen_action)
        depth += 1

    winner = current.get_winner()

    for state_key, action, actor in trajectory:
        if winner == actor:
            r = 1.0
        elif winner == 0:
            r = 0.0
        else:
            r = -1.0
        agent.update_q_with_terminal_reward(state_key, action, r)
        

    for state_key, action, actor in trajectory:
        if winner == actor:
            r = 1.0
        elif winner == 0:
            r = 0.0
        else:
            r = -1.0
        agent.update_q_with_terminal_reward(state_key, action, r)
        
    if winner == root_player:
        reward_root_root = 1.0
    elif winner == 0:
        reward_root_root = 0.0
    else:
        reward_root = -1.0
        reward_root = -1.0

    return reward_root
    return reward_root


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
    agent=None,
) -> Tuple[int, Dict[int, Tuple[int, float]]]:
    """
    Ejecuta MCTS desde el estado dado y devuelve:
    - la mejor acción (columna)
    - un diccionario con estadísticas en la raíz: acción -> (N, Q)

    Si agent is not None, la fase de simulación usa TBOPI con Q(s,a),
    actualizando la Q-table del agente en cada rollout.
    """
    root = MCTSNode(state=state, parent=None, action_from_parent=None)

    for _ in range(config.n_simulations):
        # Selección + expansión
        leaf, depth = _tree_policy(root, config)

        # Simulación
        max_depth_remaining = config.max_depth - depth
        if agent is not None:
            reward = _default_policy_tbopi(
                leaf.state,
                root_player=root_player,
                max_depth_remaining=max_depth_remaining,
                agent=agent,
            )
        else:
            reward = _default_policy(
                leaf.state,
                root_player=root_player,
                max_depth_remaining=max_depth_remaining,
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
