# connect4/agente/agent.py

import json
import os
import math
import random
from typing import Dict, Optional, Tuple

from connect4.connect_state import ConnectState
from connect4.encoder import encode_state
from connect4.utils import (
    get_legal_actions,
    is_action_legal,
    find_immediate_win_action,
    find_block_action_against_immediate_win,
)

from .config import MCTSConfig, get_default_config, MODEL_PATH
from .mcts import run_mcts_for_state
from .storage import save_binary_policy, load_binary_policy



# Tipo interno: Q_table[state_key][action] = (N, Q)
QTable = Dict[str, Dict[int, Tuple[float, float]]]
# Debug: ver cómo decide select_action
DEBUG_AGENT_SELECT = False  # activar solo con pocas partidas


class Connect4MCTSAgent:
    """
    Agente de Connect4 con Q(s,a) global tabular + MCTS.

    - En entrenamiento:
        * Para cada estado s, ejecuta MCTS.
        * Toma las estadísticas de la raíz (N_local, Q_local) por acción.
        * Las fusiona en una tabla global Q(s,a) que se persiste en JSON.

    - En juego/inferencia:
        * Si Q(s,a) tiene datos para el estado, juega acción greedy: argmax_a Q(s,a).
        * Si no hay datos suficientes, usa MCTS online como fallback.
    """

    def __init__(
        self,
        config: Optional[MCTSConfig] = None,
        q_table: Optional[QTable] = None,
    ) -> None:
        self.config: MCTSConfig = config or get_default_config()
        self.q_table: QTable = q_table or {}
        self._dirty: bool = False  # si se modificó la Q-table en esta sesión

        # Parámetro de exploración para softmax:
        #   - Valores más altos => más exploración (acciones subóptimas tienen más probabilidad).
        #   - Valores bajos (~0.1–0.5) => comportamiento casi-greedy.
        self.softmax_temperature: float = 0.5

    # ---------------------------------------------------------------------
    # Helpers internos sobre Q-table
    # ---------------------------------------------------------------------

    def _get_state_key(self, state: ConnectState) -> str:
        return encode_state(state)

    def _get_q_for_state(self, state_key: str) -> Dict[int, Tuple[float, float]]:
        """
        Devuelve el diccionario de acciones para un estado.
        Si no existe, devuelve {} (no crea entrada nueva).
        """
        return self.q_table.get(state_key, {})

    def update_q_with_terminal_reward(
        self,
        state_key: str,
        action: int,
        reward: float,
    ) -> None:
        """
        Actualiza Q(s,a) para un solo par (state_key, action) a partir
        de una recompensa terminal (tipo Monte Carlo).

        Esta rutina se usa ahora como bloque básico de aprendizaje Monte Carlo:
        al finalizar una partida, para cada (state_key, action) visitado por el
        agente, se llama con reward igual al retorno final de ese episodio.
        """
        if state_key not in self.q_table:
            self.q_table[state_key] = {}

        n_old, q_old = self.q_table[state_key].get(action, (0.0, 0.0))
        n_new = n_old + 1.0
        # media empírica incremental: Q_new = Q_old + (reward - Q_old)/N
        q_new = q_old + (reward - q_old) / n_new

        self.q_table[state_key][action] = (n_new, q_new)
        self._dirty = True




    def _select_action_from_q(
        self,
        state: ConnectState,
        state_key: str,
    ) -> Optional[int]:
        """
        Elige acción greedy según Q(s,a) para un estado dado.
        Solo considera acciones legales. Si no hay ninguna con información,
        devuelve None.
        """
        legal_actions = get_legal_actions(state)
        if not legal_actions:
            return None

        q_actions = self._get_q_for_state(state_key)
        if not q_actions:
            return None

        best_action: Optional[int] = None
        best_q: float = -float("inf")

        for a in legal_actions:
            if a not in q_actions:
                continue
            _, q_val = q_actions[a]
            if q_val > best_q:
                best_q = q_val
                best_action = a

        return best_action

    # ---------------------------------------------------------------------
    # Lógica principal de juego (inferencia)
    # ---------------------------------------------------------------------

    def _select_action_from_q_softmax(
        self,
        state_key: str,
        legal_actions: list[int],
        temperature: float | None = None,
    ) -> int | None:
        """
        Elige una acción entre las legales usando SOFTMAX sobre Q(s,a).
        - Si no hay Q para este estado, devuelve None.
        - Si temperature -> 0, se parece a una política greedy.
        """
        if temperature is None:
            temperature = self.config.temperature_eval

        q_dict = self._get_q_for_state(state_key)  # {a: (N, Q)} o {a: Q}
        if not q_dict:
            return None

        actions: list[int] = []
        q_values: list[float] = []

        for a in legal_actions:
            stats = q_dict.get(a)
            if stats is None:
                # Nunca hemos visto esta acción en este estado
                continue

            # Tus debug prints muestran algo tipo: {4: (N, Q), ...}
            if isinstance(stats, (tuple, list)) and len(stats) >= 2:
                _, q_val = stats
            else:
                q_val = float(stats)

            actions.append(a)
            q_values.append(q_val)

        if not actions:
            # No hay ninguna acción legal con Q conocido
            return None

        # Si la temperatura es muy baja, hacemos casi-greedy
        if temperature <= 1e-6:
            best_idx = max(range(len(actions)), key=lambda i: q_values[i])
            return actions[best_idx]

        # --- Softmax(Q / T) ---
        # 1) Escalar por temperatura
        scaled = [q / temperature for q in q_values]
        # 2) Estabilizar restando el máximo
        m = max(scaled)
        exps = [math.exp(s - m) for s in scaled]
        Z = sum(exps)
        # 3) Probabilidades
        probs = [e / Z for e in exps]

        # 4) Muestreo según esas probabilidades
        r = random.random()
        acc = 0.0
        for a, p in zip(actions, probs):
            acc += p
            if r <= acc:
                return a

        # Fallback por temas numéricos
        return actions[-1]



    def select_action(self, state: ConnectState) -> int:
        """
        Inferencia:
        - Primero: forzar ganar en 1 si se puede.
        - Luego: bloquear victoria inmediata del rival.
        - Después: usar Q(s,a) con SOFTMAX sobre los valores Q (exploración
          basada en probabilidad de victoria).
        - Si no hay datos Q suficientes, usar MCTS online como fallback.
        """
        state_key = self._get_state_key(state)
        legal_actions = get_legal_actions(state)

        if not legal_actions:
            # Sin movimientos posibles; por convención devolvemos 0.
            return 0
        
        # 1) Si puedo ganar en una, juego eso
        win_action = find_immediate_win_action(state)
        if win_action is not None:
            if DEBUG_AGENT_SELECT:
                print(f"[DEBUG select_action][WIN_IN_1] player={state.player}, action={win_action}")
            return win_action

        # 2) Si el rival tiene mate en 1, BLOQUEAR sí o sí
        block_action = find_block_action_against_immediate_win(state)
        if block_action is not None:
            if DEBUG_AGENT_SELECT:
                print(
                    f"[DEBUG select_action][BLOCK_IN_1] player={state.player}, "
                    f"action={block_action}"
                )
            return block_action

        # 3) Intentar acción usando Q(s,a) con SOFTMAX (exploración por prob. de victoria)
        #    Aquí puedes ajustar la temperatura según lo "explorador" que quieras ser.
        action = self._select_action_from_q_softmax(
            state_key=state_key,
            legal_actions=legal_actions,
        )
        if action is not None:
            return action


        # 4) Fallback: MCTS online
        if DEBUG_AGENT_SELECT:
            print(
                f"[DEBUG select_action][MCTS] player={state.player}, "
                f"state_key={state_key[:20]}..., sin Q o sin acciones, llamando MCTS"
            )
        best_action, _root_stats = run_mcts_for_state(
            state=state,
            root_player=state.player,
            config=self.config,
        )

        if best_action not in legal_actions:
            # Fallback defensivo extra
            if DEBUG_AGENT_SELECT:
                print(
                    f"[DEBUG select_action][MCTS_DONE] player={state.player}, "
                    f"best_action_invalida={best_action}, "
                    f"legal={legal_actions}"
                )
            best_action = legal_actions[0]

        return best_action

    # ---------------------------------------------------------------------
    # Entrenamiento local: mejora de Q(s,a) usando MCTS
    # ---------------------------------------------------------------------

    def improve_policy_with_mcts(self, state: ConnectState) -> int:
        """
        Entrenamiento:
        - Ejecuta MCTS desde el estado dado.
        - Usa las estadísticas en la raíz para actualizar Q(s,a) global.
        - Devuelve la acción greedy según la Q actualizada (para jugar la partida).
        """
        state_key = self._get_state_key(state)
        legal_actions = get_legal_actions(state)

        if not legal_actions:
            return 0

        best_action, root_stats = run_mcts_for_state(
            state=state,
            root_player=state.player,
            config=self.config,
            agent=self,  # <- activamos TBOPI en simulaciones
        )

        # Elegir acción final (greedy sobre Q, con fallback a best_action)
        action = self._select_action_from_q(state, state_key)
        if action is None or action not in legal_actions:
            action = best_action if best_action in legal_actions else legal_actions[0]

        return action

    # ---------------------------------------------------------------------
    # Persistencia de Q-table
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Persistencia de Q-table (versión binaria)
    # ---------------------------------------------------------------------

    def load_policy(self, path: str = MODEL_PATH) -> None:
        """
        Carga la Q-table desde un archivo binario comprimido (.pkl.gz)
        en el formato definido por agent.storage.

        Si el archivo no existe o está corrupto:
            - la Q-table queda como {} (sin política previa),
            - no se lanza excepción.
        """
        self.q_table = load_binary_policy(path)
        self._dirty = False

    def save_policy(self, path: str = MODEL_PATH) -> None:
        """
        Guarda la Q-table en disco como archivo binario comprimido (.pkl.gz),
        usando la representación compacta definida en agent.storage.

        Si _dirty es False, no hace nada.
        """
        if not self._dirty:
            return

        save_binary_policy(self.q_table, path)
        self._dirty = False
