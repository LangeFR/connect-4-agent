# agent/storage.py
"""
Módulo de persistencia binaria para la Q-table del agente.

Convierte entre:
    QTable (dict en memoria)  <->  BinaryPolicy (arrays densos)  <->  .pkl.gz

No usa JSON ni depende de connect4: solo trabaja con la estructura QTable.
"""

from __future__ import annotations

import os
import gzip
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

# Número fijo de acciones en Connect4 (columnas 0..6)
NUM_ACTIONS: int = 7

# Tipo en memoria usado por agent.Connect4MCTSAgent
# Q_table[state_key][action] = (N, Q)
QTable = Dict[str, Dict[int, Tuple[float, float]]]


@dataclass
class BinaryPolicy:
    """
    Representación compacta de la política en disco.

    - state_keys: lista de claves de estado (mismo formato que encode_state).
    - N: matriz S x NUM_ACTIONS con visitas N(s,a) en uint16.
    - Q: matriz S x NUM_ACTIONS con valores Q(s,a) en float16.
    """
    state_keys: List[str]
    N: np.ndarray  # shape = (S, NUM_ACTIONS), dtype=uint16
    Q: np.ndarray  # shape = (S, NUM_ACTIONS), dtype=float16


def qtable_to_binary(q_table: QTable) -> BinaryPolicy:
    """
    Convierte la Q-table en memoria (dict anidado) a una BinaryPolicy
    con arrays densos y tipos compactos.
    """
    # Filtrar estados vacíos por seguridad
    non_empty_items = [(k, v) for k, v in q_table.items() if v]
    if not non_empty_items:
        # Política vacía
        empty_N = np.zeros((0, NUM_ACTIONS), dtype=np.uint16)
        empty_Q = np.zeros((0, NUM_ACTIONS), dtype=np.float16)
        return BinaryPolicy(state_keys=[], N=empty_N, Q=empty_Q)

    # Ordenar claves para tener un orden estable (mejor compresión)
    state_keys = sorted(k for k, _ in non_empty_items)
    S = len(state_keys)

    N = np.zeros((S, NUM_ACTIONS), dtype=np.uint16)
    Q = np.zeros((S, NUM_ACTIONS), dtype=np.float16)

    max_uint16 = np.iinfo(np.uint16).max 

    for s_idx, state_key in enumerate(state_keys):
        actions_dict = q_table.get(state_key, {})
        if not actions_dict:
            continue

        for a, (N_sa, Q_sa) in actions_dict.items():
            # Ignorar acciones fuera de rango por seguridad
            if not (0 <= a < NUM_ACTIONS):
                continue

            # Truncar visitas a uint16 si se pasa del máximo
            n_val = int(N_sa)
            if n_val < 0:
                n_val = 0
            if n_val > max_uint16:
                n_val = max_uint16

            N[s_idx, a] = np.uint16(n_val)
            Q[s_idx, a] = np.float16(Q_sa)

    return BinaryPolicy(state_keys=state_keys, N=N, Q=Q)


def binary_to_qtable(bp: BinaryPolicy) -> QTable:
    """
    Reconstruye una QTable (dict anidado) a partir de una BinaryPolicy
    cargada desde disco.
    """
    q_table: QTable = {}

    state_keys = bp.state_keys
    N = bp.N
    Q = bp.Q

    # Validaciones básicas de forma
    if N.ndim != 2 or Q.ndim != 2:
        return {}
    if N.shape != Q.shape:
        return {}
    if N.shape[1] != NUM_ACTIONS:
        return {}

    S = N.shape[0]
    if S != len(state_keys):
        return {}

    for s_idx in range(S):
        state_key = state_keys[s_idx]
        actions: Dict[int, Tuple[float, float]] = {}

        for a in range(NUM_ACTIONS):
            n_val = int(N[s_idx, a])
            if n_val <= 0:
                # No se visitó esta acción en este estado
                continue

            q_val = float(Q[s_idx, a])
            # Mantener N como float para ser consistente con el resto del código
            actions[a] = (float(n_val), q_val)

        if actions:
            q_table[state_key] = actions

    return q_table


def save_binary_policy(q_table: QTable, path: str) -> None:
    """
    Serializa la QTable a disco como .pkl.gz usando BinaryPolicy.
    """
    bp = qtable_to_binary(q_table)

    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    print("Saving Policy")

    with gzip.open(path, "wb") as f:
        # Usar el protocolo más alto disponible para mejor tamaño
        pickle.dump(bp, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_binary_policy(path: str) -> QTable:
    """
    Carga una QTable desde un archivo .pkl.gz.
    Si el archivo no existe o hay error, devuelve {}.
    """
    if not os.path.exists(path):
        return {}

    print("Loading Policy...")

    try:
        with gzip.open(path, "rb") as f:
            bp: BinaryPolicy = pickle.load(f)
    except Exception:
        # Política ilegible o corrupta -> empezar desde cero
        print("")
        return {}

    return binary_to_qtable(bp)
