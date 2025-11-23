from dataclasses import dataclass

# Ruta donde se guarda/carga la política aprendida (estado -> mejor acción)
MODEL_PATH = "models/current/policy_model.json"


@dataclass
class MCTSConfig:
    # Número de simulaciones MCTS por movimiento
    n_simulations: int = 300
    # Constante de exploración de UCB
    c_explore: float = 1.4
    # Profundidad máxima de las simulaciones (rollouts)
    max_depth: int = 42  # máximo de turnos posibles en Connect4

    # Temperatura para el softmax en selección con Q
    # T = 0  -> política greedy pura (elige el mejor Q)
    # T > 0  -> política estocástica: más T = más exploración
    temperature_eval: float = 0.5

def get_default_config() -> MCTSConfig:
    """Devuelve una configuración por defecto para el agente MCTS."""
    return MCTSConfig()
