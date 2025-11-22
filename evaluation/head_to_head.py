# tournament/evaluate_head_to_head.py
import os
import sys

# Agregar la carpeta raíz (tournament/) al path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from agent import Connect4MCTSAgent
from agent.config import get_default_config
from metrics import evaluate_head_to_head




# Rutas explícitas para modelo actual y modelo viejo
NEW_MODEL_PATH = "connect4/agente/policy_model.json"
OLD_MODEL_PATH = "connect4/metrics/old_policy_model.json"


def main() -> None:
    config = get_default_config()

    # Modelo viejo
    agent_old = Connect4MCTSAgent(config=config)
    agent_old.load_policy(OLD_MODEL_PATH)

    # Modelo nuevo (actual)
    agent_new = Connect4MCTSAgent(config=config)
    agent_new.load_policy(NEW_MODEL_PATH)

    win_rate_new, wins_new, wins_old, draws = evaluate_head_to_head(
        agent_a=agent_new,
        agent_b=agent_old,
        n_games=200,
    )

    total = wins_new + wins_old + draws
    print("Resultados Nuevo vs Viejo:")
    print(f"  Partidas totales: {total}")
    print(f"  Gana NUEVO: {wins_new}")
    print(f"  Gana VIEJO: {wins_old}")
    print(f"  Empates: {draws}")
    print(f"  Win-rate NUEVO: {win_rate_new:.3f}")


if __name__ == "__main__":
    main()
