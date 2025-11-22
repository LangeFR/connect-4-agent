# tournament/evaluate_vs_random.py
import os
import sys

# Agregar la carpeta raíz (tournament/) al path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

from connect4.agente.agent import Connect4MCTSAgent
from connect4.agente.config import get_default_config, MODEL_PATH
from ..metrics import evaluate_vs_random



def main() -> None:
    config = get_default_config()
    agent = Connect4MCTSAgent(config=config)

    # Usa el modelo ACTUAL por defecto: connect4/agente/policy_model.json
    agent.load_policy(MODEL_PATH)

    win_rate, wins, losses, draws = evaluate_vs_random(agent, n_games=200)

    total = wins + losses + draws
    print("Resultados vs Random:")
    print(f"  Partidas totales: {total}")
    print(f"  Gana agente : {wins}")
    print(f"  Pierde agente: {losses}")
    print(f"  Empates: {draws}")
    print(f"  Win-rate agente: {win_rate:.3f}")


if __name__ == "__main__":
    main()
