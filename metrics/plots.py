from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


#  Helpers base: leer winrates de logs/train 

def load_winrates(path, step: int = 20):
    """
    Lee un archivo donde cada línea es un winrate (float).
    Devuelve:
      - episodios: [step, 2*step, 3*step, ...]
      - winrates:  lista de floats
    """
    path = Path(path)
    winrates: list[float] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                winrates.append(float(line))
            except ValueError:
                continue

    episodios = [step * (i + 1) for i in range(len(winrates))]
    return episodios, winrates


def _find_logs_train_dir():
    base = Path.cwd()
    candidates = [
        base / "logs" / "train",
        base.parent / "logs" / "train",
        base.parent.parent / "logs" / "train",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("No se encontró 'logs/train' cerca del notebook.")


#  1) Curvas “crudas” 

def plot_training_winrates(
    log_dir=None,
    files=("winrate1.txt", "winrate2.txt"),
    labels=("VS1_Mejorada", "VS2_Mejorada"),
    steps=(20, 20),
):
    if log_dir is None:
        log_dir = _find_logs_train_dir()
    else:
        log_dir = Path(log_dir)

    plt.figure(figsize=(10, 5))

    for fname, label, step in zip(files, labels, steps):
        episodios, winrates = load_winrates(log_dir / fname, step=step)
        plt.plot(episodios, winrates, marker="o", linestyle="-", label=label)

    plt.xlabel("Episodios de entrenamiento")
    plt.ylabel("Winrate vs oponente")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Versiones")
    plt.title("Curvas de aprendizaje por versión del agente")
    plt.tight_layout()
    plt.show()


#  2) Curvas suavizadas 

def moving_average(values, window: int = 3):
    """Promedio móvil simple (ventana hacia atrás)."""
    values = np.asarray(values, dtype=float)
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(values[start : i + 1].mean())
    return smoothed


def plot_smoothed_winrates(
    log_dir=None,
    files=("winrate1.txt", "winrate2.txt"),
    labels=("VS1_Mejorada", "VS2_Mejorada"),
    steps=(20, 20),
    window: int = 3,
):
    """
    Curvas suavizadas -> sirve para la parte de “validación del agente”
    porque muestra estabilidad del rendimiento en el tiempo.
    """
    if log_dir is None:
        log_dir = _find_logs_train_dir()
    else:
        log_dir = Path(log_dir)

    plt.figure(figsize=(10, 5))

    for fname, label, step in zip(files, labels, steps):
        episodios, winrates = load_winrates(log_dir / fname, step=step)
        winrates_smooth = moving_average(winrates, window=window)
        plt.plot(episodios, winrates_smooth, marker="o", linestyle="-", label=label)

    plt.xlabel("Episodios de entrenamiento")
    plt.ylabel("Winrate suavizado (promedio móvil)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Versiones")
    plt.title(f"Curvas suavizadas (ventana = {window})")
    plt.tight_layout()
    plt.show()


#  3) Curva de mejora relativa VS2 - VS1 

def plot_improvement_curve(
    log_dir=None,
    file_base="winrate1.txt",
    file_new="winrate2.txt",
    step_base: int = 20,
    step_new: int = 20,
    label_base="VS1_Mejorada",
    label_new="VS2_Mejorada",
):
    """
    Muestra Δ winrate = VS2 - VS1 por checkpoint -> parte fuerte para rúbrica 3 (optimización).
    """
    if log_dir is None:
        log_dir = _find_logs_train_dir()
    else:
        log_dir = Path(log_dir)

    ep1, wr1 = load_winrates(log_dir / file_base, step=step_base)
    ep2, wr2 = load_winrates(log_dir / file_new, step=step_new)

    # Alinear por longitud mínima
    k = min(len(wr1), len(wr2))
    wr1 = np.array(wr1[:k])
    wr2 = np.array(wr2[:k])
    episodios = ep1[:k]  # asumimos mismo step

    delta = wr2 - wr1

    plt.figure(figsize=(10, 4))
    plt.axhline(0.0, color="black", linewidth=1)
    plt.plot(episodios, delta, marker="o", linestyle="-")
    plt.xlabel("Episodios de entrenamiento")
    plt.ylabel("Δ winrate (VS2 - VS1)")
    plt.title(f"Mejora relativa de {label_new} sobre {label_base}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


#  4) Barras con winrate final de cada versión 

def plot_final_winrate_bar(
    log_dir=None,
    files=("winrate1.txt", "winrate2.txt"),
    labels=("VS1_Mejorada", "VS2_Mejorada"),
):
    """
    Gráfico de barras del winrate final de cada versión -> comparación clara de configuraciones.
    """
    if log_dir is None:
        log_dir = _find_logs_train_dir()
    else:
        log_dir = Path(log_dir)

    finals = []
    for fname in files:
        _, wr = load_winrates(log_dir / fname, step=1)
        finals.append(wr[-1] if wr else 0.0)

    x = np.arange(len(labels))

    plt.figure(figsize=(6, 4))
    plt.bar(x, finals)
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    for xi, val in zip(x, finals):
        plt.text(xi, val + 0.01, f"{val:.3f}", ha="center", va="bottom")
    plt.ylabel("Winrate final vs oponente")
    plt.title("Comparación de versiones (winrate final)")
    plt.tight_layout()
    plt.show()



plot_training_winrates()
plot_smoothed_winrates(window=3)
plot_improvement_curve()
plot_final_winrate_bar()
