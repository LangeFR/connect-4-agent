from agent.policy import Policy
from evaluation.tournament.utils import find_importable_classes
from evaluation.tournament.tournament import run_tournament, play

# Read all files within subfolder of "groups"
participants = find_importable_classes("groups", Policy)

# Build a participant list (name, class)
players = list(participants.items())

# Run the tournament
champion = run_tournament(
    players,
    play,  # You could also create your own play function for testing purposes
    shuffle=True,
)
print("Champion:", champion)
