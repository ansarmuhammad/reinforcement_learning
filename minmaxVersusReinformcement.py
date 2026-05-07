import random
from collections import defaultdict
import math

# =====================================
# PARAMETERS
# =====================================
ALPHA = 0.5
GAMMA = 0.9
EPSILON = 0.1
EPISODES = 50000000   # reduced for faster test

# =====================================
# STATE UTILITIES
# =====================================
def empty_board():
    return tuple([0] * 9)

def get_available_actions(state):
    return [i for i, v in enumerate(state) if v == 0]

def apply_action(state, action, player):
    s = list(state)
    s[action] = player
    return tuple(s)

# =====================================
# GAME LOGIC
# =====================================
def check_winner(state):
    lines = [(0,1,2),(3,4,5),(6,7,8),
             (0,3,6),(1,4,7),(2,5,8),
             (0,4,8),(2,4,6)]
    for a,b,c in lines:
        if state[a] != 0 and state[a] == state[b] == state[c]:
            return state[a]
    return 0

def is_draw(state):
    return 0 not in state

# =====================================
# RL HELPERS (UNCHANGED)
# =====================================
def find_winning_move(state, player):
    for a in get_available_actions(state):
        if check_winner(apply_action(state, a, player)) == player:
            return a
    return None

def find_blocking_move(state, player):
    opponent = -player
    for a in get_available_actions(state):
        if check_winner(apply_action(state, a, opponent)) == opponent:
            return a
    return None

def has_losing_risk(state):
    opponent = -1
    for a in get_available_actions(state):
        if check_winner(apply_action(state, a, opponent)) == opponent:
            return True
    return False

Q = defaultdict(lambda: defaultdict(float))

# =====================================
# RL POLICY
# =====================================
def choose_action_rl(state):

    move = find_winning_move(state, 1)
    if move is not None:
        return move

    move = find_blocking_move(state, 1)
    if move is not None:
        return move

    actions = get_available_actions(state)
    danger = has_losing_risk(state)

    if not danger and random.random() < EPSILON:
        return random.choice(actions)

    max_q = max([Q[state][a] for a in actions])
    best = [a for a in actions if Q[state][a] == max_q]

    return random.choice(best)

# =====================================
# ✅ MINIMAX IMPLEMENTATION
# =====================================
def minimax(state, player):

    winner = check_winner(state)
    if winner == 1:
        return 1
    elif winner == -1:
        return -1
    elif is_draw(state):
        return 0

    actions = get_available_actions(state)

    if player == -1:  # minimax (opponent is minimizing RL)
        best = math.inf
        for a in actions:
            val = minimax(apply_action(state, a, -1), 1)
            best = min(best, val)
        return best
    else:
        best = -math.inf
        for a in actions:
            val = minimax(apply_action(state, a, 1), -1)
            best = max(best, val)
        return best

def choose_action_minimax(state, player):
    actions = get_available_actions(state)

    best_action = None

    if player == -1:
        best_value = math.inf
        for a in actions:
            val = minimax(apply_action(state, a, player), 1)
            if val < best_value:
                best_value = val
                best_action = a
    else:
        best_value = -math.inf
        for a in actions:
            val = minimax(apply_action(state, a, player), -1)
            if val > best_value:
                best_value = val
                best_action = a

    return best_action

# =====================================
# TRAIN RL
# =====================================
def train():
    for episode in range(EPISODES):
        state = empty_board()
        player = 1
        history = []

        while True:

            if player == 1:
                action = choose_action_rl(state)
            else:
                action = random.choice(get_available_actions(state))

            history.append((state, action, player))
            state = apply_action(state, action, player)

            winner = check_winner(state)

            if winner != 0:
                reward = 1 if winner == 1 else -1
                break

            if is_draw(state):
                reward = 0
                break

            player *= -1

        # Reward propagation
        for s, a, p in reversed(history):
            if p == 1:
                Q[s][a] += ALPHA * (reward - Q[s][a])
                reward *= GAMMA

    print("Training complete ✅")

# =====================================
# ✅ MATCH RL vs MINIMAX
# =====================================
def play_match(starting_player):

    state = empty_board()
    player = starting_player

    while True:

        if player == 1:
            action = choose_action_rl(state)
        else:
            action = choose_action_minimax(state, -1)

        state = apply_action(state, action, player)

        winner = check_winner(state)

        if winner != 0:
            return winner

        if is_draw(state):
            return 0

        player *= -1

# =====================================
# RUN 10 GAMES
# =====================================
def run_matches():

    rl_wins = 0
    minimax_wins = 0
    draws = 0

    for i in range(10):

        starting_player = 1 if i % 2 == 0 else -1
        result = play_match(starting_player)

        if result == 1:
            rl_wins += 1
        elif result == -1:
            minimax_wins += 1
        else:
            draws += 1

    print("\n=== RESULTS (10 Games) ===")
    print(f"RL Wins: {rl_wins}")
    print(f"Minimax Wins: {minimax_wins}")
    print(f"Draws: {draws}")

# =====================================
# RUN
# =====================================
if __name__ == "__main__":
    train()
    run_matches()
