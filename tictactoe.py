import numpy as np
import random
import pickle
import tkinter as tk
from tkinter import messagebox

# Initialize the main window
root = tk.Tk()
root.title("Tic Tac Toe with AI")
root.geometry("300x325")

# Q-learning parameters
alpha = 0.1       # Learning rate
gamma = 0.9       # Discount factor
epsilon = 0.1     # Exploration rate

# Q-table to store state-action values
try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
except FileNotFoundError:
    q_table = {}  # Initialize empty if file not found

# Game variables
board = [" " for _ in range(9)]
current_player = "X"

# Functions
def state_to_tuple(board):
    return tuple(board)

def get_available_actions(state):
    return [i for i, cell in enumerate(state) if cell == " "]

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(get_available_actions(state))
    else:
        q_values = q_table.get(state, [0]*9)
        max_value = max(q_values)
        actions = [i for i, q in enumerate(q_values) if q == max_value]
        return random.choice(actions)

def update_q_value(state, action, reward, next_state):
    current_q = q_table.get(state, [0]*9)[action]
    max_next_q = max(q_table.get(next_state, [0]*9))
    new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
    if state not in q_table:
        q_table[state] = [0] * 9
    q_table[state][action] = new_q

def check_winner():
    win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    for i, j, k in win_conditions:
        if board[i] == board[j] == board[k] and board[i] != " ":
            return board[i]
    return "Tie" if " " not in board else None

def reset_board(training_mode=False):
    global board, current_player
    board = [" " for _ in range(9)]
    current_player = "X"
    if not training_mode:
        for button in buttons:
            button.config(text=" ")

def button_click(button, index):
    global current_player
    if board[index] == " " and current_player == "X":
        board[index] = "X"
        button.config(text="X")
        winner = check_winner()
        if winner:
            end_game(winner)
        else:
            current_player = "O"
            root.after(500, ai_move)  # Delay AI move for a smoother UI experience

def find_blocking_or_winning_move(player):
    """Find a move that blocks the opponent or lets the AI win."""
    for i in range(9):
        if board[i] == " ":
            # Temporarily place player's move to check for win
            board[i] = player
            if check_winner() == player:
                board[i] = " "  # Reset the board
                return i  # Return the winning/blocking position
            board[i] = " "  # Reset the board
    return None

def ai_move():
    global current_player
    if current_player == "O":  # Ensure it is the AI's turn
        state = state_to_tuple(board)
        
        # Try to find a winning move first, then block if needed
        action = find_blocking_or_winning_move("O")  # Try to win
        if action is None:
            action = find_blocking_or_winning_move("X")  # Block player from winning
        if action is None:
            action = choose_action(state)  # Choose action based on Q-learning

        # Update board and UI with the AI's move
        board[action] = "O"
        buttons[action].config(text="O")
        
        # Calculate the next state and update Q-values
        next_state = state_to_tuple(board)
        reward = 10 if check_winner() == "O" else 0  # Increased reward for winning
        update_q_value(state, action, reward, next_state)
        
        winner = check_winner()
        if winner:
            end_game(winner)
        else:
            current_player = "X"  # Switch back to player after AI move

def end_game(winner):
    if winner == "Tie":
        messagebox.showinfo("Game Over", "It's a Tie!")
    else:
        messagebox.showinfo("Game Over", f"Player {winner} wins!")
    reset_board()

# Train the AI by playing against itself
def train_ai(episodes=5000):
    for _ in range(episodes):
        reset_board(training_mode=True)
        turn = "X"
        while True:
            state = state_to_tuple(board)
            if turn == "O":
                action = find_blocking_or_winning_move("O") or find_blocking_or_winning_move("X") or choose_action(state)
            else:
                action = random.choice(get_available_actions(state))
                
            board[action] = turn
            winner = check_winner()
            if winner:
                reward = 10 if winner == "O" else -10
                update_q_value(state, action, reward, state_to_tuple(board))
                break
            
            next_state = state_to_tuple(board)
            update_q_value(state, action, 0, next_state)
            turn = "O" if turn == "X" else "X"

train_ai()  # Pre-train the AI

# UI Setup
buttons = []
for i in range(9):
    button = tk.Button(root, text=" ", font=("Arial", 24), width=5, height=2,
                       command=lambda i=i: button_click(buttons[i], i))
    button.grid(row=i//3, column=i%3)
    buttons.append(button)

# Save Q-table when the game closes
def on_closing():
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
