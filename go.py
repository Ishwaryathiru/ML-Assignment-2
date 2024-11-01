import numpy as np
import random
import tkinter as tk
from tkinter import messagebox

# Step 1: Define the Go Game Environment
class GoGame:
    def __init__(self, board_size=5):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size))
        self.current_player = 1  # Player 1 starts

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size))
        self.current_player = 1
        return self.board.flatten()

    def get_valid_actions(self):
        valid_actions = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x, y] == 0:  # Empty spot
                    valid_actions.append((x, y))
        return valid_actions

    def step(self, action):
        x, y = action
        if self.board[x, y] != 0:
            raise ValueError("Invalid action: position already occupied")

        self.board[x, y] = self.current_player
        reward = self.get_reward()
        done = self.is_done()
        self.current_player = 3 - self.current_player  # Switch player (1 <-> 2)

        return self.board.flatten(), reward, done

    def get_reward(self):
        return 1 if self.current_player == 1 else -1  # Placeholder reward

    def is_done(self):
        return len(self.get_valid_actions()) == 0  # Game ends when no valid moves are left

    def check_win(self, state):
        # Check for five in a row horizontally, vertically, and diagonally
        for x in range(self.board_size):
            for y in range(self.board_size):
                if state[x, y] != 0:  # Only check if there's a piece
                    # Check horizontal
                    if y <= self.board_size - 5 and all(state[x, y + i] == state[x, y] for i in range(5)):
                        return True
                    # Check vertical
                    if x <= self.board_size - 5 and all(state[x + i, y] == state[x, y] for i in range(5)):
                        return True
                    # Check diagonal (top-left to bottom-right)
                    if x <= self.board_size - 5 and y <= self.board_size - 5 and all(state[x + i, y + i] == state[x, y] for i in range(5)):
                        return True
                    # Check diagonal (bottom-left to top-right)
                    if x >= 4 and y <= self.board_size - 5 and all(state[x - i, y + i] == state[x, y] for i in range(5)):
                        return True
        return False

    def render(self):
        print(self.board)


# Step 2: Implement Q-Learning Agent
class QLearningAgent:
    def __init__(self, actions, board_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        self.q_table = {}
        self.actions = actions
        self.board_size = board_size  # Save the board size for blocking logic
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def get_state_key(self, state):
        return tuple(state)

    def choose_action(self, state, valid_actions):
        state_key = self.get_state_key(state)

        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(valid_actions)  # Choose a valid random action

        q_values = {action: self.q_table.get(state_key, {}).get(action, 0) for action in valid_actions}
        return max(q_values, key=q_values.get)

    def learn(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0 for action in self.actions}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {action: 0 for action in self.actions}

        old_value = self.q_table[state_key][action]
        future_reward = max(self.q_table[next_state_key].values(), default=0)
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * future_reward - old_value)
        self.q_table[state_key][action] = new_value

        self.exploration_rate *= self.exploration_decay  # Decay exploration rate

    def get_blocking_action(self, state, valid_actions):
        for action in valid_actions:
            x, y = action
            # Simulate the player's move
            temp_state = state.copy()
            temp_state[x * self.board_size + y] = 1  # Assume player 1 is the opponent
            if GoGame(self.board_size).check_win(temp_state.reshape(self.board_size, self.board_size)):
                return action  # Return this action if it blocks a win
        return None  # No blocking action found


# Step 3: Training the Agent
def train_agent(episodes=1000):
    game = GoGame()
    agent = QLearningAgent(actions=game.get_valid_actions(), board_size=game.board_size)

    for episode in range(episodes):
        state = game.reset()
        done = False

        while not done:
            valid_actions = game.get_valid_actions()

            # Check for blocking action before choosing the action
            blocking_action = agent.get_blocking_action(state, valid_actions)
            if blocking_action:
                action = blocking_action
            else:
                action = agent.choose_action(state, valid_actions)

            try:
                next_state, reward, done = game.step(action)
                agent.learn(state, action, reward, next_state)
                state = next_state
            except ValueError as e:
                continue

        if episode % 100 == 0:
            print(f"Episode {episode}: Exploration rate {agent.exploration_rate}")

    return agent

# Step 4: Playing the Game with Tkinter UI
class GoGameUI:
    def __init__(self, agent):
        self.agent = agent
        self.game = GoGame()
        self.root = tk.Tk()
        self.root.title("Go Game")

        self.buttons = [[None for _ in range(self.game.board_size)] for _ in range(self.game.board_size)]
        for i in range(self.game.board_size):
            for j in range(self.game.board_size):
                button = tk.Button(self.root, text="", width=5, height=2, command=lambda x=i, y=j: self.player_move(x, y))
                button.grid(row=i, column=j)
                self.buttons[i][j] = button

        self.reset_button = tk.Button(self.root, text="Reset Game", command=self.reset_game)
        self.reset_button.grid(row=self.game.board_size, columnspan=self.game.board_size)

        self.root.mainloop()

    def player_move(self, x, y):
        try:
            self.game.step((x, y))
            self.update_button(x, y, "X")  # Player symbol
            if self.game.check_win(self.game.board):  # Check if the player has won
                messagebox.showinfo("Game Over", "Player wins!")
                self.reset_game()
                return

            # Agent's turn
            state = self.game.board.flatten()
            valid_actions = self.game.get_valid_actions()
            blocking_action = self.agent.get_blocking_action(state, valid_actions)
            action = blocking_action if blocking_action else self.agent.choose_action(state, valid_actions)

            self.game.step(action)
            self.update_button(action[0], action[1], "O")  # Agent symbol

            if self.game.check_win(self.game.board):  # Check if the agent has won
                messagebox.showinfo("Game Over", "Agent wins!")
                self.reset_game()

        except ValueError as e:
            messagebox.showwarning("Invalid Move", str(e))

    def update_button(self, x, y, text):
        self.buttons[x][y].config(text=text, state=tk.DISABLED)

    def reset_game(self):
        self.game.reset()
        for i in range(self.game.board_size):
            for j in range(self.game.board_size):
                self.buttons[i][j].config(text="", state=tk.NORMAL)

# Train the agent
trained_agent = train_agent(episodes=1000)

# Start the game UI
GoGameUI(trained_agent)
