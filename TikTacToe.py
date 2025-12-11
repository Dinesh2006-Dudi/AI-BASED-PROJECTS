import tkinter as tk
from tkinter import messagebox

class TicTacToe:
    def __init__(self):
        self.current_player = "X"
        self.board = [["", "", ""], ["", "", ""], ["", "", ""]]

        self.window = tk.Tk()
        self.window.title("Tic Tac Toe")

        self.buttonsGrid = []
        for i in range(3):
            row = []
            for j in range(3):
                button = tk.Button(self.window, text="", width=20, height=10,command=lambda i=i, j=j: self.on_button_click(i, j))
                button.grid(row=i, column=j)
                row.append(button)
            self.buttonsGrid.append(row)

    def on_button_click(self, i, j):
        if self.board[i][j] == "" :
            self.board[i][j] = self.current_player
            self.buttonsGrid[i][j].config(text=self.current_player)
            if self.check_winner(self.current_player):
                messagebox.showinfo("Game Over", f"Player {self.current_player} wins!")
                self.window.destroy()
            elif self.is_draw():
                messagebox.showinfo("Game Over", "It's a draw!")
                self.window.destroy()
              
            self.current_player = "O" if self.current_player == "X" else "X"

    def check_winner(self, player):
        for i in range(3):
            if player==self.board[i][0] == self.board[i][1] == self.board[i][2]:
                return True
            if player==self.board[0][i] == self.board[1][i] == self.board[2][i]:
                return True
            if player==self.board[0][0] == self.board[1][1] == self.board[2][2]:    
                return True
            if player==self.board[0][2] == self.board[1][1] == self.board[2][0]:
                return True   
            
        return False
    def is_draw(self):
        for row in self.board:
            if "" in row:
                return False
        return True

    def run(self):
        self.window.mainloop()


game = TicTacToe()
game.run()
