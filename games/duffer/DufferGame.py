import numpy as np
import sys
sys.path.append('..')
from Game import Game
from duffer import BaseBoard, Board, Move, SIDE

class DufferGame(Game):

    def __init__(self):
        pass

    @classmethod
    def getAction(cls, move: Move) -> int:
        stone, jump1, jump2, jump3 = move.to_id()
        return (729*stone + 81*jump1 + 9*jump2 + jump3)

    def getMove(cls, action) -> Move:
        stone = action // 729
        action = action % 729
        jump1 = action // 81
        action = action % 81
        jump2 = action // 9
        jump3 = action % 9
        return Move.from_id((stone, jump1, jump2, jump3))

    def getInitBoard(self):
        b = Board()
        return np.array(b.array_representation())

    def getBoardSize(self):
        return (SIDE, SIDE)

    def getActionSize(self):
        # 16 stones x
        # 9 directions (1st jump) x
        # 9 directions (2nd jump) x
        # 9 directions (3rd jump)
        return 11664

    def getNextState(self, board, player, action):
        b = Board.from_array(board)
        move = self.getMove(action)
        if not b.push(move):
            raise ValueError("DufferGame: Unable to apply move.")
        return (np.array(b.array_representation()), -player)

    def getValidMoves(self, board, player):
        valids = [0]*self.getActionSize()
        b = Board.from_array(board)
        for move in b.legal_moves():
            valids[self.getAction(move)] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        b = Board.from_array(board)
        return -1 if b.is_over() else 0

    def getCanonicalForm(self, board, player):
        return board

    def getSymmetries(self, board, pi):
        b =  Board.from_array(board)
        board_flip_vertical = b.flip_vertical()
        board_flip_horizontal = b.flip_horizontal()
        board_flip_diagonal_A1D4 = b.flip_diagA1D4()
        board_flip_diagonal_A4D1 = b.flip_diagA4D1()
        board_rotate90 = b.rotate90()
        board_rotate180 = b.rotate180()
        board_rotate270 = b.rotate270()
        pi_flip_vertical = [0]*self.getActionSize()
        pi_flip_horizontal = [0]*self.getActionSize()
        pi_flip_diagonal_A1D4 = [0]*self.getActionSize()
        pi_flip_diagonal_A4D1 = [0]*self.getActionSize()
        pi_rotate90 = [0]*self.getActionSize()
        pi_rotate180 = [0]*self.getActionSize()
        pi_rotate270 = [0]*self.getActionSize()
        for move in b.legal_moves():
            pi_flip_vertical[self.getAction(move.flip_vertical())] = pi[self.getAction(move)]
            pi_flip_horizontal[self.getAction(move.flip_horizontal())] = pi[self.getAction(move)]
            pi_flip_diagonal_A1D4[self.getAction(move.flip_diagA1D4())] = pi[self.getAction(move)]
            pi_flip_diagonal_A4D1[self.getAction(move.flip_diagA4D1())] = pi[self.getAction(move)]
            pi_rotate90[self.getAction(move.rotate90())] = pi[self.getAction(move)]
            pi_rotate180[self.getAction(move.rotate180())] = pi[self.getAction(move)]
            pi_rotate270[self.getAction(move.rotate270())] = pi[self.getAction(move)]
        return [
            (np.array(board), pi),
            (np.array(board_flip_vertical.array_representation()), pi_flip_vertical),
            (np.array(board_flip_horizontal.array_representation()), pi_flip_horizontal),
            (np.array(board_flip_diagonal_A1D4.array_representation()), pi_flip_diagonal_A1D4),
            (np.array(board_flip_diagonal_A4D1.array_representation()), pi_flip_diagonal_A4D1),
            (np.array(board_rotate90.array_representation()), pi_rotate90),
            (np.array(board_rotate180.array_representation()), pi_rotate180),
            (np.array(board_rotate270.array_representation()), pi_rotate270)
        ]

    def display(self, board):
        print(Board.from_array(board))

    def stringRepresentation(self, board):
        b = Board.from_array(board)
        return b.string_representation()
