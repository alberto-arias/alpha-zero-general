import numpy as np
from duffer import BaseBoard, Board, Move, SIDE

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanDufferPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        b = Board.from_array(board)
        # print(b)
        while True:
            input_move = input("Your move: ")
            try:
                move = Move.from_string(input_move)
                if b.is_legal(move):
                    a = self.game.getAction(move)
                    break
                else:
                    print("Illegal move")
            except:
                pass
        return a


class GreedyDufferPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]
