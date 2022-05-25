import argparse
import glob
import os
import shutil
import time
import random
import numpy as np
import math
import sys
import datetime
sys.path.append('..')
from utils import *
from NeuralNet import NeuralNet
from keras.callbacks import *

import argparse

from .DufferNNet import DufferNNet as dnnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 100,
    'batch_size': 2048,
    'cuda': False,
    'num_channels': 32,
    'validation_split': 0.1,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = dnnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        earlyStoppingCallback = EarlyStopping(monitor='val_loss', patience=3)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorBoardCallback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        return self.nnet.model.fit(
            x = input_boards,
            y = [target_pis, target_vs],
            batch_size = args.batch_size,
            epochs = args.epochs,
            callbacks=[earlyStoppingCallback, tensorBoardCallback],
            validation_split=args.validation_split
        )

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(board, verbose=0)

        print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        # Using glob to support the ``.data-00000-of-00001`` suffix in checkpoint files when running in a single machine
        if len(glob.glob(filepath + '.*')) == 0:
        # if not os.path.exists(filepath + '.*'):
            raise ValueError("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)
