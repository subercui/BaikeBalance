# -*- coding: utf-8 -*-
"""
Execute a training run of OPDAC

Created on Sun June 5 2016

@author: Subercui

"""
import launcher
import sys

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    '''STEPS_PER_EPOCH = 50000
    EPOCHS = 100
    STEPS_PER_TEST = 10000'''
    STEPS_PER_EPOCH = 5000
    EPOCHS = 20
    STEPS_PER_TEST = 1000

    # ----------------------
    # ALE Parameters
    # ----------------------
    BASE_ROM_PATH = "../roms/"
    #ROM = 'breakout.bin'
    #FRAME_SKIP = 4
    REPEAT_ACTION_PROBABILITY = 0

    # ----------------------
    # Agent/Network parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    BATCH_ACCUMULATOR = 'mean'
    LEARNING_RATE = .0002
    DISCOUNT = .95
    RMS_DECAY = .99 # (Rho)
    RMS_EPSILON = 1e-6
    MOMENTUM = 0
    CLIP_DELTA = 0
    EPSILON_START = 1.0#还没改到
    EPSILON_MIN = .1#还没改到
    EPSILON_DECAY = 1000000#还没改到
    PHI_LENGTH = 1
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    NETWORK_TYPE = "linear"
    FREEZE_INTERVAL = -1
    REPLAY_START_SIZE = 100#还没改到
    RESIZE_METHOD = 'crop'#还没改到
    STATE_WIDTH = 4
    DEATH_ENDS_EPISODE = 'false'#还没改到
    MAX_START_NULLOPS = 0#还没改到
    DETERMINISTIC = True#还没改到
    CUDNN_DETERMINISTIC = False#还没改到

if __name__ == "__main__":
    launcher.launch(sys.argv[1:], Defaults, __doc__)