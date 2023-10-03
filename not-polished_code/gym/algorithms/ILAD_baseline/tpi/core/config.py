#!/usr/bin/env python3

"""Configuration file."""

import os

from yacs.config import CfgNode as CN


# Global config object
_C = CN()
cfg = _C

# General options
_C.ENV_NAME = 'relocate-v0'
_C.DEMO_FILE = ''
_C.NUM_ITER = 100
_C.NUM_TRAJ = 200
_C.DENSE_REWARD = True
_C.EVAL_RS = 10

# BC options
_C.BC_INIT = True

# IMPERFECT DEMONSTRATIONS
_C.NOISE_LEVEL = 0.0

# DENSITY OF OBJECT
_C.DENSITY = 1000

# ABLATION PARAMETERS
_C.DEMO_RATIO = 180

# Policy options
_C.POLICY_WS = [64, 64]
_C.POLICY_INIT_LOG_STD = 0.0
_C.POLICY_LEARN_LOG_STD = True
_C.POLICY_MIN_LOG_STD = -3.0

# Scratch options
_C.SCRATCH_ADV_W = 1.0

# DAPG options
_C.USE_DAPG = False
_C.DAPG_LAM0 = 1.0
_C.DAPG_LAM1 = 0.95
_C.DAPG_ADV_W = 1e-2


# LFA options
_C.USE_LFA = False
_C.LFA_LAM0 = 1.0
_C.LFA_LAM1 = 0.95
_C.LFA_LAM2 = 0.99
_C.LFA_LAM3 = 0.001
_C.LFA_ADV_W = 1e-2
_C.LFA_POLICY = False
_C.LFA_EVAL_ADV = False
_C.DAPG_BASELINE = False
_C.RANDOM_EMBEDDING = False
_C.BC_FINETUNE = False
_C.PRIORITY = False
_C.FT_BATCHSIZE = 2048
_C.FT_LR = 3e-4
_C.FT_INTERVAL = 1
_C.EMB_FILE = None

# SDAPG options
_C.USE_SDAPG = False
_C.SDAPG_DIST = 'nn'
_C.SDAPG_TYPE = 'mul'
_C.SDAPG_LAM0 = 1.0
_C.SDAPG_LAM1 = 1.0
_C.SDAPG_ADV_W = 1.0
_C.SDAPG_OBS_L = 0
_C.SDAPG_OBS_R = 39



# AFF options
_C.USE_AFF = False
_C.AFF_DIST = 'affordance'
_C.AFF_TYPE = 'mul'
_C.AFF_LAM0 = 1.0
_C.AFF_LAM1 = 1.0
_C.AFF_ADV_W = 1.0
_C.AFF_OBS_L = 0
_C.AFF_OBS_R = 39



# Offline inverse dynamics model options
_C.INVDYN_WS = [64, 64]
_C.INVDYN_TRAIN_DATA = ''
_C.INVDYN_NUM_EP = 5
_C.INVDYN_MB_SIZE = 32
_C.INVDYN_LR = 1e-3
_C.INVDYN_TEST_MODEL = ''
_C.INVDYN_TEST_DATA = ''
_C.INVDYN_GEN_POLICY = ''
_C.INVDYN_GEN_MODE = ''
_C.INVDYN_GEN_NUM_TRAJ = 5

# Online inverse dynamics model options
_C.USE_INVDYN_ONPG = False
_C.INVDYN_ONPG_MLP_W = 64
_C.INVDYN_ONPG_INC_D = False
_C.INVDYN_ONPG_NUM_ITER = 100
_C.INVDYN_ONPG_MB_SIZE = 32
_C.INVDYN_ONPG_LR = 1e-3
_C.INVDYN_ONPG_WD = 0.0
_C.INVDYN_ONPG_LAM0 = 1.0
_C.INVDYN_ONPG_LAM1 = 1.0
_C.INVDYN_ONPG_ADV_W = 1.0
_C.INVDYN_ONPG_WU_ITER = 0
_C.INVDYN_ONPG_TEST = False
_C.INVDYN_ONPG_TEST_N = 500
_C.INVDYN_ONPG_VAL = False
_C.INVDYN_ONPG_VAL_N = 40
_C.INVDYN_ONPG_NORM = False
_C.INVDYN_ONPG_SUBSET = False
_C.INVDYN_ONPG_DEBUG_STATS = False
_C.INVDYN_ONPG_DUMP_INIT = False
_C.INVDYN_ONPG_AGG = False
_C.INVDYN_ONPG_RBS = 1000000
_C.INVDYN_ONPG_ENS_N = 1
_C.INVDYN_ONPG_TT = False
_C.INVDYN_ONPG_FREEZE = False
_C.INVDYN_ONPG_LONG_N = 0
_C.INVDYN_ONPG_LONG_N_MUL = 5
_C.INVDYN_ONPG_ACT_SUBSET = False

# Offline inverse dynamics model options
_C.USE_INVDYN_OFPG = False
_C.INVDYN_OFPG_WS = [64, 64]
_C.INVDYN_OFPG_MODEL = ''
_C.INVDYN_OFPG_NORM = False
_C.INVDYN_OFPG_TEST = False
_C.INVDYN_OFPG_LAM0 = 1.0
_C.INVDYN_OFPG_LAM1 = 1.0
_C.INVDYN_OFPG_ADV_W = 1.0

# Offline density model options
_C.DENSITY_WS = [64, 64]
_C.DENSITY_TRAIN_POS = ''
_C.DENSITY_TRAIN_NEG = ''
_C.DENSITY_TRAIN_POS_FRAC = 0.5
_C.DENSITY_NUM_ITER = 5
_C.DENSITY_MB_SIZE = 32
_C.DENSITY_LR = 1e-3
_C.DENSITY_GEN_POLICY = ''
_C.DENSITY_GEN_MODE = ''
_C.DENSITY_GEN_NUM_TRAJ = 5
_C.USE_DENSITY_OFPG = False
_C.DENSITY_OFPG_MODEL = ''
_C.DENSITY_OFPG_LAM0 = 1.0
_C.DENSITY_OFPG_LAM1 = 1.0
_C.DENSITY_OFPG_ADV_W = 1.0

# Online density model options
_C.USE_DENSITY_ONPG = False
_C.DENSITY_ONPG_WS = [64, 64]
_C.DENSITY_ONPG_POS_FRAC = 0.5
_C.DENSITY_ONPG_NUM_ITER = 100
_C.DENSITY_ONPG_MB_SIZE = 32
_C.DENSITY_ONPG_LR = 1e-3
_C.DENSITY_ONPG_LAM0 = 1.0
_C.DENSITY_ONPG_LAM1 = 1.0
_C.DENSITY_ONPG_ADV_W = 1.0

# SOIL
_C.SOIL = CN()
_C.SOIL.ENABLED = False
_C.SOIL.MLP_W = 64
_C.SOIL.NUM_ITER = 500
_C.SOIL.MB_SIZE = 32
_C.SOIL.LR = 1e-3
_C.SOIL.WD = 0.0
_C.SOIL.RBS = 1000000
_C.SOIL.LAM0 = 0.1
_C.SOIL.LAM1 = 0.99
_C.SOIL.ADV_W = 0.1
_C.SOIL.CHECKPOINT = ''


# GASIL
_C.GASIL = CN()
_C.GASIL.ENABLED = False
_C.GASIL.MLP_W = 64
_C.GASIL.NUM_ITER = 500
_C.GASIL.MB_SIZE = 32
_C.GASIL.LR = 1e-3
_C.GASIL.WD = 0.0
_C.GASIL.RBS = 1000000
_C.GASIL.LAM0 = 0.1
_C.GASIL.LAM1 = 0.99
_C.GASIL.ADV_W = 0.1
_C.GASIL.CHECKPOINT = ''

_C.GASIL_ONPG_WS = [64, 64]
_C.GASIL_ONPG_POS_FRAC = 0.5
_C.GASIL_ONPG_NUM_ITER = 500
_C.GASIL_ONPG_MB_SIZE = 32
_C.GASIL_ONPG_LR = 1e-3
_C.GASIL_ONPG_LAM0 = 1.0
_C.GASIL_ONPG_LAM1 = 1.0
_C.GASIL_ONPG_ADV_W = 1.0
_C.GASIL_OBS_L = 0
_C.GASIL_OBS_R = 39
_C.GASIL_TYPE = 'dtw'

# SL
_C.SL = CN()
_C.SL.ENABLED = False
_C.SL.MLP_W = 64
_C.SL.NUM_ITER = 500
_C.SL.MB_SIZE = 32
_C.SL.LR = 1e-3
_C.SL.WD = 0.0
_C.SL.RBS = 1000000
_C.SL.LAM0 = 0.1
_C.SL.LAM1 = 0.99
_C.SL.ADV_W = 0.1
_C.SL.CHECKPOINT = ''
_C.SL.UPDATE_ITV = 5

# TRPO
_C.TRPO = CN()
_C.TRPO.ENABLED = False
_C.TRPO.MLP_W = 64
_C.TRPO.NUM_ITER = 500
_C.TRPO.MB_SIZE = 32
_C.TRPO.LR = 1e-3
_C.TRPO.WD = 0.0
_C.TRPO.RBS = 1000000
_C.TRPO.LAM0 = 0.1
_C.TRPO.LAM1 = 0.99
_C.TRPO.ADV_W = 0.1
_C.TRPO.CHECKPOINT = ''
_C.TRPO.UPDATE_ITV = 5


# SPPO
_C.SPPO = CN()
_C.SPPO.ENABLED = False
_C.SPPO.MLP_W = 64
_C.SPPO.NUM_ITER = 500
_C.SPPO.MB_SIZE = 32
_C.SPPO.LR = 1e-3
_C.SPPO.WD = 0.0
_C.SPPO.RBS = 1000000
_C.SPPO.LAM0 = 0.1
_C.SPPO.LAM1 = 0.99
_C.SPPO.ADV_W = 0.1
_C.SPPO.CHECKPOINT = ''


# GAIL
_C.GAIL = CN()
_C.GAIL.ENABLED = False
_C.GAIL.MLP_W = 64
_C.GAIL.NUM_ITER = 500
_C.GAIL.MB_SIZE = 32
_C.GAIL.LR = 1e-3
_C.GAIL.WD = 0.0
_C.GAIL.RBS = 1000000
_C.GAIL.LAM0 = 0.1
_C.GAIL.LAM1 = 0.99
_C.GAIL.ADV_W = 0.1
_C.GAIL.CHECKPOINT = ''


# GAIL_TRPO
_C.GAIL_TRPO = CN()
_C.GAIL_TRPO.ENABLED = False
_C.GAIL_TRPO.MLP_W = 64
_C.GAIL_TRPO.NUM_ITER = 500
_C.GAIL_TRPO.MB_SIZE = 32
_C.GAIL_TRPO.LR = 1e-3
_C.GAIL_TRPO.WD = 0.0
_C.GAIL_TRPO.RBS = 1000000
_C.GAIL_TRPO.LAM0 = 0.1
_C.GAIL_TRPO.LAM1 = 0.99
_C.GAIL_TRPO.ADV_W = 0.1
_C.GAIL_TRPO.CHECKPOINT = ''

# Demo matching options
_C.MATCH_DIST = 'h_dtw'
_C.MATCH_POLICY = ''
_C.MATCH_NUM_TRAJ = 5
_C.MATCH_DATA = ''

# Custom object
_C.CUSTOM_OBJECT = False
_C.CUSTOM_OBJECT_TYPE = 'box'

# Custom mass
_C.CUSTOM_MASS = False
_C.CUSTOM_MASS_MUL = 1.0

# Custom friction
_C.CUSTOM_FRICT = False
_C.CUSTOM_FRICT_MUL = 1.0

# Custom object mass
_C.CUSTOM_OBJ_MASS = False
_C.CUSTOM_OBJ_MASS_MUL = 1.0

# Custom object size
_C.CUSTOM_OBJ_SIZE = False
_C.CUSTOM_OBJ_SIZE_MUL = 1.0

# Custom fingers
_C.CUSTOM_FINGERS = False
_C.CUSTOM_FINGERS_MASK = '11111'

# Checkpoints
_C.CHECKPOINT_POLICY = ''
_C.CHECKPOINT_BASELINE = ''
_C.CHECKPOINT_INVDYN = []

# Sampler options
_C.SAMPLER_INIT_STATE = False

# Misc options
_C.RNG_SEED = 100
_C.NUM_CPU = 5
_C.SAVE_FREQ = 25
_C.JOB_DIR = ''
_C.JOB_NAME = 'relocate_scratch'


def assert_cfg():
    assert not (_C.BC_INIT or _C.USE_DAPG) or os.path.exists(cfg.DEMO_FILE), \
        'Demo file not found: {}'.format(cfg.DEMO_FILE)
    assert (_C.SDAPG_OBS_L == 0) or _C.SDAPG_DIST in ['h_dtw', 'mp_dtw'], \
        'Using obs left ind and {} dist not supported'.format(_C.SDAPG_DIST)
    assert (_C.SDAPG_OBS_R == 39) or _C.SDAPG_DIST in ['h_dtw', 'mp_dtw'], \
        'Using obs right ind and {} dist not supported'.format(_C.SDAPG_DIST)
    assert 0.0 <= _C.DENSITY_ONPG_POS_FRAC <= 1.0, \
        'Positives fraction must be in [0, 1]'
    assert not _C.CUSTOM_OBJECT or _C.ENV_NAME == 'relocate-v0', \
        'Using custom objects supported only with relocate'
    assert not _C.CUSTOM_MASS or _C.ENV_NAME == 'relocate-v0', \
        'Using custom mass supported only with relocate'
    assert not _C.CUSTOM_FRICT or _C.ENV_NAME == 'relocate-v0', \
        'Using custom friction supported only with relocate'
    assert not _C.CUSTOM_OBJ_MASS or _C.ENV_NAME == 'relocate-v0', \
        'Using custom object mass supported only with relocate'
    assert not _C.CUSTOM_OBJ_SIZE or _C.ENV_NAME == 'relocate-v0', \
        'Using custom object size supported only with relocate'
    assert not _C.CUSTOM_FINGERS or _C.ENV_NAME == 'relocate-v0', \
        'Using custom fingers supported only with relocate'
    assert not _C.INVDYN_ONPG_ACT_SUBSET or _C.CUSTOM_FINGERS, \
        'Using action subset expects custom fingers'
    assert not _C.INVDYN_ONPG_INC_D, 'Deprecated config option'
