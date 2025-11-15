"""
Configuration file for the 1D World Model project.

Contains hyperparameters and shared constants for all modules.
"""

# Environment settings
ENV_SEED = 42
ACTION_SPACE = [-1, 0, 1]

# Model dimensions
OBS_DIM = 2
LATENT_DIM = 4
ACTION_DIM = len(ACTION_SPACE)
HIDDEN_DIM_ENCODER = 64
HIDDEN_DIM_WORLD_MODEL = 128

# Data collection
NUM_EPISODES_COLLECT = 400
EPISODE_LENGTH = 50

# Encoder/Decoder training
ENCODER_EPOCHS = 30
ENCODER_BATCH_SIZE = 128
ENCODER_LEARNING_RATE = 1e-3

# World model training
WORLD_MODEL_EPOCHS = 20
WORLD_MODEL_BATCH_SIZE = 128
WORLD_MODEL_LEARNING_RATE = 1e-3

# MPC Controller
MPC_HORIZON = 12
MPC_NUM_SAMPLES = 512
MPC_GAMMA = 0.99

# File paths
ENCODER_WEIGHTS_PATH = "weights/encoder.safetensors"
DECODER_WEIGHTS_PATH = "weights/decoder.safetensors"
WORLD_MODEL_WEIGHTS_PATH = "weights/world_model.safetensors"
DATA_DIR = "data"
