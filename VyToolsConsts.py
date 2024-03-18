import os

# Constants
# =========
DATA_PATH = 'dataset'
TRAINING_PATH = os.path.join('features', 'data.bin')
DETAILS_PATH = os.path.join('features', 'details.txt')
LATENT_SIZE = 2048 # all data smaller than this size will be upscaled by repeating the smaller size data until it fits.
SAMPLE_RATE = 44100