# ----------------------------------------------------------------------------
# Created By  : Bortch - JBS
# Created Date: 09/01/2021
# version ='1.0'
# source = https://github.com/bortch/second_hand_UK_car_challenge
# modification for kaggle
# ---------------------------------------------------------------------------
from os.path import join


INPUT_PATH = "/kaggle/input"
OUTPUT_PATH = "./"
ORIGINAL_DATASET_PATH = join(INPUT_PATH,"used-car-dataset-ford-and-mercedes")
PROCESSED_FILES_PATH = join(OUTPUT_PATH,'processed_files')
PREPARED_DATASET_PATH = join(OUTPUT_PATH,"prepared_dataset")
BS_LIB_PATH = join(INPUT_PATH,"bs-lib")
MODEL_DIR_PATH = join(OUTPUT_PATH,'model')