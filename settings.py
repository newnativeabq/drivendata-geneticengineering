"""
Settings. 
    Reads from .env or .ini file to set up training and inference jobs.
"""

from decouple import config
from models import ModelRegistry

import os

class Config():
    def __init__(self, train=True):

        self.debug_print()

        self.train = train

        self.DEBUG = config('DEBUG', default=True, cast=bool)

        self.ROOT_DIR = self._build_path(config('ROOT_DIR', default=''))

        self.MODEL_SAVE_PATH = self._build_path(config('MODEL_SAVE_PATH', default=''))

        self.MODEL_CHECKPOINT_DIR = self._build_path(config('MODEL_CHECKPOINT_DIR', default=''))

        self.NUM_EPOCHS = config('NUM_EPOCHS', default=1, cast=int)


        # Model Selection
        self.MODEL_NAME = config('MODEL_NAME', default=None)
        self.MODEL = self._get_model(model_name=self.MODEL_NAME)


        # Inference Setup
        self.MODEL_WEIGHTS_PATH = self._build_path(config('MODEL_WEIGHTS_PATH', default=''))


        # Dataset Setup
        self.DATASET_PATH = self._build_path(config('DATASET_PATH'))




    def _get_model(self, model_name):
        if self.train:
            pass 
        if model_name is not None:
            return ModelRegistry[model_name]




    def _build_path(self, *args):
        
        def is_dir_or_file(path):
            return os.path.isdir(path) or os.path.isfile(path)

        
        path = os.path.join(os.getcwd(), *args)

        if is_dir_or_file(path):
            return path


    def debug_print(self):
        print(f'Models Available: {ModelRegistry}')



    def __getitem__(self, key):
        return getattr(self, key)


    