from utils.registry import Registry
from .logistic import *

ModelRegistry = Registry('ModelRegistry')

ModelRegistry.register(BASELINELOGISTIC)
