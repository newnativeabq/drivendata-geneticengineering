from utils.registry import Registry
from .logistic import *
from .rforest import *
from .nnet import *

ModelRegistry = Registry('ModelRegistry')

ModelRegistry.register(BASELINELOGISTIC)
ModelRegistry.register(RANDOMFOREST)
ModelRegistry.register(NNET)