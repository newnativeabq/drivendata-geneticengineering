from utils.registry import Registry
from .logistic import *
from .rforest import *

ModelRegistry = Registry('ModelRegistry')

ModelRegistry.register(BASELINELOGISTIC)
ModelRegistry.register(RANDOMFOREST)
