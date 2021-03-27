# =============================================================================
# 一斉に読み込む
# =============================================================================
from deepshima.core import Variable
from deepshima.core import Parameter
from deepshima.core import Function
from deepshima.core import using_config
from deepshima.core import no_grad
from deepshima.core import test_mode
from deepshima.core import as_array
from deepshima.core import as_variable
from deepshima.core import setup_variable
from deepshima.core import Config
from deepshima.layers import Layer
from deepshima.models import Model
from deepshima.datasets import Dataset
from deepshima.dataloaders import DataLoader
# from deepshima.dataloaders import SeqDataLoader

import deepshima.datasets
import deepshima.dataloaders
import deepshima.optimizers
import deepshima.functions
import deepshima.functions_conv
import deepshima.layers
import deepshima.utils
import deepshima.cuda
import deepshima.transforms

setup_variable()
