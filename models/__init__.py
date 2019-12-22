import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(curPath)
from vgg import *
from dpn import *
from lenet import *
from senet import *
from pnasnet import *
from densenet import *
from googlenet import *
from shufflenet import *
from shufflenetv2 import *
from resnet import *
from resnext import *
from preact_resnet import *
from mobilenet import *
from mobilenetv2 import *
from myVGG import *
from myAlexNet import *