import sys
import os
os.chdir('../ref/StructuredForests/')
sys.path.append(os.getcwd())

from StructuredForests import *

import matplotlib.pyplot as plt
edge = edge_predict('../../Project1/data/img/14037.jpg')
plt.imshow(edge)
plt.show()