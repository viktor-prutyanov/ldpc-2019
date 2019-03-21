#!/bin/python3

import numpy as np
from gen_tests import get_H

H = get_H(16, 16, 4, 32)
np.savetxt('H2.txt', H, fmt="%i")
