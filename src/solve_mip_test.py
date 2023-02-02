import numpy as np
import pandas as pd
import time
import random
import math
from collections import defaultdict
import math
import os
import json

from pyscipopt import Model

data_path = '/Users/chang/PhD_workplace/MIPcc23/datasets/vary_rhs/series_2/rhs_s2_i01.mps.gz'
soln_path = '/Users/chang/PhD_workplace/MIPcc23/solutions/vary_rhs/series_2/rhs_s2_i01_best.sol'

model = Model("mipcc23")
model.readProblem(data_path)
model.setParam('display/verblevel',3)

cur_soln = model.readSolFile(soln_path)
print(len(model.getSols()))
model.addSol(cur_soln)
print(len(model.getSols()))
model.optimize()