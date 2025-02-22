# %%
import math
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img

MAIN = __name__ == "__main__"
# %%
if MAIN:
    arr = np.load(section_dir / "numbers.npy")
# %%
if MAIN:
    print(arr[0].shape)
    display_array_as_img(arr[0])  # plotting the first image in the batch
# %%
if MAIN:
    print("hello world")

# %%
import nbformat
print(nbformat.__version__)
# %%
