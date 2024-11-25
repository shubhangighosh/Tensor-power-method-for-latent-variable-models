# common imports shared across scripts
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random, jit, grad
import numpy as np
from tqdm import tqdm
import time
from scipy.linalg import orth
import matplotlib.pyplot as plt