import yaml
import os
import numpy as np
from argparse import ArgumentParser 
from data_generation.data_generation_kdv import KdV

# import yml param file
with open('params_datagen.yml', 'r') as file:
    p = yaml.safe_load(file)

np.random.seed(p["seed"])

filename = os.path.join(p["folder"], p["filename"])

parser = ArgumentParser()
parser.parse_args()

kdv = KdV(p["x_max"], p["x_points"], p["t_max"], p["t_points"], p["gamma"], p["eta"])
kdv.produce_samples(p["nr_realizations"], filename)