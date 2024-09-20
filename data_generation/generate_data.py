import yaml
import os
import sys
import numpy as np
from argparse import ArgumentParser 
from data_generation_kdv import KdV
from data_generation_bbm import BBM
from data_generation_cahn_hilliard import CahnHill

# import yml param file
with open('params_datagen.yml', 'r') as file:
    p = yaml.safe_load(file)

np.random.seed(p["seed"])

filename = os.path.join(p["folder"], f"{p['filename']}.npz")

parser = ArgumentParser()

parser.add_argument("equation", type=str, help="generate data based off of a given equation name")
args = parser.parse_args()

eq = args.equation.lower()

if eq == "kdv":
    kdv = KdV(p["x_max"], p["x_points"], p["t_max"], p["t_points"], p["gamma"], p["eta"])
    kdv.produce_samples(p["nr_realizations"], filename)
elif eq == "bbm":
    bbm = BBM(p["x_max"], p["x_points"], p["t_max"], p["t_points"], p["gamma"], p["eta"])
    bbm.produce_samples(p["nr_realizations"], filename)
elif eq == "cahnhill":
    cahn = CahnHill(p["x_max"], p["x_points"], p["t_max"], p["t_points"], p["gamma"], p["eta"])
    cahn.produce_samples(p["nr_realizations"], filename)
else:
    print("fatal error: not a valid equation", file=sys.stderr)
