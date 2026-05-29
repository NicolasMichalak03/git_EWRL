import os
import hashlib
import numpy as np
import pymc as pm 
import pymc.math as pmm 
import matplotlib.pyplot as plt
import time
import json
import sys
import copy
from tqdm import tqdm

from joblib import Parallel, delayed

from mdp_jax import backward_induction
from tools import truncated_normal, make_json_serializable
from env import Individu, SimulateurMultiIndividusGlobal
from politiques import PolitiqueMCMCpymc,PolitiqueMCMCGibs, PolitiqueOracleOnline, Politiquefrequentistescipy


parametre_connu = {'sigma': 0.5 , 'cout': [0,200] ,'x_max': 30 , 
                   'x_min' : -30 ,'ecart_de_pas': 1, "horizon":10}
parametre_cacher = {'alpha': [0,0.8]}
politiques_dic = {"Oracle": lambda seed : PolitiqueOracleOnline(parametre_connu),
                  "Gibs" : lambda seed : PolitiqueMCMCGibs(parametre_connu, seed = seed),
                  "freq" : lambda seed :Politiquefrequentistescipy(parametre_connu)}

rng_phi = np.random.RandomState(42)  # générateur local

def random_phi():
  return rng_phi.uniform(-0.5, 1)

calendrier = {}

# cles = [120, 140, 160, 180, 200]
cles = [50]

ind = 2  # car ind_1 est pour la clé 1

for cle in cles:
    calendrier[cle] = []
    for _ in range(2):      # 2 individus par clé
        calendrier[cle].append((f"ind_{ind}", {'phi': random_phi()}))

        ind += 1
# print(calendrier[120])
# print(calendrier[140])
# Ajouts particulier
calendrier[1] = [("ind_1", {'phi': 0.8})]
calendrier[150] = [("ind_22", {'phi': 0.8})]


calendrier = {1:[("ind_1", {'phi': 0.8})]}

historique_offline = {"XT+1" : [], "XT" : [], "XT-1" : [], "patient" :[], "action" : [],"temps":[]}

def run_one_simulation(seed,calendrier,historique_offline):
    
    politique = {key: politiques_dic[key](seed) for key in politiques_dic}
   
    simu = SimulateurMultiIndividusGlobal(
        calendrier_arrivees=calendrier,
        politiques=politique,
        parametre_connu=parametre_connu,
        parametre_cacher=parametre_cacher,
        historique_offline=historique_offline,
        globalseed=seed
    )
    simu.run(t_max=parametre_connu["horizon"])
    hist = simu.get_Historique()
    filename = f"resultats/historique_run_{seed}.json"
    with open(filename, "w") as f:
        json.dump(hist, f, indent=2)
    print("Wrote ", filename)
    return hist

def run_parallel_simulations(calendrier, historique_offline, n=2):
    os.makedirs("resultats", exist_ok=True)
    results = Parallel(n_jobs=1)(delayed(lambda seed : run_one_simulation(seed,calendrier,historique_offline)
                                          )(seed) for seed in range(n))
    return results

import time
a = time.time()
# run_parallel_simulations(calendrier=calendrier,historique_offline=historique_offline,n=1)
run_one_simulation(0,calendrier,historique_offline)
print(time.time()-a)
