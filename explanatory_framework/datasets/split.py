#!/Users/hugo/Documents/recomendadores/explanatory_framework/venv/bin/python3

import pickle
import random
from typing import ClassVar
import numpy as np

from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from scipy.sparse import csr_matrix

####################################
#        Loading do dataset        #
####################################

# DS = "ml-100k"
# U = 943
# I = 1682
# R = 100000
# ml_ds = movielens.load_feedback(variant="100K") # array of tuples (U, I, R)

DS = "ml-1M"
U = 6040
I = 3952
R = 1000209
ml_ds = movielens.load_feedback(variant="1M")

users, itens, ratings = [], [], []

for (u, i, r) in ml_ds:
    users.append(int(u)-1)
    itens.append(int(i)-1)
    ratings.append(float(r))

URM = csr_matrix((ratings, (users, itens)), shape=(U, I))


####################################
# Criação de subsamples do dataset #
####################################

nu = 0 # numero de usuarios do urm_n
ni = 0 # numero de itens do urm_n
nr = 0 # numero de ratings do urm_n
tu = 10 # constraint no numero medio de ratings por usuario
ti = 70000 # constraint no numero maximo de itens
N = 600 # numero de subsets
n = 1

while(n <= N):
    nu = random.randint(100, U)
    ni = random.randint(100, I)
    chosen_users = random.sample(range(0, U), nu)
    chosen_users.sort()
    chosen_itens = random.sample(range(0, I), ni)
    chosen_itens.sort()
    urm_n = URM[chosen_users, :][:, chosen_itens]
    nr = urm_n.count_nonzero()
    if nr/nu > tu and ni < ti:
        csv_name = "%03d_UIR.csv" % n
        csv_name = DS + "/UIR/" + csv_name
        pickle_name = "%03d_URM.pk" % n
        pickle_name = DS + "/URM/" + pickle_name
        csv_string = "user, item, rating\n"
        for u, i in zip(*urm_n.nonzero()):
            csv_string += f'{chosen_users[u]+1}, {chosen_itens[i]+1}, {urm_n[u, i]}\n'
        
        pickle.dump(urm_n, open(pickle_name, "wb" ), pickle.HIGHEST_PROTOCOL)
        with open(csv_name, 'w') as csv_file:
            csv_file.write(csv_string)
        n += 1
        print(n)