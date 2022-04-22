#!/Users/hugo/Documents/recomendadores/explanatory_framework/venv/bin/python3

from collections import defaultdict
import pickle
import random
from typing import ClassVar
import numpy as np
import pandas as pd

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

# DS = "ml-1M"
# U = 6040
# I = 3952
# R = 1000209
# ml_ds = movielens.load_feedback(variant="1M")

# users, itens, ratings = [], [], []

# csv_string = "user, item, rating\n"
# for (u, i, r) in ml_ds:
#     users.append(int(u)-1)
#     itens.append(int(i)-1)
#     ratings.append(float(r))
#     csv_string += f'{u}, {i}, {r}\n'

# URM = csr_matrix((ratings, (users, itens)), shape=(U, I))

# pickle.dump(URM, open(f"{DS}/urm.pk", "wb" ), pickle.HIGHEST_PROTOCOL)
# with open(f"{DS}/uir.csv", 'w') as csv_file:
#     csv_file.write(csv_string)

####################################
# Criação de subsamples do dataset #
####################################

# nu = 0 # numero de usuarios do urm_n
# ni = 0 # numero de itens do urm_n
# nr = 0 # numero de ratings do urm_n
# tu = 10 # constraint no numero medio de ratings por usuario
# ti = 70000 # constraint no numero maximo de itens
# N = 600 # numero de subsets
# n = 1

# while(n <= N):
#     nu = random.randint(100, U)
#     ni = random.randint(100, I)
#     chosen_users = random.sample(range(0, U), nu)
#     chosen_users.sort()
#     chosen_itens = random.sample(range(0, I), ni)
#     chosen_itens.sort()
#     urm_n = URM[chosen_users, :][:, chosen_itens]
#     nr = urm_n.count_nonzero()
#     if nr/nu > tu and ni < ti:
#         csv_name = "%03d_UIR.csv" % n
#         csv_name = DS + "/UIR/" + csv_name
#         pickle_name = "%03d_URM.pk" % n
#         pickle_name = DS + "/URM/" + pickle_name
#         csv_string = "user, item, rating\n"
#         for u, i in zip(*urm_n.nonzero()):
#             csv_string += f'{chosen_users[u]+1}, {chosen_itens[i]+1}, {urm_n[u, i]}\n'
        
#         pickle.dump(urm_n, open(pickle_name, "wb" ), pickle.HIGHEST_PROTOCOL)
#         with open(csv_name, 'w') as csv_file:
#             csv_file.write(csv_string)
#         n += 1
#         print(n)


# Load dataset
DS = "ml-10m"
# DS = "ml-1m"
# DS = "ml-100k"

df = pd.read_csv(
    f"../../data/{DS}/ratings.dat",
    engine='python',
    sep="::",
    names=["uid", "iid", "rating", "ts"],
    header = None)

# A overly complex (and unecessary) way of sorting rating by TS.
user_ratings = defaultdict(list)
size = df.shape[0]
row = df.iloc[0]
i = 0
while i < size:
    curr_user = int(row['uid'])
    while(int(row['uid']) == curr_user):
        rating = (int(row['iid']), float(row['rating']), int(row['ts']))
        user_ratings[curr_user].append(rating)
        i += 1
        if (i >= size):
            break
        row = df.iloc[i]
    user_ratings[curr_user].sort(key=lambda tup : tup[2])

train = []
validation = []
test = []

# Divide into training, validation and testing (and writing in files)
for uid in user_ratings:
    ratings = user_ratings[uid]
    num_ratings = len(ratings)
    for i in range(num_ratings//2):
        train.append((int(uid), ratings[i][0], ratings[i][1]))
    for i in range(num_ratings//2, (num_ratings//2 + num_ratings//4)):
        validation.append((int(uid), ratings[i][0], ratings[i][1]))
    for i in range((num_ratings//2 + num_ratings//4), num_ratings):
        test.append((int(uid), ratings[i][0], ratings[i][1]))

train.sort()
validation.sort()
test.sort()

with open(f'TS-split/{DS}/train.csv', 'w') as train_csv:
    train_csv.write(
        "uid, iid, rating\n" + 
        '\n'.join([f"{u}, {i}, {r:.1f}" for (u, i, r) in train]))

with open(f'TS-split/{DS}/validation.csv', 'w') as validation_csv:
    validation_csv.write(
        "uid, iid, rating\n" + 
        '\n'.join([f"{u}, {i}, {r:.1f}" for (u, i, r) in validation]))

with open(f'TS-split/{DS}/test.csv', 'w') as test_csv:
    test_csv.write(
        "uid, iid, rating\n" + 
        '\n'.join([f"{u}, {i}, {r:.1f}" for (u, i, r) in test]))