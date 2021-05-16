import csv
import math

import numpy as np

from collections import defaultdict

def entropy_kontoyiannis(sequence):
    if not sequence:
        return 0.0    
    lambdas = 0
    n = len(sequence)
    for i in range(n):
        current_sequence = ''.join(sequence[0:i])
        match = True
        k = i
        while match and k < n:
            k += 1
            match = ''.join(sequence[i:k]) in current_sequence
        lambdas += (k - i)
    return (1.0 * len(sequence) / lambdas) * np.log2(len(sequence))

def max_predictability(S, N):
    if S == 0.0 or N <= 1:
        return 1.0
    for p in np.arange(0.0001, 1.0000, 0.0001):
        h = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        pi_max = h + (1 - p) * math.log2(N - 1) - S
        if pi_max <= 0.001:
            return round(p, 5)
    return 0.0

def pred(x):
    ent = entropy_kontoyiannis(x)
    return max_predictability(ent, len(set(x)))

# Used data structures.
genre_per_itm = {}
user_views_genre = defaultdict(list)

# Populate genre_per_itm.
with open('data/ml-100k/u.item', encoding = "ISO-8859-1") as itm_file:
    for itm in itm_file.readlines():
        itm_genres = []
        chars = itm.split('|')
        iid = int(chars[0])
        for i in range(6, 24):
            if chars[i]=='1':
                itm_genres.append(i-5)
        genre_per_itm[iid] = itm_genres

# Populate user_view_by_genre.
with open('data/ml-100k/train.csv') as train_file:
    for row in csv.reader(train_file, delimiter='\t'):
        for genre in genre_per_itm[int(row[1])]:
            user_views_genre[int(row[0])].append(str(genre))

# Create uid x pred str to save.
out_str = ""
user_prevs = []
max_prev, min_prev = -1, 1
for uid in range(1, 944):
    prev = pred(user_views_genre[uid])
    if prev>0.3:
        user_prevs.append((uid, prev))
        min_prev = min(min_prev, prev)
        max_prev = max(max_prev, prev)
    else:
        user_prevs.append((uid, 0.5))

# Normalize preds.
for i in range(len(user_prevs)):
    u = user_prevs[i][0]
    p = user_prevs[i][1]
    p2 = (p-min_prev)/(max_prev-min_prev)
    user_prevs[i] = (u, p2)
    out_str += "{}\t{}\n".format(u, p2)

print(out_str)

# with open("data/digested/ml-100k/user_vs_user_pred.csv", 'w') as pred_file:
#     pred_file.write(out_str)