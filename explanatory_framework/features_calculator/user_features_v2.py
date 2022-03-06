#!/Users/hugo/Documents/recomendadores/explanatory_framework/venv/bin/python

import pickle
import numpy as np
import pandas as pd

from collections import defaultdict
from scipy.stats import variation, tstd, entropy

METRICS = [
    "num_ratings",
    "log_num_ratings",
    "gini",
    "rating_avg",
    "rating_std",
    "phi_avg",
    "phi_std",
    "proportion_long_tailors",
    "avg_entropy",
    "abnormality",
    "abnormality_CR",
    "itens_avg_num_ratings",
    "avg_rating_from_itens_consumed",
]

class UserFeatures:
    def __init__(self, id, urm, itm_phis, itm_H, itm_long_tailors, itm_avg, itm_std, itm_num_rating):
        self.urm = urm
        self.id = id
        self.ratings = []
        self.item_phis = itm_phis
        self.itm_H = itm_H
        self.long_tailors = itm_long_tailors
        self.itm_avg = itm_avg
        self.itm_std = itm_std
        self.itm_num_rating = itm_num_rating
        self.non_zeros = urm.getrow(id).nonzero()[1]
        for j in self.non_zeros:
            self.ratings.append(urm[self.id, j])

    def num_ratings(self):
        return len(self.ratings)
    
    def log_num_ratings(self):
        if len(self.ratings) > 0:
            return np.log2(len(self.ratings))
        return 0
    
    def rating_avg(self):
        if len(self.ratings) > 0:
            return sum(self.ratings)/len(self.ratings)
        return 0
    
    def rating_std(self):
        if len(self.ratings) > 0:
            return tstd(self.ratings)
        return 0

    def gini(self):
        # Mean absolute difference
        mad = np.abs(np.subtract.outer(self.ratings, self.ratings)).mean()
        # Relative mean absolute difference
        mean = np.mean(self.ratings)
        if mean != 0:
            rmad = mad/np.mean(self.ratings)
            # Gini coefficient
            g = 0.5 * rmad
            return g
        else:
            return 0
    
    def phi_avg(self):
        phis = []
        row = self.urm.getrow(self.id).nonzero()[1]
        for itm in row:
            phis.append(self.item_phis[itm])
        if len(phis) > 0:
            avg = sum(phis)/len(phis)
            return avg
        return 0

    def phi_std(self):
        phis = []
        row = self.urm.getrow(self.id).nonzero()[1]
        for itm in row:
            phis.append(self.item_phis[itm])
        if len(phis) > 0:
            std = tstd(phis)
            return std
        return 0

    def avg_entropy(self):
        ent = 0
        row = self.urm.getrow(self.id).nonzero()[1]
        if len(row) > 0:
            for itm in row:
                ent += self.itm_H[itm]
            return ent/len(row)
        return 0
    
    def proportion_long_tailors(self):
        num_tails = 0
        row = self.urm.getrow(self.id).nonzero()[1]
        if len(row) > 0:
            for i in row:
                if i in self.long_tailors:
                    num_tails += 1
            return num_tails/len(row)
        return 0
    
    def abnormality(self):
        abnormality = 0
        row = self.urm.getrow(self.id).nonzero()[1]
        if len(row) > 0:
            for itm in row:
                abnormality += abs(self.urm[self.id, itm] - self.itm_avg[itm])
            return abnormality/len(row)
        return 0
    
    def abnormality_CR(self):
        abnormality_cr = 0
        row = self.urm.getrow(self.id).nonzero()[1]
        stds = [self.itm_std[itm] for itm in row]
        std_min = min(stds)
        std_max = max(stds)
        
        if len(row) > 0:
            for itm in row:
                contr = (self.itm_std[itm] - std_min)/(std_max - std_min)
                abnormality_cr += (abs((self.urm[self.id, itm] - self.itm_avg[itm]) * contr)) ** 2
            return abnormality_cr/len(row)
        return 0
    
    def itens_avg_num_ratings(self):
        num_ratings = 0
        row = self.urm.getrow(self.id).nonzero()[1]
        if len(row) > 0:
            for itm in row:
                num_ratings += self.itm_num_rating[itm]
            return num_ratings/len(row)
        return 0
    
    def avg_rating_from_itens_consumed(self):
        sum_ratings = 0
        row = self.urm.getrow(self.id).nonzero()[1]
        if len(row) > 0:
            for itm in row:
                sum_ratings += self.itm_avg[itm]
            return sum_ratings/len(row)
        return 0

    def get_metrics(self):
        return {
            "id": self.id,
            "num_ratings": self.num_ratings(),
            "log_num_ratings": self.log_num_ratings(),
            "gini": self.gini(),
            "rating_avg": self.rating_avg(),
            "rating_std": self.rating_std(),
            "phi_avg": self.phi_avg(),
            "phi_std": self.phi_std(),
            "proportion_long_tailors": self.proportion_long_tailors(),
            "avg_entropy": self.avg_entropy(),
            "abnormality": self.abnormality(),
            "abnormality_CR": self.abnormality_CR(),
            "itens_avg_num_ratings": self.itens_avg_num_ratings(),
            "avg_rating_from_itens_consumed": self.avg_rating_from_itens_consumed()
        }

def get_phis(urm):
    num_itens = urm.getrow(0).shape[1]
    num_users = urm.getcol(0).shape[0]
    item_phis = defaultdict(int)
    
    for i in range(num_itens):
        item_phis[i] = len(urm.getcol(i).nonzero()[0]) / num_users
    return item_phis

def get_entropies(urm):
    def entropy_(x):
        pd_series = pd.Series(x, dtype='float64')
        counts = pd_series.value_counts()
        return entropy(counts)
    
    num_itens = urm.getrow(0).shape[1]
    item_entropies = defaultdict(float)

    for itm in range(num_itens):
        x = []
        for usr in urm.getcol(itm).nonzero()[0]:
            x.append(urm[usr, itm])
        item_entropies[itm] = entropy_(x)

    return item_entropies

def get_long_tailors(urm):
    (U, I) = urm.get_shape()
    R = urm.count_nonzero()
    distribution = []
    for j in range(I):
        distribution.append((urm.getcol(j).count_nonzero(), j))
    distribution.sort(key=lambda x: x[0])
    item_long_tails = set()
    views, i = 0, 0
    while views/R < 0.2:
        item_long_tails.add(distribution[i][1])
        views += distribution[i][0]
        i += 1
    return item_long_tails

def get_itens_avg_std_and_num_ratings(urm):
    num_itens = urm.getrow(0).shape[1]
    num_users = urm.getcol(0).shape[0]
    itens_avg = defaultdict(float)
    itens_std = defaultdict(float)
    itens_num_rating = defaultdict(int)

    for itm in range(num_itens):
        x = []
        for usr in urm.getcol(itm).nonzero()[0]:
            x.append(urm[usr, itm])
        if len(x) > 0:
            itens_avg[itm] = sum(x)/len(x)
            itens_std[itm] = tstd(x)
            itens_num_rating[itm] = len(x)
        else:
            itens_avg[itm] = 0
            itens_std[itm] = 0
            itens_num_rating[itm] = 0
    
    return itens_avg, itens_std, itens_num_rating


DS = "ml-100k"
# DS = "ml-1M"
urm = pickle.load(open(f"../datasets/{DS}/urm.pk", 'rb'))

print("Calculating Phis...")
item_phis = get_phis(urm)

print("Calculating Entropies...")
item_entropies = get_entropies(urm)

print("Calculating Long Tailors...")
item_long_tails = get_long_tailors(urm)

print("Calculating AVG, STD and NUM...")
item_avg, item_std, item_num_rating = get_itens_avg_std_and_num_ratings(urm)

print("Getting metrics...")
num_users = urm.getcol(0).shape[0]
usr_out_str = "uid, " + ", ".join(METRICS) + '\n'
for usr in range(num_users):
    uf = UserFeatures(
        id = usr,
        urm = urm,
        itm_phis = item_phis,
        itm_H = item_entropies,
        itm_long_tailors = item_long_tails,
        itm_avg = item_avg,
        itm_std = item_std,
        itm_num_rating = item_num_rating
    )
    m = uf.get_metrics()
    usr_out_str += f'{int(m["id"])+1}, ' + ", ".join([f'{m[metric]:.5f}' for metric in METRICS]) + "\n"

# print(usr_out_str)
with open(f"../datasets/{DS}/user_metrics_v2.csv", 'w') as metrics_file:
    metrics_file.write(usr_out_str)