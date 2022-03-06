#!/Users/hugo/Documents/recomendadores/explanatory_framework/venv/bin/python

import pickle
import numpy as np
import pandas as pd

from collections import defaultdict
from scipy.stats import variation, tstd, entropy

METRICS = ["gini", "variation", "num_ratings", "rating_avg",
           "rating_std", "phi", "avg_phi", "entropy", "entropy_of_consumed_sum",
           "entropy_of_consumed_avg", "proportion_long_tailors"]

class UserFeatures:
    def __init__(self, id, urm, itm_phis, usr_phis, itm_H, usr_H, item_long_tails, user_long_tails, is_user=True):
        self.urm = urm
        self.id = id
        self.is_user = is_user
        self.ratings = []
        self.num_ratings_urm = len(urm.nonzero()[0])
        self.num_itens = urm.getrow(0).shape[1]
        self.num_users = urm.getcol(0).shape[0]
        self.non_zeros = None
        self.long_tailors = None,
        # if is_user:
        self.long_tailors = user_long_tails
        self.non_zeros = urm.getrow(id).nonzero()[1]
        for j in self.non_zeros:
            self.ratings.append(urm[self.id, j])
        # else:
        #     self.long_tailors = item_long_tails
        #     self.non_zeros = urm.getcol(id).nonzero()[0]
        #     for i in self.non_zeros:
        #         self.ratings.append(urm[i, self.id])
        self.item_phis = itm_phis
        self.user_phis = usr_phis
        self.itm_H = itm_H
        self.usr_H = usr_H
    
    def _gini(self, x):
        # Mean absolute difference
        mad = np.abs(np.subtract.outer(x, x)).mean()
        # Relative mean absolute difference
        mean = np.mean(x)
        if mean != 0:
            rmad = mad/np.mean(x)
            # Gini coefficient
            g = 0.5 * rmad
            return g
        else:
            return 0
    
    def gini(self):
        return self._gini(self.ratings)
    
    def phi_id(self):
        phi = 0
        if self.is_user:
            row = self.urm.getrow(self.id).nonzero()[1]
            for itm in row:
                phi += self.item_phis[itm]
            return phi
        # elif is item:
        row = self.urm.getcol(self.id).nonzero()[0]
        for usr in row:
            phi += self.user_phis[usr]
        return phi

    def entropy_consumed(self):
        ent = 0
        if self.is_user:
            row = self.urm.getrow(self.id).nonzero()[1]
            for itm in row:
                ent += self.itm_H[itm]
            return ent
        # elif is item:
        row = self.urm.getcol(self.id).nonzero()[0]
        for usr in row:
            ent += self.user_phis[usr]
        return ent
    
    def long_tail(self):
        num_tails = 0
        total = len(self.non_zeros)
        for i in self.non_zeros:
            if i in self.long_tailors:
                num_tails += 1
        if total != 0:
            return num_tails/total
        else:
            return 0

    def get_metrics(self):
        phi = self.phi_id()
        avg = 0
        avg_phi = 0
        std = tstd(self.ratings)
        gini = self.gini()
        var = variation(self.ratings)
        ent = 0
        long_tail = self.long_tail()
        avg_ent_consumed = 0
        if self.is_user:
            ent = self.usr_H[self.id]
        else:
            ent = self.itm_H[self.id]
        ent_consumed = self.entropy_consumed()

        if len(self.ratings) != 0:
            avg = sum(self.ratings)/len(self.ratings)
            avg_phi = phi/len(self.ratings)
            avg_ent_consumed = ent_consumed/len(self.ratings)
        
        if np.isnan(std):
            std = 0
        
        if np.isnan(gini):
            gini = 0
        
        if np.isnan(var):
            var = 0
        
        metrics_map = {
            "id": self.id+1,
            "gini": gini,
            "variation": var,
            "num_ratings": len(self.ratings),
            "rating_avg": avg,
            "rating_std": std,
            "phi": phi,
            "avg_phi": avg_phi,
            "entropy": ent,
            "entropy_of_consumed_sum": ent_consumed,
            "entropy_of_consumed_avg": avg_ent_consumed,
            "proportion_long_tailors": long_tail
        }
        return metrics_map



# class UserFeatures(FeatureCalculator):
#     def __init__(self, uid, urm, itm_phis, usr_phis, itm_H, usr_H, item_long_tails, user_long_tails):
#         super().__init__(uid, urm, itm_phis, usr_phis, itm_H, usr_H, item_long_tails, user_long_tails, True)
# 
# class ItemFeatures(FeatureCalculator):
#     def __init__(self, iid, urm, itm_phis, usr_phis, itm_H, usr_H, item_long_tails, user_long_tails):
#         super().__init__(iid, urm, itm_phis, usr_phis, itm_H, usr_H, item_long_tails, user_long_tails, False)

def get_phis(urm):
    num_ratings_urm = len(urm.nonzero()[0])
    num_itens = urm.getrow(0).shape[1]
    num_users = urm.getcol(0).shape[0]
    item_phis, user_phis = defaultdict(int), defaultdict(int)
    
    for i in range(num_itens):
        item_phis[i] = len(urm.getcol(i).nonzero()[0])/num_ratings_urm
    for u in range(num_users):
        user_phis[u] = len(urm.getrow(u).nonzero()[1])/num_ratings_urm
    
    return item_phis, user_phis

def get_entropies(urm):
    def entropy_(x):
        pd_series = pd.Series(x)
        counts = pd_series.value_counts()
        return entropy(counts)
    
    num_itens = urm.getrow(0).shape[1]
    num_users = urm.getcol(0).shape[0]
    item_entropies, user_entropies = defaultdict(float), defaultdict(float)  

    for itm in range(num_itens):
        x = []
        for usr in urm.getcol(itm).nonzero()[0]:
            x.append(urm[usr, itm])
        item_entropies[itm] = entropy_(x)
    for usr in range(num_users):
        x = []
        for itm in urm.getrow(usr).nonzero()[1]:
            x.append(urm[usr, itm])
        user_entropies[usr] = entropy_(x)

    return item_entropies, user_entropies

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
    
    distribution = []
    for i in range(U):
        distribution.append((urm.getrow(i).count_nonzero(), i))
    distribution.sort(key=lambda x: x[0])
    user_long_tails = set()
    views, i = 0, 0
    while views/R < 0.2:
        user_long_tails.add(distribution[i][1])
        views += distribution[i][0]
        i += 1
    return item_long_tails, user_long_tails


# DS = "ml-100k"
DS = "ml-1M"
urm = pickle.load(open(f"../datasets/{DS}/urm.pk", 'rb'))

item_phis, user_phis = get_phis(urm)
item_entropies, user_entropies = get_entropies(urm)
item_long_tails, user_long_tails = get_long_tailors(urm)

num_users = urm.getcol(0).shape[0]
usr_out_str = "uid, " + ", ".join(METRICS) + '\n'
for usr in range(num_users):
    uf = UserFeatures(usr, urm, item_phis, user_phis, item_entropies, user_entropies, item_long_tails, user_long_tails)
    m = uf.get_metrics()
    usr_out_str += f'{m["id"]}, ' + ", ".join([f'{m[metric]:.5f}' for metric in METRICS]) + "\n"

# print(usr_out_str)
with open(f"../datasets/{DS}/user_metrics.csv", 'w') as metrics_file:
    metrics_file.write(usr_out_str)

# num_itens = urm.getrow(0).shape[1]
# itm_out_str = "iid, " + ", ".join(METRICS) + '\n'
# for itm in range(num_itens):
#     uf = ItemFeatures(itm, urm, item_phis, user_phis, item_entropies, user_entropies, item_long_tails, user_long_tails)
#     m = uf.get_metrics()
#     itm_out_str += f'{m["id"]}, ' + ", ".join([f'{m[metric]:.5f}' for metric in METRICS]) + "\n"

# # print(itm_out_str)
# with open(f"../datasets/{DS}/item_metrics.csv", 'w') as metrics_file:
#     metrics_file.write(itm_out_str)