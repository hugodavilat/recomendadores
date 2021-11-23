#!/Users/hugo/Documents/recomendadores/explanatory_framework/venv/bin/python

import pickle
import numpy as np

from collections import defaultdict
from scipy.stats import variation, tstd


class FeatureCalculator:
    def __init__(self, id, urm, itm_phis, usr_phis, is_user=True):
        self.urm = urm
        self.id = id
        self.is_user = is_user
        self.ratings = []
        self.num_ratings_urm = len(urm.nonzero()[0])
        self.num_itens = urm.getrow(0).shape[1]
        self.num_users = urm.getcol(0).shape[0]
        if is_user:
            for j in urm.getrow(id).nonzero()[1]:
                self.ratings.append(urm[self.id, j])
        else:
            for i in urm.getcol(id).nonzero()[0]:
                self.ratings.append(urm[i, self.id])
        self.item_phis = itm_phis
        self.user_phis = usr_phis
    
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

    
    def get_metrics(self):
        phi = self.phi_id()
        avg = 0
        avg_phi = 0
        std = tstd(self.ratings)
        gini = self.gini()
        var = variation(self.ratings)

        if len(self.ratings) != 0:
            avg = sum(self.ratings)/len(self.ratings)
            avg_phi = phi/len(self.ratings)
        
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
            "rating_std_dev": std,
            "phi": phi,
            "avg_phi": avg_phi,
        }
        return metrics_map



class UserFeatures(FeatureCalculator):
    def __init__(self, uid, urm, itm_phis, usr_phis):
        super().__init__(uid, urm, itm_phis, usr_phis, True)

class ItemFeatures(FeatureCalculator):
    def __init__(self, iid, urm, itm_phis, usr_phis):
        super().__init__(iid, urm, itm_phis, usr_phis, False)

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


# DS = "ml-100k"
DS = "ml-1M"
urm = pickle.load(open(f"../datasets/{DS}/urm.pk", 'rb'))

item_phis, user_phis = get_phis(urm)

num_users = urm.getcol(0).shape[0]
usr_out_str = f"uid, gini, variation, num_ratings, rating_avg, rating_std, phi, avg_phi\n"
for usr in range(num_users):
    uf = UserFeatures(usr, urm, item_phis, user_phis)
    m = uf.get_metrics()
    usr_out_str += f"{m['id']}, {m['gini']:.5f}, {m['variation']:.5f}, {m['num_ratings']:.5f}, {m['rating_avg']:.5f}, {m['rating_std_dev']:.5f}, {m['phi']:.5f}, {m['avg_phi']:.5f}\n"

with open(f"../datasets/{DS}/user_metrics.csv", 'w') as metrics_file:
    metrics_file.write(usr_out_str)

num_itens = urm.getrow(0).shape[1]
itm_out_str = f"iid, gini, variation, num_ratings, rating_avg, rating_std, phi, avg_phi\n"
for itm in range(num_itens):
    itmf = ItemFeatures(itm, urm, item_phis, user_phis)
    m = itmf.get_metrics()
    # print(m)
    itm_out_str += f"{m['id']}, {m['gini']:.5f}, {m['variation']:.5f}, {m['num_ratings']:.5f}, {m['rating_avg']:.5f}, {m['rating_std_dev']:.5f}, {m['phi']:.5f}, {m['avg_phi']:.5f}\n"

with open(f"../datasets/{DS}/item_metrics.csv", 'w') as metrics_file:
    metrics_file.write(itm_out_str)