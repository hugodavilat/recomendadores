#!/Users/hugo/Documents/recomendadores/explanatory_framework/venv/bin/python

import pickle
import pandas as pd
import numpy as np
# from scipy.sparse import csr_matrix
from scipy.stats import skew, kurtosis

class IndependentsFeaturesCalculator():
    # Constructor -> Precisa de uma urm que seja uma csr_matrix.
    def __init__(self, urm):
        self.urm = urm
        (self.U, self.I) = self.urm.get_shape()
        self.R = self.urm.count_nonzero()
        self._phi_i = {}
        self._pop = []
        self._user_long_tail_ratio = []
        self._calculate_phi()
        self._calculate_pop()
        self._calculate_long_tail()
    
    # Private
    def _calculate_phi(self):
        for j in range(self.I):
            self._phi_i[j] = self.urm.getcol(j).count_nonzero()/self.U
    
    def _calculate_pop(self):
        self.pop = []
        for i in range(self.U):
            popularity_bias_i = 0
            for j in self.urm.getrow(i).nonzero()[1]:
                non_zero_count = self.urm.getrow(i).count_nonzero()
                if non_zero_count != 0:
                    popularity_bias_i += (self._phi_i[j]/self.urm.getrow(i).count_nonzero())
            self._pop.append(popularity_bias_i)
    
    def _calculate_long_tail(self):
        distribution = []
        for j in range(self.I):
            distribution.append((self.urm.getcol(j).count_nonzero(), j))
        distribution.sort(key=lambda x: x[0])
        long_tails = set()
        views, i = 0, 0
        while views/self.R < 0.2:
            long_tails.add(distribution[i][1])
            views += distribution[i][0]
            i += 1
        for i in range(self.U):
            user_tails = 0
            user_total = 0
            for j in self.urm.getrow(i).nonzero()[1]:
                user_total += 1
                if j in long_tails:
                    user_tails += 1
            if user_total != 0:
                self._user_long_tail_ratio.append(user_tails/user_total)
            else:
                self._user_long_tail_ratio.append(0)

    
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

    def _avg(self, x):
        return np.average(x)
    
    def _std(self, x):
        return np.std(x)

    def _skew(self, x):
        return skew(x)
    
    def _kurtosis(self, x):
        return kurtosis(x)
    
    # Public
    def space_size(self):
        return self.U*self.I
    
    def shape(self):
        return self.U/self.I
    
    def density(self):
        return self.R/(self.U*self.I)
    
    def rpu(self):
        return self.R/self.U

    def rpi(self):
        return self.R/self.I
    
    def gini_i(self):
        x = []
        for j in range(self.I):
            col = self.urm.getcol(j)
            x.append(col.count_nonzero())
        return self._gini(x)
    
    def gini_u(self):
        x = []
        for i in range(self.U):
            row = self.urm.getrow(i)
            x.append(row.count_nonzero())
        return self._gini(x)
    
    def pop_avg(self):
        return self._avg(self._pop)
    
    def pop_std(self):
        return self._std(self._pop)
    
    def pop_skew(self):
        return self._skew(self._pop)
    
    def pop_kurtosis(self):
        return self._kurtosis(self._pop)

    def long_tail_avg(self):
        return self._avg(self._user_long_tail_ratio)
    
    def long_tail_std(self):
        return self._std(self._user_long_tail_ratio)
    
    def long_tail_skew(self):
        return self._skew(self._user_long_tail_ratio)
    
    def long_tail_kurtosis(self):
        return self._kurtosis(self._user_long_tail_ratio)


# urm = pickle.load(open("../datasets/ml-100k/URM/001_URM.pk", 'rb'))
# # df = pd.read_csv("../datasets/ml-100k/UIR/001_UIR.csv")
# # uir = [tuple(row) for row in df.values]
# c = IndependentsFeaturesCalculator(urm)

# print(c.space_size(), c.shape(), c.density(), c.rpu(), c.rpi(), c.gini_u(), c.gini_i(), c.pop_avg(), c.pop_std(), c.pop_skew(), c.pop_kurtosis(), c.long_tail_avg(), c.long_tail_std(), c.long_tail_skew(), c.long_tail_kurtosis())